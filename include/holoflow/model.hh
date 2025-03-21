#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "curaii/curaii.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/error.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/task.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace dh {

class ModelVisitor;

class ModelNode {
public:
  using Child = std::reference_wrapper<ModelNode>;

  ModelNode(const std::string &kind, const std::string &name,
            const json &params, int itens_id, int otens_id,
            cudaStream_t stream);

  virtual ~ModelNode() = default;

  /**
   * @brief Accepts a visitor for the Visitor pattern.
   *
   * @param visitor The visitor to process this node.
   */
  virtual void accept(ModelVisitor &visitor) = 0;

  /**
   * @brief Gets the mutable name of the node.
   * @return Reference to the node's name.
   */
  std::string &name();

  /**
   * @brief Gets the immutable name of the node.
   * @return Constant reference to the node's name.
   */
  const std::string &name() const;

  /**
   * @brief Gets the mutable kind of the node.
   * @return Reference to the node's kind.
   */
  std::string &kind();

  /**
   * @brief Gets the immutable kind of the node.
   * @return Constant reference to the node's kind.
   */
  const std::string &kind() const;

  /**
   * @brief Gets the mutable parameters of the node.
   * @return Reference to the JSON object representing node parameters.
   */
  json &params();

  /**
   * @brief Gets the immutable parameters of the node.
   * @return Constant reference to the JSON object representing node parameters.
   */
  const json &params() const;

  int &itens_id();

  const int &itens_id() const;

  int &otens_id();

  const int &otens_id() const;

  /**
   * @brief Adds a child node.
   * @param child Reference to the child node.
   */
  void add_child(ModelNode &child);

  /**
   * @brief Returns a modifiable view of child nodes.
   * @return A span of reference wrappers to child nodes.
   */
  std::span<Child> children();

  /**
   * @brief Returns a read-only view of child nodes.
   * @return A span of constant reference wrappers to child nodes.
   */
  std::span<const Child> children() const;

private:
  std::string name_; ///< The name of the node.
  std::string kind_; ///< The kind/type of the node.
  json params_;      ///< JSON object storing the node parameters.

  int itens_id_;        ///< Id of input tensor.
  int otens_id_;        ///< Id of output tensor.
  cudaStream_t stream_; ///< Cuda stream used by the node.

  std::vector<Child> children_; ///< List of child nodes.
};

class TaskNode : public ModelNode {
public:
  TaskNode(const std::string &kind, const std::string &name, const json &params,
           int itens_id, int otens_id, cudaStream_t stream,
           std::unique_ptr<Task> task, const TaskMeta &task_meta);

  void accept(ModelVisitor &visitor) override;

  Task &task();

  const Task &task() const;

  TaskMeta &task_meta();

  const TaskMeta &task_meta() const;

private:
  std::unique_ptr<Task> task_;
  TaskMeta task_meta_;
};

class AccumulatorNode : public ModelNode {
public:
  AccumulatorNode(const std::string &kind, const std::string &name,
                  const json &params, int itens_id, int otens_id,
                  cudaStream_t stream, std::unique_ptr<Accumulator> accumulator,
                  const AccumulatorMeta &accumulator_meta);

  void accept(ModelVisitor &visitor) override;

  Accumulator &accumulator();

  const Accumulator &accumulator() const;

  AccumulatorMeta &accumulator_meta();

  const AccumulatorMeta &accumulator_meta() const;

private:
  std::unique_ptr<Accumulator> accumulator_;
  AccumulatorMeta &accumulator_meta_;
};

class ModelVisitor {
public:
  virtual ~ModelVisitor() = default;

  virtual void visit(TaskNode &node) = 0;

  virtual void visit(AccumulatorNode &node) = 0;
};

class Model {
public:
  struct TensorSlot {
    TensorSlot(TensorMeta meta, unique_host_ptr<uint8_t> h = nullptr,
               unique_device_ptr<uint8_t> d = nullptr);

    TensorView view();

    TensorMeta meta;                        ///< Metadata for the tensor.
    unique_host_ptr<uint8_t> host_data;     ///< Owned host memory (if used).
    unique_device_ptr<uint8_t> device_data; ///< Owned device memory (if used).
  };

  enum class State {
    RUNNING,
    STOPPED,
  };

  static tl::expected<Model, Error>
  from_descriptor(const ModelDescriptor &descriptor);

  Model(std::vector<std::unique_ptr<ModelNode>> nodes,
        std::unordered_map<int, TensorSlot> tensors,
        std::vector<std::reference_wrapper<ModelNode>> pes, ModelNode &root);

  Accumulator *input(const std::string &name);
  const Accumulator *input(const std::string &name) const;

  Accumulator *output(const std::string &name);
  const Accumulator *output(const std::string &name) const;

  void start();
  void stop();

private:
  ModelNode &root_;
  std::vector<std::unique_ptr<ModelNode>> nodes_;
  std::unordered_map<int, TensorSlot> tensors_;
  std::vector<std::reference_wrapper<ModelNode>> pes_;

  State state_;
  std::atomic_bool stop_flag_;
};

} // namespace dh