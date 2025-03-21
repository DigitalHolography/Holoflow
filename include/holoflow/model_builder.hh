#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "holoflow/accumulator.hh"
#include "holoflow/model.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class ModelBuilderVisitor;

class ModelNodeBuilder {
public:
  using Child = std::reference_wrapper<ModelNodeBuilder>;

  virtual ~ModelNodeBuilder() = default;

  const std::string &name() const;

  const std::string &kind() const;

  const json &params() const;

  int itens_id() const;

  int otens_id() const;

  cudaStream_t stream() const;

  std::span<Child> children();

  std::span<const Child> children() const;

  std::optional<int> get_itens_id() const;

  std::optional<int> get_otens_id() const;

  ModelNodeBuilder &set_name(const std::string &name);

  ModelNodeBuilder &set_kind(const std::string &kind);

  ModelNodeBuilder &set_params(const json &params);
  ModelNodeBuilder &set_itens_id(int id);

  ModelNodeBuilder &set_otens_id(int id);

  ModelNodeBuilder &set_stream(cudaStream_t stream);

  ModelNodeBuilder &add_child(ModelNodeBuilder &child);
  virtual void accept(ModelBuilderVisitor &visitor) = 0;

  virtual std::unique_ptr<ModelNode> build() = 0;

protected:
  std::optional<std::string> name_;
  std::optional<std::string> kind_;
  std::optional<json> params_;
  std::optional<int> itens_id_;
  std::optional<int> otens_id_;
  std::optional<cudaStream_t> stream_;
  std::vector<std::reference_wrapper<ModelNodeBuilder>> children_;
};

class TaskNodeBuilder : public ModelNodeBuilder {
public:
  Task &task() const;

  const TaskMeta &task_meta() const;

  TaskNodeBuilder &set_task(std::unique_ptr<Task> task);

  TaskNodeBuilder &set_task_meta(const TaskMeta &task_meta);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Task> task_;
  std::optional<TaskMeta> task_meta_;
};

class AccumulatorNodeBuilder : public ModelNodeBuilder {
public:
  Accumulator &accumulator() const;

  const AccumulatorMeta &accumulator_meta() const;

  AccumulatorNodeBuilder &
  set_accumulator(std::unique_ptr<Accumulator> accumulator);

  AccumulatorNodeBuilder &
  set_accumulator_meta(const AccumulatorMeta &accumulator_meta);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Accumulator> accumulator_;
  std::optional<AccumulatorMeta> accumulator_meta_;
};

class ModelBuilderVisitor {
public:
  virtual ~ModelBuilderVisitor() = default;

  virtual void visit(TaskNodeBuilder &node) = 0;

  virtual void visit(AccumulatorNodeBuilder &node) = 0;
};

class ModelBuilder {
public:
  tl::expected<std::unique_ptr<Model>, Error>
  build(const ModelDescriptor &descriptor, const TensorMeta &imeta);
};

} // namespace dh