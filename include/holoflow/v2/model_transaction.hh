#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <stack>
#include <string>
#include <tl/expected.hpp>
#include <unordered_map>

#include "bug_buster/bug_buster.hh"
#include "curaii/cuda_runtime.hh"
#include "holoflow/v2/error.hh"
#include "holoflow/v2/model.hh"

using json = nlohmann::json;

namespace dh::v2 {

class ModelTransaction {
public:
  enum class RemoveNodeBehavior {
    RemoveSubtree,
    OrphanChildren,
    ReparentChildren,
  };

  enum class DisconnectNodeBehavior {
    OrphanChild,
    ReparentChildren,
  };

  ModelTransaction &add_source(const std::string &name, const std::string &kind,
                               const json &params);

  ModelTransaction &add_sink(const std::string &name, const std::string &kind,
                             const json &params);

  ModelTransaction &add_task(const std::string &name, const std::string &kind,
                             const json &params);

  ModelTransaction &add_accumulator(const std::string &name,
                                    const std::string &kind,
                                    const json &params);

  ModelTransaction &update_node_params(const std::string &name,
                                       const json &params);

  ModelTransaction &remove_node(const std::string &name,
                                RemoveNodeBehavior behavior);

  ModelTransaction &connect(const std::string &parent_name,
                            const std::string &child_name);

  ModelTransaction &disconnect(const std::string &parent_name,
                               const std::string &child_name,
                               DisconnectNodeBehavior behavior);

  bool has_source(const std::string &name) const;
  bool has_sink(const std::string &name) const;
  bool has_task(const std::string &name) const;
  bool has_accumulator(const std::string &name) const;
  bool has_node(const std::string &name) const;

  tl::expected<void, Error> commit();

private:
  class AssignCudaStreamVisitor : public Model::NodeVisitor {
  public:
    AssignCudaStreamVisitor(std::vector<CudaStream> &streams,
                            std::vector<Error> &errors);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    std::stack<CudaStreamRef> stream_stack_;
    std::vector<CudaStream> &streams_;
    std::vector<Error> &errors_;
  };

  class TypeCheckVisitor : public Model::NodeVisitor {
  public:
    TypeCheckVisitor(
        std::vector<Error> &errors,
        const Model::TaskFactoryMap &task_factories_map,
        const Model::AccumulatorFactoryMap &accumulator_factories_map,
        const Model::SourceFactoryMap &source_factories_map,
        const Model::SinkFactoryMap &sink_factories_map);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    std::vector<Error> &errors_;
    const Model::TaskFactoryMap &task_factories_map_;
    const Model::AccumulatorFactoryMap &accumulator_factories_map_;
    const Model::SourceFactoryMap &source_factories_map_;
    const Model::SinkFactoryMap &sink_factories_map_;

    std::stack<TensorMeta> imetas_stack_;
  };

  class NonInlinedChildCheckVisitor : public Model::NodeVisitor {
  public:
    NonInlinedChildCheckVisitor(std::vector<Error> &errors);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    std::vector<Error> &errors_;
    std::stack<bool> seen_non_inlined_;
  };

  class AssignAccumulatorTensorsVisitor : public Model::NodeVisitor {
  public:
    AssignAccumulatorTensorsVisitor(int &next_id);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    int &next_id_;
    std::stack<std::reference_wrapper<Model::Node>> parents_stack_;
  };

  class AssignInlinedTaskTensorsVisitor : public Model::NodeVisitor {
  public:
    AssignInlinedTaskTensorsVisitor(int &next_id);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    int &next_id_;
    std::stack<std::reference_wrapper<Model::Node>> parents_stack_;
  };

  class AssignNonInlinedTaskTensorsVisitor : public Model::NodeVisitor {
  public:
    AssignNonInlinedTaskTensorsVisitor(int &next_id);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    int &next_id_;
    std::stack<std::reference_wrapper<Model::Node>> parents_stack_;
  };

  class CallFactoriesVisitor : public Model::NodeVisitor {
  public:
    CallFactoriesVisitor(
        const Model::TaskFactoryMap &task_factories_map,
        const Model::AccumulatorFactoryMap &accumulator_factories_map,
        const Model::SourceFactoryMap &source_factories_map,
        const Model::SinkFactoryMap &sink_factories_map,
        std::vector<Error> &errors);

    void visit(Model::TaskNode &node) override;
    void visit(Model::AccumulatorNode &node) override;
    void visit(Model::SourceNode &node) override;
    void visit(Model::SinkNode &node) override;

  private:
    const Model::TaskFactoryMap &task_factories_map_;
    const Model::AccumulatorFactoryMap &accumulator_factories_map_;
    const Model::SourceFactoryMap &source_factories_map_;
    const Model::SinkFactoryMap &sink_factories_map_;
    std::vector<Error> &errors_;
  };

  explicit ModelTransaction(Model &model);

  Model::Node *get_root();

  void validate_orphan_nodes();
  void validate_childless_nodes();
  void validate_source_nodes();
  void validate_sink_nodes();
  void assign_cuda_streams();
  void perform_type_checking();
  void perform_single_inlined_child_checking();
  void perform_non_inlined_child_checking();
  void reset_non_accumulator_tensors();
  void assign_accumulator_tensors();
  void assign_inlined_task_tensors();
  void assign_non_inlined_task_tensors();
  void call_factories();

  Model &model_;
  std::vector<Error> errors_;

  friend class Model;
};

} // namespace dh::v2