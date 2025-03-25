#include "holoflow/model_builder.hh"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "holoflow/holoflow.hh"

namespace dh {

// ==========================================================================
//                     ModelNodeBuilder Implementation
// ==========================================================================

const std::string &ModelNodeBuilder::name() const {
  assert(name_);
  return *name_;
}

const std::string &ModelNodeBuilder::kind() const {
  assert(kind_);
  return *kind_;
}

const json &ModelNodeBuilder::params() const {
  assert(params_);
  return *params_;
}

cudaStream_t ModelNodeBuilder::stream() const {
  assert(stream_);
  return *stream_;
}

std::span<ModelNodeBuilder::Child> ModelNodeBuilder::children() {
  return children_;
}

std::span<const ModelNodeBuilder::Child> ModelNodeBuilder::children() const {
  return children_;
}

void ModelNodeBuilder::set_name(const std::string &name) { name_ = name; }

void ModelNodeBuilder::set_kind(const std::string &kind) { kind_ = kind; }

void ModelNodeBuilder::set_params(const json &params) { params_ = params; }

void ModelNodeBuilder::set_stream(cudaStream_t stream) { stream_ = stream; }

void ModelNodeBuilder::add_child(ModelNodeBuilder &child) {
  children_.push_back(child);
}

// ==========================================================================
//                     TaskNodeBuilder Implementation
// ==========================================================================

int TaskNodeBuilder::itens_id() const {
  assert(itens_id_);
  return *itens_id_;
}

int TaskNodeBuilder::otens_id() const {
  assert(otens_id_);
  return *otens_id_;
}

Task &TaskNodeBuilder::task() const {
  assert(task_);
  return *task_;
}

const TaskMeta &TaskNodeBuilder::task_meta() const {
  assert(task_meta_);
  return *task_meta_;
}

std::optional<int> TaskNodeBuilder::get_itens_id() const { return itens_id_; }

std::optional<int> TaskNodeBuilder::get_otens_id() const { return otens_id_; }

void TaskNodeBuilder::set_task(std::unique_ptr<Task> task) {
  task_ = std::move(task);
}

void TaskNodeBuilder::set_task_meta(const TaskMeta &task_meta) {
  task_meta_ = task_meta;
}

void TaskNodeBuilder::set_itens_id(int id) { itens_id_ = id; }

void TaskNodeBuilder::set_otens_id(int id) { otens_id_ = id; }

std::unique_ptr<ModelNode> TaskNodeBuilder::build() {
  assert(name_);
  assert(kind_);
  assert(params_);
  assert(itens_id_);
  assert(otens_id_);
  assert(stream_);
  assert(task_);
  assert(task_meta_);

  return std::make_unique<TaskNode>(*kind_, *name_, *params_, *itens_id_,
                                    *otens_id_, *stream_, std::move(task_),
                                    *task_meta_);
}

void TaskNodeBuilder::accept(ModelBuilderVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     AccumulatorNodeBuilder Implementation
// ==========================================================================

int AccumulatorNodeBuilder::itens_id() const {
  assert(itens_id_);
  return *itens_id_;
}

int AccumulatorNodeBuilder::otens_id() const {
  assert(otens_id_);
  return *otens_id_;
}

Accumulator &AccumulatorNodeBuilder::accumulator() const {
  assert(accumulator_);
  return *accumulator_;
}

const AccumulatorMeta &AccumulatorNodeBuilder::accumulator_meta() const {
  assert(accumulator_meta_);
  return *accumulator_meta_;
}

std::optional<int> AccumulatorNodeBuilder::get_itens_id() const {
  return itens_id_;
}

std::optional<int> AccumulatorNodeBuilder::get_otens_id() const {
  return otens_id_;
}

void AccumulatorNodeBuilder::set_accumulator(
    std::unique_ptr<Accumulator> accumulator) {
  accumulator_ = std::move(accumulator);
}

void AccumulatorNodeBuilder::set_accumulator_meta(
    const AccumulatorMeta &accumulator_meta) {
  accumulator_meta_ = accumulator_meta;
}

void AccumulatorNodeBuilder::set_itens_id(int id) { itens_id_ = id; }

void AccumulatorNodeBuilder::set_otens_id(int id) { otens_id_ = id; }

std::unique_ptr<ModelNode> AccumulatorNodeBuilder::build() {
  assert(name_);
  assert(kind_);
  assert(params_);
  assert(itens_id_);
  assert(otens_id_);
  assert(stream_);
  assert(accumulator_);
  assert(accumulator_meta_);

  return std::make_unique<AccumulatorNode>(
      *kind_, *name_, *params_, *itens_id_, *otens_id_, *stream_,
      std::move(accumulator_), *accumulator_meta_);
}

void AccumulatorNodeBuilder::accept(ModelBuilderVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     SourceNodeBuilder Implementation
// ==========================================================================

int SourceNodeBuilder::otens_id() const {
  assert(otens_id_);
  return *otens_id_;
}

Source &SourceNodeBuilder::source() const {
  assert(source_);
  return *source_;
}

const SourceMeta &SourceNodeBuilder::source_meta() const {
  assert(source_meta_);
  return *source_meta_;
}

std::optional<int> SourceNodeBuilder::get_otens_id() const { return otens_id_; }

void SourceNodeBuilder::set_source(std::unique_ptr<Source> source) {
  source_ = std::move(source);
}

void SourceNodeBuilder::set_source_meta(const SourceMeta &source_meta) {
  source_meta_ = source_meta;
}

void SourceNodeBuilder::set_otens_id(int id) { otens_id_ = id; }

std::unique_ptr<ModelNode> SourceNodeBuilder::build() {
  assert(name_);
  assert(kind_);
  assert(params_);
  assert(otens_id_);
  assert(stream_);
  assert(source_);
  assert(source_meta_);

  return std::make_unique<SourceNode>(*kind_, *name_, *params_, *otens_id_,
                                      *stream_, std::move(source_),
                                      *source_meta_);
}

void SourceNodeBuilder::accept(ModelBuilderVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     SinkNodeBuilder Implementation
// ==========================================================================

int SinkNodeBuilder::itens_id() const {
  assert(itens_id_);
  return *itens_id_;
}

Sink &SinkNodeBuilder::sink() const {
  assert(sink_);
  return *sink_;
}

const SinkMeta &SinkNodeBuilder::sink_meta() const {
  assert(sink_meta_);
  return *sink_meta_;
}

std::optional<int> SinkNodeBuilder::get_itens_id() const { return itens_id_; }

void SinkNodeBuilder::set_sink(std::unique_ptr<Sink> sink) {
  sink_ = std::move(sink);
}

void SinkNodeBuilder::set_sink_meta(const SinkMeta &sink_meta) {
  sink_meta_ = sink_meta;
}

void SinkNodeBuilder::set_itens_id(int id) { itens_id_ = id; }

std::unique_ptr<ModelNode> SinkNodeBuilder::build() {
  assert(name_);
  assert(kind_);
  assert(params_);
  assert(itens_id_);
  assert(stream_);
  assert(sink_);
  assert(sink_meta_);

  return std::make_unique<SinkNode>(*kind_, *name_, *params_, *itens_id_,
                                    *stream_, std::move(sink_), *sink_meta_);
}

void SinkNodeBuilder::accept(ModelBuilderVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     ModelBuilder Implementation
// ==========================================================================

namespace {

class CreateBuilderNodes : public ModelDescriptorVisitor {
public:
  void visit(TaskDescriptorNode &node) {
    holoflow_logger()->trace("[CreateBuilderNodes] visiting: {}", node.name());

    auto builder_node = std::make_unique<TaskNodeBuilder>();
    builder_node->set_kind(node.kind());
    builder_node->set_name(node.name());
    builder_node->set_params(node.params());

    for (auto child : node.children()) {
      child.get().accept(*this);
      builder_node->add_child(*root_);
    }

    root_ = &*builder_node;
    nodes_.push_back(std::move(builder_node));
  }

  void visit(AccumulatorDescriptorNode &node) {
    holoflow_logger()->trace("[CreateBuilderNodes] visiting: {}", node.name());

    auto builder_node = std::make_unique<AccumulatorNodeBuilder>();
    builder_node->set_kind(node.kind());
    builder_node->set_name(node.name());
    builder_node->set_params(node.params());

    for (auto child : node.children()) {
      child.get().accept(*this);
      builder_node->add_child(*root_);
    }

    root_ = &*builder_node;
    nodes_.push_back(std::move(builder_node));
  }

  void visit(SourceDescriptorNode &node) {
    holoflow_logger()->trace("[CreateBuilderNodes] visiting: {}", node.name());

    auto builder_node = std::make_unique<SourceNodeBuilder>();
    builder_node->set_kind(node.kind());
    builder_node->set_name(node.name());
    builder_node->set_params(node.params());

    for (auto child : node.children()) {
      child.get().accept(*this);
      builder_node->add_child(*root_);
    }

    root_ = &*builder_node;
    nodes_.push_back(std::move(builder_node));
  }

  void visit(SinkDescriptorNode &node) {
    holoflow_logger()->trace("[CreateBuilderNodes] visiting: {}", node.name());

    auto builder_node = std::make_unique<SinkNodeBuilder>();
    builder_node->set_kind(node.kind());
    builder_node->set_name(node.name());
    builder_node->set_params(node.params());

    root_ = &*builder_node;
    nodes_.push_back(std::move(builder_node));
    assert(node.children().empty());
  }

  std::pair<std::reference_wrapper<ModelNodeBuilder>,
            std::vector<std::unique_ptr<ModelNodeBuilder>>>
  result() {
    return std::make_pair(std::ref(*root_), std::move(nodes_));
  }

private:
  ModelNodeBuilder *root_;
  std::vector<std::unique_ptr<ModelNodeBuilder>> nodes_;
};

class AssignStreams : public ModelBuilderVisitor {
  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[AssignStreams] visiting: {}", node.name());

    assert(!streams_stack_.empty());
    node.set_stream(streams_stack_.top());

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[AssignStreams] visiting: {}", node.name());

    auto stream = make_unique_cuda_stream();
    node.set_stream(stream.get());
    streams_stack_.push(stream.get());
    streams_.push_back(std::move(stream));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }

    streams_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[AssignStreams] visiting: {}", node.name());

    auto stream = make_unique_cuda_stream();
    node.set_stream(stream.get());
    streams_stack_.push(stream.get());
    streams_.push_back(std::move(stream));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }

    streams_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[AssignStreams] visiting: {}", node.name());

    assert(!streams_stack_.empty());
    node.set_stream(streams_stack_.top());
    assert(node.children().empty());
  }

public:
  std::vector<unique_cuda_stream> result() { return std::move(streams_); }

private:
  std::vector<unique_cuda_stream> streams_;
  std::stack<cudaStream_t> streams_stack_;
};

class TypeCheck : public ModelBuilderVisitor {
public:
  /**
   * @brief Map from kind to task factory.
   */
  using TaskFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<TaskFactory>>;

  /**
   * @brief Map from kind to accumulator factory.
   */
  using AccumulatorFactoryMap =
      std::unordered_map<std::string,
                         std::reference_wrapper<AccumulatorFactory>>;

  /**
   * @brief Map from kind to source factory.
   */
  using SourceFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SourceFactory>>;

  /**
   * @brief Map from kind to sink factory.
   */
  using SinkFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SinkFactory>>;

  TypeCheck(const TaskFactoryMap &task_factories_map,
            const AccumulatorFactoryMap &accumulator_factories_map,
            const SourceFactoryMap &source_factories_map,
            const SinkFactoryMap &sink_factories_map)
      : task_factories_map_(task_factories_map),
        accumulator_factories_map_(accumulator_factories_map),
        source_factories_map_(source_factories_map),
        sink_factories_map_(sink_factories_map), result_(true) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[TypeCheck] visiting: {}", node.name());

    assert(!imetas_stack_.empty());
    assert(task_factories_map_.contains(node.kind()));

    auto &factory = task_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      holoflow_logger()->warn("Type checking failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_task_meta(result.value());
    imetas_stack_.push(result.value().ometa());
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    imetas_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[TypeCheck] visiting: {}", node.name());

    assert(!imetas_stack_.empty());
    assert(accumulator_factories_map_.contains(node.kind()));

    auto &factory = accumulator_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      holoflow_logger()->warn("Type checking failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_accumulator_meta(result.value());
    imetas_stack_.push(result.value().ometa());
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    imetas_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[TypeCheck] visiting: {}", node.name());

    assert(imetas_stack_.empty());
    assert(source_factories_map_.contains(node.kind()));

    auto &factory = source_factories_map_.at(node.kind());
    auto result = factory.get().type_check(node.params());
    if (!result) {
      holoflow_logger()->warn("Type checking failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_source_meta(result.value());
    imetas_stack_.push(result.value().ometa());
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    imetas_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[TypeCheck] visiting: {}", node.name());

    assert(!imetas_stack_.empty());
    assert(sink_factories_map_.contains(node.kind()));

    auto &factory = sink_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      holoflow_logger()->warn("Type checking failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_sink_meta(result.value());
    assert(node.children().empty());
  }

  bool result() { return result_; }

private:
  const TaskFactoryMap &task_factories_map_;
  const AccumulatorFactoryMap &accumulator_factories_map_;
  const SourceFactoryMap &source_factories_map_;
  const SinkFactoryMap &sink_factories_map_;

  bool result_;
  std::stack<TensorMeta> imetas_stack_;
};

bool is_inlined(ModelNodeBuilder &node) {
  if (auto *task = dynamic_cast<TaskNodeBuilder *>(&node)) {
    return task->task_meta().inlined();
  }

  return false;
}

class SingleInlinedChildCheck : public ModelBuilderVisitor {
public:
  SingleInlinedChildCheck() : result_(true) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[SingleInlinedChildCheck] visiting: {}",
                             node.name());

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      holoflow_logger()->warn("Single inlined check failed at node: {}",
                              node.name());
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[SingleInlinedChildCheck] visiting: {}",
                             node.name());

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      holoflow_logger()->warn("Single inlined check failed at node: {}",
                              node.name());
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[SingleInlinedChildCheck] visiting: {}",
                             node.name());

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      holoflow_logger()->warn("Single inlined check failed at node: {}",
                              node.name());
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[SingleInlinedChildCheck] visiting: {}",
                             node.name());

    assert(node.children().empty());
  }

  bool result() { return result_; }

private:
  bool result_;
};

class NonInlinedTaskCheck : public ModelBuilderVisitor {
public:
  NonInlinedTaskCheck() : result_(true) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[NonInlinedTaskCheck] visiting: {}", node.name());

    assert(!seen_non_inlined_stack_.empty());

    if (!is_inlined(node)) {
      seen_non_inlined_stack_.push(true);
    } else {
      seen_non_inlined_stack_.push(seen_non_inlined_stack_.top());
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    seen_non_inlined_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[NonInlinedTaskCheck] visiting: {}", node.name());

    assert(!seen_non_inlined_stack_.empty());

    if (!seen_non_inlined_stack_.top()) {
      holoflow_logger()->warn("Non inlined task check failed at node: {}",
                              node.name());
      result_ = false;
      return;
    }

    seen_non_inlined_stack_.push(false);
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    seen_non_inlined_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[NonInlinedTaskCheck] visiting: {}", node.name());

    assert(seen_non_inlined_stack_.empty());

    seen_non_inlined_stack_.push(true);
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    seen_non_inlined_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[NonInlinedTaskCheck] visiting: {}", node.name());

    assert(node.children().empty());
  }

  bool result() { return result_; }

private:
  bool result_;
  std::stack<bool> seen_non_inlined_stack_;
};

class AssignAccumulatorsTensors : public ModelBuilderVisitor {
public:
  AssignAccumulatorsTensors(int &next_id) : next_id_(next_id) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[AssignAccumulatorsTensors] visiting: {}",
                             node.name());

    assert(!parents_stack_.empty());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[AssignAccumulatorsTensors] visiting: {}",
                             node.name());

    assert(!parents_stack_.empty());

    // Set current
    node.set_itens_id(next_id_++);
    node.set_otens_id(next_id_++);
    holoflow_logger()->info("{} <= itens_id: {}, otens_id: {}", node.name(),
                            node.itens_id(), node.otens_id());

    // Set parent
    auto parent = parents_stack_.top();
    if (auto *task = dynamic_cast<TaskNodeBuilder *>(&parent.get())) {
      task->set_otens_id(node.itens_id());
      holoflow_logger()->info("{} <= otens_id: {}", task->name(),
                              task->otens_id());
    } else if (auto *source =
                   dynamic_cast<SourceNodeBuilder *>(&parent.get())) {
      source->set_otens_id(node.itens_id());
      holoflow_logger()->info("{} <= otens_id: {}", source->name(),
                              source->otens_id());
    } else {
      holoflow_logger()->critical(
          "Expected parent to be TaskBuilderNode or SourceBuilderNode");
      std::exit(1);
    }

    // Set children
    for (auto child : node.children()) {
      if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
        task->set_itens_id(node.otens_id());
        holoflow_logger()->info("{} <= itens_id: {}", task->name(),
                                task->itens_id());
      } else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
        sink->set_itens_id(node.otens_id());
        holoflow_logger()->info("{} <= itens_id: {}", sink->name(),
                                sink->itens_id());
      } else {
        holoflow_logger()->critical(
            "Expected child to be TaskBuilderNode or SinkBuilderNode");
        std::exit(1);
      }
    }

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[AssignAccumulatorsTensors] visiting: {}",
                             node.name());

    assert(parents_stack_.empty());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[AssignAccumulatorsTensors] visiting: {}",
                             node.name());

    assert(!parents_stack_.empty());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

private:
  int &next_id_;
  std::stack<std::reference_wrapper<ModelNodeBuilder>> parents_stack_;
};

class AssignInlinedTaskTensors : public ModelBuilderVisitor {
public:
  AssignInlinedTaskTensors(int &next_id) : next_id_(next_id) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[AssignInlinedTaskTensors] visiting: {}",
                             node.name());

    assert(!parents_stack_.empty());

    // Forward propagation (otens_id <= itens_id)
    if (node.task_meta().inlined() && node.get_itens_id() &&
        node.get_otens_id() != node.get_itens_id()) {

      // Set current
      assert(!node.get_otens_id());
      node.set_otens_id(node.itens_id());
      holoflow_logger()->info("{} <= otens_id: {}", node.name(),
                              node.otens_id());

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", task->name(),
                                  task->itens_id());
        } else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", sink->name(),
                                  sink->itens_id());
        } else {
          holoflow_logger()->critical(
              "Expected child to be TaskBuilderNode or SinkBuilderNode");
          std::exit(1);
        }
      }
    }

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();

    // Backward propagation (itens_id <= otens_id)
    if (node.task_meta().inlined() && node.get_otens_id() &&
        node.get_itens_id() != node.get_otens_id()) {

      // Set current
      assert(!node.get_itens_id());
      node.set_itens_id(node.otens_id());
      holoflow_logger()->info("{} <= itens_id: {}", node.name(),
                              node.itens_id());

      // Set parent
      assert(!parents_stack_.empty());
      auto parent = parents_stack_.top();
      if (auto *task = dynamic_cast<TaskNodeBuilder *>(&parent.get())) {
        task->set_otens_id(node.itens_id());
        holoflow_logger()->info("{} <= otens_id: {}", task->name(),
                                task->otens_id());
      } else if (auto *source =
                     dynamic_cast<SourceNodeBuilder *>(&parent.get())) {
        source->set_otens_id(node.itens_id());
        holoflow_logger()->info("{} <= otens_id: {}", source->name(),
                                source->otens_id());
      } else {
        holoflow_logger()->critical(
            "Expected parent to be TaskBuilderNode or SourceBuilderNode");
        std::exit(1);
      }
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[AssignInlinedTaskTensors] visiting: {}",
                             node.name());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[AssignInlinedTaskTensors] visiting: {}",
                             node.name());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[AssignInlinedTaskTensors] visiting: {}",
                             node.name());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

private:
  int &next_id_;
  std::stack<std::reference_wrapper<ModelNodeBuilder>> parents_stack_;
};

class AssignNonInlinedTaskTensors : public ModelBuilderVisitor {
public:
  AssignNonInlinedTaskTensors(int &next_id) : next_id_(next_id) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[AssignNonInlinedTaskTensors] visiting: {}",
                             node.name());

    if (!node.task_meta().inlined() && !node.get_otens_id()) {
      // Set current
      node.set_otens_id(next_id_++);
      holoflow_logger()->info("{} <= otens_id: {}", node.name(),
                              node.otens_id());

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", task->name(),
                                  task->itens_id());
        } else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", sink->name(),
                                  sink->itens_id());
        } else {
          holoflow_logger()->critical(
              "Expected child to be TaskBuilderNode or SinkBuilderNode");
          std::exit(1);
        }
      }
    }

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[AssignNonInlinedTaskTensors] visiting: {}",
                             node.name());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[AssignNonInlinedTaskTensors] visiting: {}",
                             node.name());

    if (!node.get_otens_id()) {
      // Set current
      node.set_otens_id(next_id_++);
      holoflow_logger()->info("{} <= otens_id: {}", node.name(),
                              node.otens_id());

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", task->name(),
                                  task->itens_id());
        } else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          holoflow_logger()->info("{} <= itens_id: {}", sink->name(),
                                  sink->itens_id());
        } else {
          holoflow_logger()->critical(
              "Expected child to be TaskBuilderNode or SinkBuilderNode");
          std::exit(1);
        }
      }
    }

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[AssignNonInlinedTaskTensors] visiting: {}",
                             node.name());

    assert(node.children().empty());
  }

private:
  int &next_id_;
  std::stack<std::reference_wrapper<ModelNodeBuilder>> parents_stack_;
};

class CallFactories : public ModelBuilderVisitor {
public:
  /**
   * @brief Map from kind to task factory.
   */
  using TaskFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<TaskFactory>>;

  /**
   * @brief Map from kind to accumulator factory.
   */
  using AccumulatorFactoryMap =
      std::unordered_map<std::string,
                         std::reference_wrapper<AccumulatorFactory>>;

  /**
   * @brief Map from kind to source factory.
   */
  using SourceFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SourceFactory>>;

  /**
   * @brief Map from kind to sink factory.
   */
  using SinkFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SinkFactory>>;

  CallFactories(const TaskFactoryMap &task_factories_map,
                const AccumulatorFactoryMap &accumulator_factories_map,
                const SourceFactoryMap &source_factories_map,
                const SinkFactoryMap &sink_factories_map)
      : task_factories_map_(task_factories_map),
        accumulator_factories_map_(accumulator_factories_map),
        source_factories_map_(source_factories_map),
        sink_factories_map_(sink_factories_map), result_(true) {}

  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[CallFactories] visiting: {}", node.name());

    assert(task_factories_map_.contains(node.kind()));

    auto &factory = task_factories_map_.at(node.kind());
    auto result = factory.get().create(node.task_meta().imeta(), node.params(),
                                       node.stream());
    if (!result) {
      holoflow_logger()->warn("Factory call failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_task(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[CallFactories] visiting: {}", node.name());

    assert(accumulator_factories_map_.contains(node.kind()));

    auto &factory = accumulator_factories_map_.at(node.kind());
    auto result = factory.get().create(node.accumulator_meta().imeta(),
                                       node.params(), node.stream());
    if (!result) {
      holoflow_logger()->warn("Factory call failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_accumulator(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[CallFactories] visiting: {}", node.name());

    assert(source_factories_map_.contains(node.kind()));

    auto &factory = source_factories_map_.at(node.kind());
    auto result = factory.get().create(node.params(), node.stream());
    if (!result) {
      holoflow_logger()->warn("Factory call failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_source(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[CallFactories] visiting: {}", node.name());

    assert(sink_factories_map_.contains(node.kind()));

    auto &factory = sink_factories_map_.at(node.kind());
    auto result = factory.get().create(node.sink_meta().imeta(), node.params(),
                                       node.stream());
    if (!result) {
      holoflow_logger()->warn("Factory call failed at node: {}", node.name());
      result_ = false;
      return;
    }

    node.set_sink(std::move(result.value()));

    assert(node.children().empty());
  }

  bool result() { return result_; }

private:
  const TaskFactoryMap &task_factories_map_;
  const AccumulatorFactoryMap &accumulator_factories_map_;
  const SourceFactoryMap &source_factories_map_;
  const SinkFactoryMap &sink_factories_map_;

  bool result_;
};

class BuildModelNodes : public ModelBuilderVisitor {
public:
  void visit(TaskNodeBuilder &node) {
    holoflow_logger()->trace("[BuildModelNodes] visiting: {}", node.name());

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(AccumulatorNodeBuilder &node) {
    holoflow_logger()->trace("[BuildModelNodes] visiting: {}", node.name());

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(SourceNodeBuilder &node) {
    holoflow_logger()->trace("[BuildModelNodes] visiting: {}", node.name());

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(SinkNodeBuilder &node) {
    holoflow_logger()->trace("[BuildModelNodes] visiting: {}", node.name());

    auto model_node = node.build();

    assert(node.children().empty());

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  std::pair<std::reference_wrapper<ModelNode>,
            std::vector<std::unique_ptr<ModelNode>>>
  result() {
    return std::make_pair(std::ref(*root_), std::move(nodes_));
  }

private:
  ModelNode *root_;
  std::vector<std::unique_ptr<ModelNode>> nodes_;
};

class GetTensorSlots : public ModelVisitor {
public:
  void visit(TaskNode &node) override {
    holoflow_logger()->trace("[GetTensorSlots] visiting: {}", node.name());

    if (!tensors_map_.contains(node.itens_id())) {
      tensors_map_.emplace(node.itens_id(),
                           Model::TensorSlot(node.task_meta().imeta()));
    }

    if (!tensors_map_.contains(node.otens_id())) {
      tensors_map_.emplace(node.otens_id(),
                           Model::TensorSlot(node.task_meta().ometa()));
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    holoflow_logger()->trace("[GetTensorSlots] visiting: {}", node.name());

    if (!tensors_map_.contains(node.itens_id())) {
      tensors_map_.emplace(node.itens_id(),
                           Model::TensorSlot(node.accumulator_meta().imeta()));
    }

    if (!tensors_map_.contains(node.otens_id())) {
      tensors_map_.emplace(node.otens_id(),
                           Model::TensorSlot(node.accumulator_meta().ometa()));
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) {
    holoflow_logger()->trace("[GetTensorSlots] visiting: {}", node.name());

    if (!tensors_map_.contains(node.otens_id())) {
      tensors_map_.emplace(node.otens_id(),
                           Model::TensorSlot(node.source_meta().ometa()));
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    holoflow_logger()->trace("[GetTensorSlots] visiting: {}", node.name());

    if (!tensors_map_.contains(node.itens_id())) {
      tensors_map_.emplace(node.itens_id(),
                           Model::TensorSlot(node.sink_meta().imeta()));
    }

    assert(node.children().empty());
  }

  std::unordered_map<int, Model::TensorSlot> result() {
    return std::move(tensors_map_);
  }

private:
  std::unordered_map<int, Model::TensorSlot> tensors_map_;
};

class IsAccumulatorTensor : public ModelVisitor {
public:
  IsAccumulatorTensor(int tens_id) : tens_id_(tens_id), result_(false) {}

  void visit(TaskNode &node) override {
    holoflow_logger()->trace("[IsAccumulatorTensor] visiting: {}", node.name());

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    holoflow_logger()->trace("[IsAccumulatorTensor] visiting: {}", node.name());

    if (node.itens_id() == tens_id_ || node.otens_id() == tens_id_) {
      result_ = true;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) {
    holoflow_logger()->trace("[IsAccumulatorTensor] visiting: {}", node.name());

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    holoflow_logger()->trace("[IsAccumulatorTensor] visiting: {}", node.name());

    assert(node.children().empty());
  }

  bool result() { return result_; }

private:
  int tens_id_;
  bool result_;
};

class GetPES : public ModelVisitor {
public:
  void visit(TaskNode &node) override {
    holoflow_logger()->trace("[GetPES] visiting: {}", node.name());

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    holoflow_logger()->trace("[GetPES] visiting: {}", node.name());

    pes_.push_back(std::ref(node));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) {
    holoflow_logger()->trace("[GetPES] visiting: {}", node.name());

    pes_.push_back(std::ref(node));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    holoflow_logger()->trace("[GetPES] visiting: {}", node.name());

    assert(node.children().empty());
  }

  std::vector<std::reference_wrapper<ModelNode>> result() {
    return std::move(pes_);
  }

private:
  std::vector<std::reference_wrapper<ModelNode>> pes_;
};

} // namespace

tl::expected<std::unique_ptr<Model>, Error>
ModelBuilder::build(const ModelDescriptor &descriptor) {
  holoflow_logger()->info("Building model");
  assert(descriptor.root());

  holoflow_logger()->info("Creating builder nodes");
  CreateBuilderNodes create_builder_nodes;
  descriptor.root()->accept(create_builder_nodes);
  auto [builder_root, builder_nodes] = create_builder_nodes.result();

  holoflow_logger()->info("Assigning cuda streams");
  AssignStreams assign_streams;
  builder_root.get().accept(assign_streams);
  auto streams = assign_streams.result();

  holoflow_logger()->info("Performing type check");
  TypeCheck type_check(
      descriptor.task_factories(), descriptor.accumulator_factories(),
      descriptor.source_factories(), descriptor.sink_factories());
  builder_root.get().accept(type_check);
  assert(type_check.result());

  holoflow_logger()->info("Performing single inlined child check");
  SingleInlinedChildCheck single_inlined_child_check;
  builder_root.get().accept(single_inlined_child_check);
  assert(single_inlined_child_check.result());

  holoflow_logger()->info("Performing non inlined child check");
  NonInlinedTaskCheck non_inlined_child_check;
  builder_root.get().accept(non_inlined_child_check);
  assert(non_inlined_child_check.result());

  int next_tens_id = 0;

  holoflow_logger()->info("Assigning accumulators tensors");
  AssignAccumulatorsTensors assign_accumulators_tensors(next_tens_id);
  builder_root.get().accept(assign_accumulators_tensors);

  holoflow_logger()->info("Assigning inlined tasks tensors (first pass)");
  AssignInlinedTaskTensors assign_inlined_tasks_tensors1(next_tens_id);
  builder_root.get().accept(assign_inlined_tasks_tensors1);

  holoflow_logger()->info("Assigning non inlined tasks tensors");
  AssignNonInlinedTaskTensors assign_non_inlined_tasks_tensor(next_tens_id);
  builder_root.get().accept(assign_non_inlined_tasks_tensor);

  holoflow_logger()->info("Assigning inlined tasks tensors (second pass)");
  AssignInlinedTaskTensors assign_inlined_tasks_tensors2(next_tens_id);
  builder_root.get().accept(assign_inlined_tasks_tensors2);

  holoflow_logger()->info("Calling factories");
  CallFactories call_factories(
      descriptor.task_factories(), descriptor.accumulator_factories(),
      descriptor.source_factories(), descriptor.sink_factories());
  builder_root.get().accept(call_factories);
  assert(call_factories.result());

  holoflow_logger()->info("Building model nodes");
  BuildModelNodes build_model_nodes;
  builder_root.get().accept(build_model_nodes);
  auto [model_root, model_nodes] = build_model_nodes.result();

  holoflow_logger()->info("Getting tensors slots");
  GetTensorSlots get_tensor_slots;
  model_root.get().accept(get_tensor_slots);
  auto tensor_slots = get_tensor_slots.result();

  holoflow_logger()->info("Allocating non accumulator tensors");
  for (auto &[id, slot] : tensor_slots) {
    IsAccumulatorTensor is_accumulator_tensor(id);
    model_root.get().accept(is_accumulator_tensor);
    if (is_accumulator_tensor.result()) {
      holoflow_logger()->info("tens_id: {} is an accumulator tensor", id);
      continue;
    }

    switch (slot.meta.memory_location()) {
    case MemoryLocation::HOST:
      holoflow_logger()->info("allocation tens_id: {} on host", id);
      slot.host_data = make_unique_host_ptr<uint8_t>(slot.meta.size_in_bytes());
      slot.device_data.reset();
      slot.data = slot.host_data.get();
      break;
    case MemoryLocation::DEVICE:
      holoflow_logger()->info("allocation tens_id: {} on device", id);
      slot.device_data =
          make_unique_device_ptr<uint8_t>(slot.meta.size_in_bytes());
      slot.host_data.reset();
      slot.data = slot.device_data.get();
      break;
    }
  }

  holoflow_logger()->info("Getting PESs");
  GetPES get_pes;
  model_root.get().accept(get_pes);
  auto pes = get_pes.result();

  // TODO: Add the cuda streams to the model
  return std::make_unique<Model>(std::move(model_nodes),
                                 std::move(tensor_slots), std::move(pes),
                                 std::move(streams), model_root);
}

} // namespace dh
