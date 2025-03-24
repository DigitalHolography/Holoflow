#include "holoflow/model_builder.hh"

#include <stack>

namespace dh {

// ==========================================================================
//                     ModelNodeBuilder Implementation
// ==========================================================================

const std::string &ModelNodeBuilder::name() const {
  CHECK(name_);
  return *name_;
}

const std::string &ModelNodeBuilder::kind() const {
  CHECK(kind_);
  return *kind_;
}

const json &ModelNodeBuilder::params() const {
  CHECK(params_);
  return *params_;
}

cudaStream_t ModelNodeBuilder::stream() const {
  CHECK(stream_);
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
  CHECK(itens_id_);
  return *itens_id_;
}

int TaskNodeBuilder::otens_id() const {
  CHECK(otens_id_);
  return *otens_id_;
}

Task &TaskNodeBuilder::task() const {
  CHECK(task_);
  return *task_;
}

const TaskMeta &TaskNodeBuilder::task_meta() const {
  CHECK(task_meta_);
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
  CHECK(name_);
  CHECK(kind_);
  CHECK(params_);
  CHECK(itens_id_);
  CHECK(otens_id_);
  CHECK(stream_);
  CHECK(task_);
  CHECK(task_meta_);

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
  CHECK(itens_id_);
  return *itens_id_;
}

int AccumulatorNodeBuilder::otens_id() const {
  CHECK(otens_id_);
  return *otens_id_;
}

Accumulator &AccumulatorNodeBuilder::accumulator() const {
  CHECK(accumulator_);
  return *accumulator_;
}

const AccumulatorMeta &AccumulatorNodeBuilder::accumulator_meta() const {
  CHECK(accumulator_meta_);
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
  CHECK(name_);
  CHECK(kind_);
  CHECK(params_);
  CHECK(itens_id_);
  CHECK(otens_id_);
  CHECK(stream_);
  CHECK(accumulator_);
  CHECK(accumulator_meta_);

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
  CHECK(otens_id_);
  return *otens_id_;
}

Source &SourceNodeBuilder::source() const {
  CHECK(source_);
  return *source_;
}

const SourceMeta &SourceNodeBuilder::source_meta() const {
  CHECK(source_meta_);
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
  CHECK(name_);
  CHECK(kind_);
  CHECK(params_);
  CHECK(otens_id_);
  CHECK(stream_);
  CHECK(source_);
  CHECK(source_meta_);

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
  CHECK(itens_id_);
  return *itens_id_;
}

Sink &SinkNodeBuilder::sink() const {
  CHECK(sink_);
  return *sink_;
}

const SinkMeta &SinkNodeBuilder::sink_meta() const {
  CHECK(sink_meta_);
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
  CHECK(name_);
  CHECK(kind_);
  CHECK(params_);
  CHECK(itens_id_);
  CHECK(stream_);
  CHECK(sink_);
  CHECK(sink_meta_);

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

    auto builder_node = std::make_unique<SinkNodeBuilder>();
    builder_node->set_kind(node.kind());
    builder_node->set_name(node.name());
    builder_node->set_params(node.params());

    root_ = &*builder_node;
    nodes_.push_back(std::move(builder_node));
    CHECK(node.children().empty());
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!streams_stack_.empty());
    node.set_stream(streams_stack_.top());

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!streams_stack_.empty());
    node.set_stream(streams_stack_.top());
    CHECK(node.children().empty());
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!imetas_stack_.empty());
    CHECK(task_factories_map_.contains(node.kind()));

    auto &factory = task_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      LOG(WARNING) << "Type checking failed at node: " << node.name();
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!imetas_stack_.empty());
    CHECK(accumulator_factories_map_.contains(node.kind()));

    auto &factory = accumulator_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      LOG(WARNING) << "Type checking failed at node: " << node.name();
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(imetas_stack_.empty());
    CHECK(source_factories_map_.contains(node.kind()));

    auto &factory = source_factories_map_.at(node.kind());
    auto result = factory.get().type_check(node.params());
    if (!result) {
      LOG(WARNING) << "Type checking failed at node: " << node.name();
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!imetas_stack_.empty());
    CHECK(sink_factories_map_.contains(node.kind()));

    auto &factory = sink_factories_map_.at(node.kind());
    auto result = factory.get().type_check(imetas_stack_.top(), node.params());
    if (!result) {
      LOG(WARNING) << "Type checking failed at node: " << node.name();
      result_ = false;
      return;
    }

    node.set_sink_meta(result.value());
    CHECK(node.children().empty());
  }

  bool result() { return result_; }

private:
  /// Maps task kinds to factories.
  const TaskFactoryMap &task_factories_map_;

  /// Maps accumulator kinds to factories.
  const AccumulatorFactoryMap &accumulator_factories_map_;

  /// Maps source kinds to factories.
  const SourceFactoryMap &source_factories_map_;

  /// Maps sink kinds to factories.
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
    VLOG(2) << "visiting: " << node.name();

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      LOG(WARNING) << "Single inlined check failed at node: " << node.name();
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      LOG(WARNING) << "Single inlined check failed at node: " << node.name();
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    size_t nb_inlined_children = std::count_if(
        node.children().begin(), node.children().end(), is_inlined);

    if (nb_inlined_children > 1) {
      LOG(WARNING) << "Single inlined check failed at node: " << node.name();
      result_ = false;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());
  }

  bool result() { return result_; }

private:
  bool result_;
};

class NonInlinedTaskCheck : public ModelBuilderVisitor {
public:
  NonInlinedTaskCheck() : result_(true) {}

  void visit(TaskNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(!seen_non_inlined_stack_.empty());

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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!seen_non_inlined_stack_.empty());

    if (!seen_non_inlined_stack_.top()) {
      LOG(WARNING) << "Non inlined task check failed at node: " << node.name();
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(seen_non_inlined_stack_.empty());

    seen_non_inlined_stack_.push(true);
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    seen_non_inlined_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!parents_stack_.empty());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(!parents_stack_.empty());

    // Set current
    node.set_itens_id(next_id_++);
    node.set_otens_id(next_id_++);
    VLOG(1) << node.name() << " <= itens_id: " << node.itens_id()
            << ", otens_id: " << node.otens_id();

    // Set parent
    auto parent = parents_stack_.top();
    if (auto *task = dynamic_cast<TaskNodeBuilder *>(&parent.get())) {
      task->set_otens_id(node.itens_id());
      VLOG(1) << task->name() << " <= otens_id: " << task->otens_id();
    }

    else if (auto *source = dynamic_cast<SourceNodeBuilder *>(&parent.get())) {
      source->set_otens_id(node.itens_id());
      VLOG(1) << source->name() << " <= otens_id: " << source->otens_id();
    }

    else {
      LOG(FATAL)
          << "Expected parent to be TaskBuilderNode or SourceBuilderNode";
    }

    // Set children
    for (auto child : node.children()) {
      if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
        task->set_itens_id(node.otens_id());
        VLOG(1) << task->name() << " <= itens_id: " << task->itens_id();
      }

      else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
        sink->set_itens_id(node.otens_id());
        VLOG(1) << sink->name() << " <= itens_id: " << sink->itens_id();
      }

      else {
        LOG(FATAL) << "Expected child to be TaskBuilderNode or SinkBuilderNode";
      }
    }

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(parents_stack_.empty());

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(!parents_stack_.empty());

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
    VLOG(2) << "visiting: " << node.name();

    CHECK(!parents_stack_.empty());

    // Forward propagation (otens_id <= itens_id)
    if (node.task_meta().inlined() && node.get_itens_id() &&
        node.get_otens_id() != node.get_itens_id()) {

      // Set current
      CHECK(!node.get_otens_id());
      node.set_otens_id(node.itens_id());
      VLOG(1) << node.name() << " <= otens_id: " << node.otens_id();

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          VLOG(1) << task->name() << " <= itens_id: " << task->itens_id();
        }

        else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          VLOG(1) << sink->name() << " <= itens_id: " << sink->itens_id();
        }

        else {
          LOG(FATAL)
              << "Expected child to be TaskBuilderNode or SinkBuilderNode";
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
      CHECK(!node.get_itens_id());
      node.set_itens_id(node.otens_id());
      VLOG(1) << node.name() << " <= itens_id: " << node.itens_id();

      // Set parent
      CHECK(!parents_stack_.empty());
      auto parent = parents_stack_.top();
      if (auto *task = dynamic_cast<TaskNodeBuilder *>(&parent.get())) {
        task->set_otens_id(node.itens_id());
        VLOG(1) << task->name() << " <= otens_id: " << task->otens_id();
      }

      else if (auto *source =
                   dynamic_cast<SourceNodeBuilder *>(&parent.get())) {
        source->set_otens_id(node.itens_id());
        VLOG(1) << source->name() << " <= otens_id: " << source->otens_id();
      }

      else {
        LOG(FATAL)
            << "Expected parent to be TaskBuilderNode or SourceBuilderNode";
      }
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

    if (!node.task_meta().inlined() && !node.get_otens_id()) {
      // Set current
      node.set_otens_id(next_id_++);
      VLOG(1) << node.name() << " <= otens_id: " << node.otens_id();

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          VLOG(1) << task->name() << " <= itens_id: " << task->itens_id();
        }

        else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          VLOG(1) << sink->name() << " <= itens_id: " << sink->itens_id();
        }

        else {
          LOG(FATAL)
              << "Expected child to be TaskBuilderNode or SinkBuilderNode";
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
    VLOG(2) << "visiting: " << node.name();

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    if (!node.get_otens_id()) {
      // Set current
      node.set_otens_id(next_id_++);
      VLOG(1) << node.name() << " <= otens_id: " << node.otens_id();

      // Set children
      for (auto child : node.children()) {
        if (auto *task = dynamic_cast<TaskNodeBuilder *>(&child.get())) {
          task->set_itens_id(node.otens_id());
          VLOG(1) << task->name() << " <= itens_id: " << task->itens_id();
        }

        else if (auto *sink = dynamic_cast<SinkNodeBuilder *>(&child.get())) {
          sink->set_itens_id(node.otens_id());
          VLOG(1) << sink->name() << " <= itens_id: " << sink->itens_id();
        }

        else {
          LOG(FATAL)
              << "Expected child to be TaskBuilderNode or SinkBuilderNode";
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());
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
    VLOG(2) << "visiting: " << node.name();

    CHECK(task_factories_map_.contains(node.kind()));

    auto &factory = task_factories_map_.at(node.kind());
    auto result = factory.get().create(node.task_meta().imeta(), node.params(),
                                       node.stream());
    if (!result) {
      LOG(WARNING) << "Factory call failed at node: " << node.name();
      result_ = false;
      return;
    }

    node.set_task(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(accumulator_factories_map_.contains(node.kind()));

    auto &factory = accumulator_factories_map_.at(node.kind());
    auto result = factory.get().create(node.accumulator_meta().imeta(),
                                       node.params(), node.stream());
    if (!result) {
      LOG(WARNING) << "Factory call failed at node: " << node.name();
      result_ = false;
      return;
    }

    node.set_accumulator(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(source_factories_map_.contains(node.kind()));

    auto &factory = source_factories_map_.at(node.kind());
    auto result = factory.get().create(node.params(), node.stream());
    if (!result) {
      LOG(WARNING) << "Factory call failed at node: " << node.name();
      result_ = false;
      return;
    }

    node.set_source(std::move(result.value()));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(sink_factories_map_.contains(node.kind()));

    auto &factory = sink_factories_map_.at(node.kind());
    auto result = factory.get().create(node.sink_meta().imeta(), node.params(),
                                       node.stream());
    if (!result) {
      LOG(WARNING) << "Factory call failed at node: " << node.name();
      result_ = false;
      return;
    }

    node.set_sink(std::move(result.value()));

    CHECK(node.children().empty());
  }

  bool result() { return result_; }

private:
  /// Maps task kinds to factories.
  const TaskFactoryMap &task_factories_map_;

  /// Maps accumulator kinds to factories.
  const AccumulatorFactoryMap &accumulator_factories_map_;

  /// Maps source kinds to factories.
  const SourceFactoryMap &source_factories_map_;

  /// Maps sink kinds to factories.
  const SinkFactoryMap &sink_factories_map_;

  bool result_;
};

class BuildModelNodes : public ModelBuilderVisitor {
public:
  void visit(TaskNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(SourceNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    auto model_node = node.build();

    for (auto child : node.children()) {
      child.get().accept(*this);
      model_node->add_child(*root_);
    }

    root_ = &*model_node;
    nodes_.push_back(std::move(model_node));
  }

  void visit(SinkNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    auto model_node = node.build();

    CHECK(node.children().empty());

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

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
    VLOG(2) << "visiting: " << node.name();

    if (!tensors_map_.contains(node.otens_id())) {
      tensors_map_.emplace(node.otens_id(),
                           Model::TensorSlot(node.source_meta().ometa()));
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    VLOG(2) << "visiting: " << node.name();

    if (!tensors_map_.contains(node.itens_id())) {
      tensors_map_.emplace(node.itens_id(),
                           Model::TensorSlot(node.sink_meta().imeta()));
    }

    CHECK(node.children().empty());
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
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (node.itens_id() == tens_id_ || node.otens_id() == tens_id_) {
      result_ = true;
      return;
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());
  }

  bool result() { return result_; }

private:
  int tens_id_;
  bool result_;
};

class GetPES : public ModelVisitor {
public:
  void visit(TaskNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    pes_.push_back(std::ref(node));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) {
    VLOG(2) << "visiting: " << node.name();

    pes_.push_back(std::ref(node));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());
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
  LOG(INFO) << "Building model";
  CHECK(descriptor.root());

  LOG(INFO) << "Creating builder nodes";
  CreateBuilderNodes create_builder_nodes;
  descriptor.root()->accept(create_builder_nodes);
  auto [builder_root, builder_nodes] = create_builder_nodes.result();

  LOG(INFO) << "Assigning cuda streams";
  AssignStreams assign_streams;
  builder_root.get().accept(assign_streams);
  auto streams = assign_streams.result();

  LOG(INFO) << "Performing type check";
  TypeCheck type_check(
      descriptor.task_factories(), descriptor.accumulator_factories(),
      descriptor.source_factories(), descriptor.sink_factories());
  builder_root.get().accept(type_check);
  CHECK(type_check.result());

  LOG(INFO) << "Performing single inlined child check";
  SingleInlinedChildCheck single_inlined_child_check;
  builder_root.get().accept(single_inlined_child_check);
  CHECK(single_inlined_child_check.result());

  LOG(INFO) << "Performing non inlined child check";
  NonInlinedTaskCheck non_inlined_child_check;
  builder_root.get().accept(non_inlined_child_check);
  CHECK(non_inlined_child_check.result());

  int next_tens_id = 0;

  LOG(INFO) << "Assigning accumulators tensors";
  AssignAccumulatorsTensors assign_accumulators_tensors(next_tens_id);
  builder_root.get().accept(assign_accumulators_tensors);

  LOG(INFO) << "Assigning inlined tasks tensors (first pass)";
  AssignInlinedTaskTensors assign_inlined_tasks_tensors1(next_tens_id);
  builder_root.get().accept(assign_inlined_tasks_tensors1);

  LOG(INFO) << "Assigning non inlined tasks tensors";
  AssignNonInlinedTaskTensors assign_non_inlined_tasks_tensor(next_tens_id);
  builder_root.get().accept(assign_non_inlined_tasks_tensor);

  LOG(INFO) << "Assigning inlined tasks tensors (second pass)";
  AssignInlinedTaskTensors assign_inlined_tasks_tensors2(next_tens_id);
  builder_root.get().accept(assign_inlined_tasks_tensors2);

  LOG(INFO) << "Calling factories";
  CallFactories call_factories(
      descriptor.task_factories(), descriptor.accumulator_factories(),
      descriptor.source_factories(), descriptor.sink_factories());
  builder_root.get().accept(call_factories);
  CHECK(call_factories.result());

  LOG(INFO) << "Building model nodes";
  BuildModelNodes build_model_nodes;
  builder_root.get().accept(build_model_nodes);
  auto [model_root, model_nodes] = build_model_nodes.result();

  LOG(INFO) << "Getting tensors slots";
  GetTensorSlots get_tensor_slots;
  model_root.get().accept(get_tensor_slots);
  auto tensor_slots = get_tensor_slots.result();

  LOG(INFO) << "Allocating non accumulator tensors";
  for (auto &[id, slot] : tensor_slots) {
    IsAccumulatorTensor is_accumulator_tensor(id);
    model_root.get().accept(is_accumulator_tensor);
    if (is_accumulator_tensor.result()) {
      VLOG(1) << "tens_id: " << id << " is an accumulator tensor";
      continue;
    }

    switch (slot.meta.memory_location()) {
    case MemoryLocation::HOST:
      VLOG(1) << "allocation tens_id: " << id << " on host";
      slot.host_data = make_unique_host_ptr<uint8_t>(slot.meta.size_in_bytes());
      slot.device_data.reset();
      break;
    case MemoryLocation::DEVICE:
      VLOG(1) << "allocation tens_id: " << id << " on device";
      slot.device_data =
          make_unique_device_ptr<uint8_t>(slot.meta.size_in_bytes());
      slot.host_data.reset();
      break;
    }
  }

  LOG(INFO) << "Getting PESs";
  GetPES get_pes;
  model_root.get().accept(get_pes);
  auto pes = get_pes.result();

  // TODO: Add the cuda streams to the model
  return std::make_unique<Model>(std::move(model_nodes),
                                 std::move(tensor_slots), std::move(pes),
                                 std::move(streams), model_root);
}

} // namespace dh