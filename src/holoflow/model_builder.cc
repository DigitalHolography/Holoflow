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

int ModelNodeBuilder::itens_id() const {
  CHECK(itens_id_);
  return *itens_id_;
}

int ModelNodeBuilder::otens_id() const {
  CHECK(otens_id_);
  return *otens_id_;
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

std::optional<int> ModelNodeBuilder::get_itens_id() const { return itens_id_; }

std::optional<int> ModelNodeBuilder::get_otens_id() const { return itens_id_; }

ModelNodeBuilder &ModelNodeBuilder::set_name(const std::string &name) {
  name_ = name;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::set_kind(const std::string &kind) {
  kind_ = kind;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::set_params(const json &params) {
  params_ = params;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::set_itens_id(int id) {
  itens_id_ = id;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::set_otens_id(int id) {
  otens_id_ = id;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::set_stream(cudaStream_t stream) {
  stream_ = stream;
  return *this;
}

ModelNodeBuilder &ModelNodeBuilder::add_child(ModelNodeBuilder &child) {
  children_.push_back(child);
  return *this;
}

// ==========================================================================
//                     TaskNodeBuilder Implementation
// ==========================================================================

Task &TaskNodeBuilder::task() const {
  CHECK(task_);
  return *task_;
}

const TaskMeta &TaskNodeBuilder::task_meta() const {
  CHECK(task_meta_);
  return *task_meta_;
}

TaskNodeBuilder &TaskNodeBuilder::set_task(std::unique_ptr<Task> task) {
  task_ = std::move(task);
  return *this;
}

TaskNodeBuilder &TaskNodeBuilder::set_task_meta(const TaskMeta &task_meta) {
  task_meta_ = task_meta;
  return *this;
}

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

Accumulator &AccumulatorNodeBuilder::accumulator() const {
  CHECK(accumulator_);
  return *accumulator_;
}

const AccumulatorMeta &AccumulatorNodeBuilder::accumulator_meta() const {
  CHECK(accumulator_meta_);
  return *accumulator_meta_;
}

AccumulatorNodeBuilder &AccumulatorNodeBuilder::set_accumulator(
    std::unique_ptr<Accumulator> accumulator) {
  accumulator_ = std::move(accumulator);
  return *this;
}

AccumulatorNodeBuilder &AccumulatorNodeBuilder::set_accumulator_meta(
    const AccumulatorMeta &accumulator_meta) {
  accumulator_meta_ = accumulator_meta;
  return *this;
}

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

    unique_cuda_stream stream;
    node.set_stream(stream.get());
    streams_stack_.push(stream.get());
    streams_.push_back(std::move(stream));

    for (auto child : node.children()) {
      child.get().accept(*this);
    }

    streams_stack_.pop();
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

  TypeCheck(const TaskFactoryMap &task_factories_map,
            const AccumulatorFactoryMap &accumulator_factories_map,
            const TensorMeta &imeta)
      : task_factories_map_(task_factories_map),
        accumulator_factories_map_(accumulator_factories_map),
        imetas_stack_({imeta}), result_(true) {}

  void visit(TaskNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(!imetas_stack_.empty());
    CHECK(task_factories_map_.find(node.kind()) != task_factories_map_.end());

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
    CHECK(accumulator_factories_map_.find(node.kind()) !=
          accumulator_factories_map_.end());

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

  bool result() { return result_; }

private:
  /// Maps task kinds to factories.
  const TaskFactoryMap &task_factories_map_;

  /// Maps accumulator kinds to factories.
  const AccumulatorFactoryMap &accumulator_factories_map_;

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

    if (!seen_non_inlined_stack_.empty()) {
      if (!seen_non_inlined_stack_.top()) {
        LOG(WARNING) << "Non inlined task check failed at node: "
                     << node.name();
        result_ = false;
        return;
      }
    }

    seen_non_inlined_stack_.push(false);
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    seen_non_inlined_stack_.pop();
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

    parents_stack_.push(std::ref(node));
    for (auto child : node.children()) {
      child.get().accept(*this);
    }
    parents_stack_.pop();
  }

  void visit(AccumulatorNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    // Set current
    node.set_itens_id(next_id_++);
    node.set_otens_id(next_id_++);
    VLOG(1) << node.name() << " <= itens_id: " << node.itens_id()
            << ", otens_id: " << node.otens_id();

    // Set parent
    if (!parents_stack_.empty()) {
      auto parent = parents_stack_.top();
      parent.get().set_otens_id(node.itens_id());
      VLOG(1) << parent.get().name()
              << " <= otens_id: " << parent.get().otens_id();
    }

    // Set children
    for (auto child : node.children()) {
      child.get().set_itens_id(node.otens_id());
      VLOG(1) << child.get().name()
              << " <= itens_id: " << child.get().itens_id();
    }

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

    // Forward propagation (otens_id <= itens_id)
    if (node.task_meta().inlined() && node.get_itens_id() &&
        node.get_otens_id() != node.get_itens_id()) {

      // Set current
      CHECK(!node.get_otens_id());
      node.set_otens_id(node.itens_id());
      VLOG(1) << node.name() << " <= otens_id: " << node.otens_id();

      // Set children
      for (auto child : node.children()) {
        child.get().set_itens_id(node.otens_id());
        VLOG(1) << child.get().name()
                << " <= itens_id: " << child.get().itens_id();
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
      parent.get().set_otens_id(node.itens_id());
      VLOG(1) << parent.get().name()
              << " <= otens_id: " << parent.get().otens_id();
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
        child.get().set_itens_id(node.otens_id());
        VLOG(1) << child.get().name()
                << " <= itens_id: " << child.get().itens_id();
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

  CallFactories(const TaskFactoryMap &task_factories_map,
                const AccumulatorFactoryMap &accumulator_factories_map)
      : task_factories_map_(task_factories_map),
        accumulator_factories_map_(accumulator_factories_map), result_(true) {}

  void visit(TaskNodeBuilder &node) {
    VLOG(2) << "visiting: " << node.name();

    CHECK(task_factories_map_.find(node.kind()) != task_factories_map_.end());

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

    CHECK(accumulator_factories_map_.find(node.kind()) !=
          accumulator_factories_map_.end());

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

  bool result() { return result_; }

private:
  /// Maps task kinds to factories.
  const TaskFactoryMap &task_factories_map_;

  /// Maps accumulator kinds to factories.
  const AccumulatorFactoryMap &accumulator_factories_map_;

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

    if (!node.children().empty()) {
      pes_.push_back(std::ref(node));
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  std::vector<std::reference_wrapper<ModelNode>> result() {
    return std::move(pes_);
  }

private:
  std::vector<std::reference_wrapper<ModelNode>> pes_;
};

} // namespace

tl::expected<std::unique_ptr<Model>, Error>
ModelBuilder::build(const ModelDescriptor &descriptor,
                    const TensorMeta &imeta) {
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
  TypeCheck type_check(descriptor.task_factories(),
                       descriptor.accumulator_factories(), imeta);
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
  CallFactories call_factories(descriptor.task_factories(),
                               descriptor.accumulator_factories());
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

  return std::make_unique<Model>(std::move(model_nodes),
                                 std::move(tensor_slots), std::move(pes),
                                 model_root);
}

} // namespace dh