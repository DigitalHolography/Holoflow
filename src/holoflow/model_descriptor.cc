#include "holoflow/model_descriptor.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     ModelDescriptorNode Implementation
// ==========================================================================

ModelDescriptorNode::ModelDescriptorNode(const std::string &kind,
                                         const std::string &name,
                                         const json &params)
    : kind_(kind), name_(name), params_(params) {}

std::string &ModelDescriptorNode::name() { return name_; }

const std::string &ModelDescriptorNode::name() const { return name_; }

std::string &ModelDescriptorNode::kind() { return kind_; }

const std::string &ModelDescriptorNode::kind() const { return kind_; }

json &ModelDescriptorNode::params() { return params_; }

const json &ModelDescriptorNode::params() const { return params_; }

void ModelDescriptorNode::add_child(ModelDescriptorNode &child) {
  children_.push_back(child);
}

std::span<std::reference_wrapper<ModelDescriptorNode>>
ModelDescriptorNode::children() {
  return std::span(children_);
}

std::span<const std::reference_wrapper<ModelDescriptorNode>>
ModelDescriptorNode::children() const {
  return std::span(children_);
}

// ==========================================================================
//                     TaskDescriptorNode Implementation
// ==========================================================================

TaskDescriptorNode::TaskDescriptorNode(const std::string &kind,
                                       const std::string &name,
                                       const json &params)
    : ModelDescriptorNode(kind, name, params) {}

void TaskDescriptorNode::accept(ModelDescriptorVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     AccumulatorDescriptorNode Implementation
// ==========================================================================

AccumulatorDescriptorNode::AccumulatorDescriptorNode(const std::string &kind,
                                                     const std::string &name,
                                                     const json &params)
    : ModelDescriptorNode(kind, name, params) {}

void AccumulatorDescriptorNode::accept(ModelDescriptorVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     SourceDescriptorNode Implementation
// ==========================================================================

SourceDescriptorNode::SourceDescriptorNode(const std::string &kind,
                                           const std::string &name,
                                           const json &params)
    : ModelDescriptorNode(kind, name, params) {}

void SourceDescriptorNode::accept(ModelDescriptorVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     SinkDescriptorNode Implementation
// ==========================================================================

SinkDescriptorNode::SinkDescriptorNode(const std::string &kind,
                                       const std::string &name,
                                       const json &params)
    : ModelDescriptorNode(kind, name, params) {}

void SinkDescriptorNode::accept(ModelDescriptorVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     ModelDescriptor Implementation
// ==========================================================================

Error ModelDescriptor::add_task_factory(const std::string &kind,
                                        std::unique_ptr<TaskFactory> factory) {
  VLOG(2) << "Adding task factory: " << kind;

  if (in_factories(kind)) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  task_factories_map_.emplace(kind, std::ref(*factory));
  task_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_accumulator_factory(
    const std::string &kind, std::unique_ptr<AccumulatorFactory> factory) {
  VLOG(2) << "Adding accumulator factory: " << kind;

  if (in_factories(kind)) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  accumulator_factories_map_.emplace(kind, std::ref(*factory));
  accumulator_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_source_factory(
    const std::string &kind, std::unique_ptr<SourceFactory> factory) {
  VLOG(2) << "Adding source factory: " << kind;

  if (in_factories(kind)) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  source_factories_map_.emplace(kind, std::ref(*factory));
  source_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_sink_factory(const std::string &kind,
                                        std::unique_ptr<SinkFactory> factory) {
  VLOG(2) << "Adding sink factory: " << kind;

  if (in_factories(kind)) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  sink_factories_map_.emplace(kind, std::ref(*factory));
  sink_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_task(const std::string &kind,
                                const std::string &name, const json &params) {
  VLOG(2) << "Adding task: " << kind;

  if (!task_factories_map_.contains(kind)) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_nodes(name)) {
    LOG(WARNING) << "Task " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  tasks_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_accumulator(const std::string &kind,
                                       const std::string &name,
                                       const json &params) {
  VLOG(2) << "Adding accumulator: " << kind;

  if (!accumulator_factories_map_.contains(kind)) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_nodes(name)) {
    LOG(WARNING) << "Accumulator " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  accumulators_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_source(const std::string &kind,
                                  const std::string &name, const json &params) {
  VLOG(2) << "Adding source: " << kind;

  if (!source_factories_map_.contains(kind)) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_nodes(name)) {
    LOG(WARNING) << "Source " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  sources_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_sink(const std::string &kind,
                                const std::string &name, const json &params) {
  VLOG(2) << "Adding sink: " << kind;

  if (!sink_factories_map_.contains(kind)) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_nodes(name)) {
    LOG(WARNING) << "Sink " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  sinks_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::set_source(const std::string &name) {
  VLOG(2) << "Setting source: " << name;

  if (root_) {
    // TODO support reassigning source
    LOG(WARNING) << "Source is already set";
    return Error::INTERNAL_ERROR;
  }

  if (!sources_.contains(name)) {
    LOG(WARNING) << "Source " << name << " is not declared";
    return Error::INTERNAL_ERROR;
  }

  auto [kind, params] = sources_.at(name);
  auto source = std::make_unique<SourceDescriptorNode>(kind, name, params);
  root_ = source.get();
  nodes_.push_back(std::move(source));
  return Error::SUCCESS;
}

namespace {

class FindByName : public ModelDescriptorVisitor {
public:
  FindByName(const std::string &name) : name_(name), result_(nullptr) {}

  void visit(TaskDescriptorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (node.name() == name_) {
      result_ = &node;
      return;
    }

    for (auto &child : node.children()) {
      child.get().accept(*this);
      if (result_) {
        return;
      }
    }
  }

  void visit(AccumulatorDescriptorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (node.name() == name_) {
      result_ = &node;
      return;
    }

    for (auto &child : node.children()) {
      child.get().accept(*this);
      if (result_) {
        return;
      }
    }
  }

  void visit(SourceDescriptorNode &node) {
    VLOG(2) << "visiting: " << node.name();

    if (node.name() == name_) {
      result_ = &node;
      return;
    }

    for (auto &child : node.children()) {
      child.get().accept(*this);
      if (result_) {
        return;
      }
    }
  }

  void visit(SinkDescriptorNode &node) {
    VLOG(2) << "visiting: " << node.name();

    if (node.name() == name_) {
      result_ = &node;
      return;
    }

    CHECK(node.children().empty());
  }

  ModelDescriptorNode *result() { return result_; }

private:
  std::string name_;
  ModelDescriptorNode *result_;
};

} // namespace

Error ModelDescriptor::add_child(const std::string &parent_name,
                                 const std::string &child_name) {
  VLOG(2) << "Adding child: " << child_name << " to parent: " << parent_name;

  FindByName find_by_name(child_name);
  root_->accept(find_by_name);
  if (find_by_name.result()) {
    LOG(WARNING) << "Node " << child_name << " was already added as a child";
    return Error::INTERNAL_ERROR;
  }

  find_by_name = FindByName(parent_name);
  root_->accept(find_by_name);
  if (!find_by_name.result()) {
    LOG(WARNING) << "Node " << parent_name << " was not added in the tree";
    return Error::INTERNAL_ERROR;
  }

  std::unique_ptr<ModelDescriptorNode> child_node;
  if (tasks_.contains(child_name)) {
    auto [kind, params] = tasks_.at(child_name);
    child_node.reset(new TaskDescriptorNode(kind, child_name, params));
  } else if (accumulators_.contains(child_name)) {
    auto [kind, params] = accumulators_.at(child_name);
    child_node.reset(new AccumulatorDescriptorNode(kind, child_name, params));
  } else if (sources_.contains(child_name)) {
    auto [kind, params] = sources_.at(child_name);
    child_node.reset(new SourceDescriptorNode(kind, child_name, params));
  } else if (sinks_.contains(child_name)) {
    auto [kind, params] = sinks_.at(child_name);
    child_node.reset(new SinkDescriptorNode(kind, child_name, params));
  } else {
    LOG(WARNING) << "Node " << child_name << " was not declared";
    return Error::INTERNAL_ERROR;
  }

  auto parent_node = find_by_name.result();
  parent_node->add_child(*child_node);
  nodes_.push_back(std::move(child_node));
  return Error::SUCCESS;
}

const ModelDescriptor::TaskFactoryMap &ModelDescriptor::task_factories() const {
  return task_factories_map_;
}

const ModelDescriptor::AccumulatorFactoryMap &
ModelDescriptor::accumulator_factories() const {
  return accumulator_factories_map_;
}

const ModelDescriptor::SourceFactoryMap &
ModelDescriptor::source_factories() const {
  return source_factories_map_;
}

const ModelDescriptor::SinkFactoryMap &ModelDescriptor::sink_factories() const {
  return sink_factories_map_;
}

ModelDescriptorNode *ModelDescriptor::root() const { return root_; }

bool ModelDescriptor::in_factories(const std::string &kind) {
  return task_factories_map_.contains(kind) ||
         accumulator_factories_map_.contains(kind) ||
         source_factories_map_.contains(kind) ||
         sink_factories_map_.contains(kind);
}

bool ModelDescriptor::in_nodes(const std::string &name) {
  return tasks_.contains(name) || accumulators_.contains(name) ||
         sources_.contains(name) || sinks_.contains(name);
}

} // namespace dh