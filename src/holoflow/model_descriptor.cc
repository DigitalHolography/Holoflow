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
//                     ModelDescriptor Implementation
// ==========================================================================

Error ModelDescriptor::add_task_factory(const std::string &kind,
                                        std::unique_ptr<TaskFactory> factory) {
  bool in_task_factories =
      task_factories_map_.find(kind) != task_factories_map_.end();

  bool in_accumulator_factories =
      accumulator_factories_map_.find(kind) != accumulator_factories_map_.end();

  if (in_task_factories || in_accumulator_factories) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  task_factories_map_.emplace(kind, std::ref(*factory));
  task_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_accumulator_factory(
    const std::string &kind, std::unique_ptr<AccumulatorFactory> factory) {
  bool in_task_factories =
      task_factories_map_.find(kind) != task_factories_map_.end();

  bool in_accumulator_factories =
      accumulator_factories_map_.find(kind) != accumulator_factories_map_.end();

  if (in_task_factories || in_accumulator_factories) {
    LOG(WARNING) << "Factory " << kind << " was already registered";
    return Error::INTERNAL_ERROR;
  }

  accumulator_factories_map_.emplace(kind, std::ref(*factory));
  accumulator_factories_.push_back(std::move(factory));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_task(const std::string &kind,
                                const std::string &name, const json &params) {
  bool in_task_factories =
      task_factories_map_.find(kind) != task_factories_map_.end();

  bool in_declared_tasks = tasks_.find(name) != tasks_.end();

  if (!in_task_factories) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_declared_tasks) {
    LOG(WARNING) << "Task " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  tasks_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::add_accumulator(const std::string &kind,
                                       const std::string &name,
                                       const json &params) {
  bool in_accumulator_factories =
      accumulator_factories_map_.find(kind) != accumulator_factories_map_.end();

  bool in_declared_accumulators =
      accumulators_.find(name) != accumulators_.end();

  if (!in_accumulator_factories) {
    LOG(WARNING) << "Factory " << kind << " is not registered";
    return Error::INTERNAL_ERROR;
  }

  if (in_declared_accumulators) {
    LOG(WARNING) << "Accumulator " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  accumulators_.emplace(name, std::make_pair(kind, params));
  return Error::SUCCESS;
}

Error ModelDescriptor::set_root_accumulator(const std::string &name) {
  if (root_) {
    LOG(WARNING) << "Root accumulator is already set";
    return Error::INTERNAL_ERROR;
  }

  bool in_declared_accumulators =
      accumulators_.find(name) != accumulators_.end();

  if (in_declared_accumulators) {
    LOG(WARNING) << "Accumulator " << name << " is already declared";
    return Error::INTERNAL_ERROR;
  }

  auto [kind, params] = accumulators_.at(name);
  auto accu = std::make_unique<AccumulatorDescriptorNode>(kind, name, params);
  root_ = accu.get();
  nodes_.push_back(std::move(accu));
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

  ModelDescriptorNode *result() { return result_; }

private:
  std::string name_;
  ModelDescriptorNode *result_;
};

} // namespace

Error ModelDescriptor::add_child(const std::string &parent_name,
                                 const std::string &child_name) {
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
  bool in_tasks = tasks_.find(child_name) != tasks_.end();
  bool in_accus = accumulators_.find(child_name) != accumulators_.end();

  if (in_tasks) {
    auto [kind, params] = tasks_.at(child_name);
    child_node.reset(new TaskDescriptorNode(kind, child_name, params));
  } else if (in_accus) {
    auto [kind, params] = accumulators_.at(child_name);
    child_node.reset(new AccumulatorDescriptorNode(kind, child_name, params));
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

ModelDescriptorNode *ModelDescriptor::root() const { return root_; }

} // namespace dh