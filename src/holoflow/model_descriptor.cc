#include "holoflow/model_descriptor.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     ModelDescriptorNode Implementation
// ==========================================================================

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

void TaskDescriptorNode::accept(ModelDescriptorVisitor &visitor) {
  visitor.visit(*this);
}

// ==========================================================================
//                     TaskDescriptorNode Implementation
// ==========================================================================

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

  bool in_declared_accumulators = tasks_.find(name) != tasks_.end();

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

} // namespace dh