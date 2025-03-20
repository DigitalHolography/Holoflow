#include "holoflow/model_descriptor.hh"

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

void ModelDescriptorNode::add_child(
    std::unique_ptr<ModelDescriptorNode> child) {
  children_.push_back(std::move(child));
}

bool ModelDescriptorNode::remove_child(const ModelDescriptorNode *child) {
  auto it =
      std::remove_if(children_.begin(), children_.end(),
                     [child](const std::unique_ptr<ModelDescriptorNode> &ptr) {
                       return ptr.get() == child;
                     });

  if (it != children_.end()) {
    children_.erase(it, children_.end());
    return true;
  }
  return false;
}

bool ModelDescriptorNode::remove_child(size_t index) {
  if (index >= children_.size())
    return false;
  children_.erase(children_.begin() + index);
  return true;
}

std::span<ModelDescriptorNode *> ModelDescriptorNode::children() {
  static thread_local std::vector<ModelDescriptorNode *> ptrs;
  ptrs.clear();
  for (auto &child : children_) {
    ptrs.push_back(child.get());
  }
  return ptrs;
}

std::span<const ModelDescriptorNode *> ModelDescriptorNode::children() const {
  static thread_local std::vector<const ModelDescriptorNode *> ptrs;
  ptrs.clear();
  for (const auto &child : children_) {
    ptrs.push_back(child.get());
  }
  return ptrs;
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

} // namespace dh