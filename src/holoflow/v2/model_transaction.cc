#include "holoflow/v2/model_transaction.hh"

#include <memory>
#include <tl/expected.hpp>

#include "holoflow/accumulator.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v2/error.hh"

namespace dh::v2 {

// ==========================================================================
//                     Model Implementation
// ==========================================================================

ModelTransaction::ModelTransaction(Model &model) : model_(model) {}

ModelTransaction &ModelTransaction::add_source(const std::string &name,
                                               const std::string &kind,
                                               const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_source] Adding source: {}",
                           name);

  if (!model_.has_source_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_source] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_source] Source {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Source {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<SourceNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_sink(const std::string &name,
                                             const std::string &kind,
                                             const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_sink] Adding sink: {}",
                           name);

  if (!model_.has_sink_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_sink] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_sink] Sink {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Sink {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<SinkNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_task(const std::string &name,
                                             const std::string &kind,
                                             const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_task] Adding task: {}",
                           name);

  if (!model_.has_task_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_task] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_task] Task {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Task {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<TaskNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_accumulator(const std::string &name,
                                                    const std::string &kind,
                                                    const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_accumulator] Adding "
                           "accumulator: {}",
                           name);

  if (!model_.has_accumulator_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_accumulator] Factory {} "
                            "is not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_accumulator] "
                            "Accumulator {} is already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Accumulator {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<AccumulatorNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::update_node_params(const std::string &name,
                                                       const json &params) {
  holoflow_logger()->trace("[ModelTransaction::update_node_params] Updating "
                           "node params: {}",
                           name);

  for (auto &node : nodes_) {
    if (node->name_ == name) {
      node->params_ = params;
      return *this;
    }
  }

  holoflow_logger()->warn("[ModelTransaction::update_node_params] Node {} "
                          "not found",
                          name);
  errors_.push_back(
      Error::make(ErrorType::NotFound, "Node {} not found", name));
  return *this;
}

ModelTransaction &ModelTransaction::remove_node(const std::string &name,
                                                RemoveNodeBehavior behavior) {
  holoflow_logger()->trace("[ModelTransaction::remove_node] Removing node: {}",
                           name);

  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if ((*it)->name_ == name) {
      if (behavior == RemoveNodeBehavior::RemoveSubtree) {
        // Remove children
        for (auto &child : (*it)->children_) {
          remove_node(*child.get().name_, behavior);
        }

        // Unlink the node from its parent
        for (auto &parent : nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        nodes_.erase(it);
        return *this;
      }

      if (behavior == RemoveNodeBehavior::OrphanChildren) {
        // Unlink the node from its parent
        for (auto &parent : nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        nodes_.erase(it);
        return *this;
      }

      if (behavior == RemoveNodeBehavior::ReparentChildren) {
        // Add children to the parent
        for (auto &parent : nodes_) {
          if (std::any_of(parent->children_.begin(), parent->children_.end(),
                          [&name](const auto &child) {
                            return child.get().name_ == name;
                          })) {
            for (auto &child : (*it)->children_) {
              parent->children_.push_back(child);
            }
          }
        }

        // Unlink the node from its parent
        for (auto &parent : nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        nodes_.erase(it);
        return *this;
      }
    }
  }

  holoflow_logger()->warn("[ModelTransaction::remove_node] Node {} not found",
                          name);
  errors_.push_back(
      Error::make(ErrorType::NotFound, "Node {} not found", name));
  return *this;
}

ModelTransaction &ModelTransaction::connect(const std::string &parent_name,
                                            const std::string &child_name) {
  holoflow_logger()->trace("[ModelTransaction::connect] Connecting {} to {}",
                           child_name, parent_name);

  auto parent_it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&parent_name](const auto &node) { return node->name_ == parent_name; });

  auto child_it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&child_name](const auto &node) { return node->name_ == child_name; });

  if (parent_it != nodes_.end() && child_it != nodes_.end()) {
    (*parent_it)->children_.push_back(**child_it);
  } else {
    holoflow_logger()->warn("[ModelTransaction::connect] Node {} or {} not "
                            "found",
                            parent_name, child_name);
    errors_.push_back(Error::make(ErrorType::NotFound,
                                  "Node {} or {} not found", parent_name,
                                  child_name));
  }

  return *this;
}

ModelTransaction &
ModelTransaction::disconnect(const std::string &parent_name,
                             const std::string &child_name,
                             DisconnectNodeBehavior behavior) {
  holoflow_logger()->trace("[ModelTransaction::disconnect] Disconnecting {} "
                           "from {}",
                           child_name, parent_name);

  auto parent_it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&parent_name](const auto &node) { return node->name_ == parent_name; });

  auto child_it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&child_name](const auto &node) { return node->name_ == child_name; });

  if (parent_it != nodes_.end() && child_it != nodes_.end()) {
    if (behavior == DisconnectNodeBehavior::OrphanChild) {
      // Remove the child from the parent
      (*parent_it)
          ->children_.erase(std::remove_if((*parent_it)->children_.begin(),
                                           (*parent_it)->children_.end(),
                                           [&child_name](const auto &child) {
                                             return child.get().name_ ==
                                                    child_name;
                                           }),
                            (*parent_it)->children_.end());
    } else if (behavior == DisconnectNodeBehavior::ReparentChildren) {
      // Reparent children to the parent
      for (auto &child : (*child_it)->children_) {
        (*parent_it)->children_.push_back(child);
      }

      // Remove the child from the parent
      (*parent_it)
          ->children_.erase(std::remove_if((*parent_it)->children_.begin(),
                                           (*parent_it)->children_.end(),
                                           [&child_name](const auto &child) {
                                             return child.get().name_ ==
                                                    child_name;
                                           }),
                            (*parent_it)->children_.end());
    }
  } else {
    holoflow_logger()->warn("[ModelTransaction::disconnect] Node {} or {} not "
                            "found",
                            parent_name, child_name);
    errors_.push_back(Error::make(ErrorType::NotFound,
                                  "Node {} or {} not found", parent_name,
                                  child_name));
  }
  return *this;
}

bool ModelTransaction::has_source(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node->name_ == name && model_.has_source_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_sink(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node->name_ == name && model_.has_sink_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_task(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node->name_ == name && model_.has_task_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_accumulator(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node->name_ == name && model_.has_accumulator_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_node(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node->name_ == name) {
      return true;
    }
  }

  return false;
}

void ModelTransaction::validate_orphan_nodes() {
  for (const auto &node : nodes_) {
    // Check if there is any parent node that lists this node as a child.
    auto parent_it =
        std::find_if(nodes_.begin(), nodes_.end(), [&node](const auto &parent) {
          return std::any_of(parent->children_.begin(), parent->children_.end(),
                             [&node](const auto &child) {
                               return child.get().name_ == node->name_;
                             });
        });
    // If no parent is found and the node is not a source, record an error.
    if (parent_it == nodes_.end() && !model_.has_source_factory(*node->kind_)) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateOrphanNodes] Node {} has no parent",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Node {} has no parent", *node->name_));
    }
  }
}

void ModelTransaction::validate_childless_nodes() {
  for (const auto &node : nodes_) {
    // Check if the node has no children and is not a sink.
    if (node->children_.empty() && !model_.has_sink_factory(*node->kind_)) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateChildlessNodes] Node {} has no "
          "children",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Node {} has no children", *node->name_));
    }
  }
}

void ModelTransaction::validate_source_nodes() {
  // Find the first source node.
  auto source_it =
      std::find_if(nodes_.begin(), nodes_.end(), [this](const auto &node) {
        return model_.has_source_factory(*node->kind_);
      });

  if (source_it == nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] No "
                            "source node found");
    errors_.push_back(Error::make(ErrorType::NotFound, "No source node found"));
  }

  // Ensure there is only one source node.
  if (std::find_if(source_it + 1, nodes_.end(), [this](const auto &node) {
        return model_.has_source_factory(*node->kind_);
      }) != nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] Multiple "
                            "source nodes found");
    errors_.push_back(
        Error::make(ErrorType::ConnectionError, "Multiple source nodes found"));
  }

  // Check that the source node has no parent.
  if (std::find_if(
          nodes_.begin(), nodes_.end(), [&source_it](const auto &node) {
            return std::any_of(node->children_.begin(), node->children_.end(),
                               [&source_it](const auto &child) {
                                 return child.get().name_ ==
                                        (*source_it)->name_;
                               });
          }) != nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] Source "
                            "node {} has a parent",
                            *(*source_it)->name_);
    errors_.push_back(Error::make(ErrorType::ConnectionError,
                                  "Source node {} has a parent",
                                  *(*source_it)->name_));
  }
}

void ModelTransaction::validate_sink_nodes() {
  for (const auto &node : nodes_) {
    // Check if the node is a sink and has children.
    if (model_.has_sink_factory(*node->kind_) && !node->children_.empty()) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateSinkNodes] Sink node {} has children",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Sink node {} has children", *node->name_));
    }
  }
}

tl::expected<void, Error> ModelTransaction::commit() {
  holoflow_logger()->trace("[ModelTransaction::commit] Committing changes");

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  validate_orphan_nodes();
  validate_childless_nodes();
  validate_source_nodes();
  validate_sink_nodes();

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  return {};
}

} // namespace dh::v2