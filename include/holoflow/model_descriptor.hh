#pragma once

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoflow/accumulator.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

/**
 * @brief Forward declaration of ModelDescriptorVisitor.
 */
class ModelDescriptorVisitor;

/**
 * @brief Represents a node in the model descriptor tree.
 *
 * This is an abstract base class that defines the structure for model
 * descriptor nodes. Each node has a name, a kind, parameters, and child nodes.
 */
class ModelDescriptorNode {
public:
  using Child = std::reference_wrapper<ModelDescriptorNode>;

  /**
   * @brief Construct a new ModelDescriptorNode.
   * @param kind The kind of node.
   * @param name The name of the node.
   * @param params The params associated with the node.
   */
  ModelDescriptorNode(const std::string &kind, const std::string &name,
                      const json &params);

  /**
   * @brief Virtual destructor for polymorphic destruction.
   */
  virtual ~ModelDescriptorNode() = default;

  /**
   * @brief Accepts a visitor for the Visitor pattern.
   *
   * @param visitor The visitor to process this node.
   */
  virtual void accept(ModelDescriptorVisitor &visitor) = 0;

  /**
   * @brief Gets the mutable name of the node.
   * @return Reference to the node's name.
   */
  std::string &name();

  /**
   * @brief Gets the immutable name of the node.
   * @return Constant reference to the node's name.
   */
  const std::string &name() const;

  /**
   * @brief Gets the mutable kind of the node.
   * @return Reference to the node's kind.
   */
  std::string &kind();

  /**
   * @brief Gets the immutable kind of the node.
   * @return Constant reference to the node's kind.
   */
  const std::string &kind() const;

  /**
   * @brief Gets the mutable parameters of the node.
   * @return Reference to the JSON object representing node parameters.
   */
  json &params();

  /**
   * @brief Gets the immutable parameters of the node.
   * @return Constant reference to the JSON object representing node parameters.
   */
  const json &params() const;

  /**
   * @brief Adds a child node.
   * @param child Reference to the child node.
   */
  void add_child(ModelDescriptorNode &child);

  /**
   * @brief Returns a modifiable view of child nodes.
   * @return A span of reference wrappers to child nodes.
   */
  std::span<Child> children();

  /**
   * @brief Returns a read-only view of child nodes.
   * @return A span of constant reference wrappers to child nodes.
   */
  std::span<const Child> children() const;

private:
  std::string name_;            ///< The name of the node.
  std::string kind_;            ///< The kind/type of the node.
  json params_;                 ///< JSON object storing the node parameters.
  std::vector<Child> children_; ///< List of child nodes.
};

/**
 * @brief Represents a task node in the model descriptor tree.
 *
 * This node type corresponds to computational tasks.
 */
class TaskDescriptorNode : public ModelDescriptorNode {
public:
  /**
   * @brief Construct a new TaskDescriptorNode.
   * @param kind The kind of node.
   * @param name The name of the node.
   * @param params The params associated with the node.
   */
  TaskDescriptorNode(const std::string &kind, const std::string &name,
                     const json &params);

  /**
   * @brief Accepts a visitor for the Visitor pattern.
   * @param visitor The visitor to process this node.
   */
  void accept(ModelDescriptorVisitor &visitor) override;
};

/**
 * @brief Represents an accumulator node in the model descriptor tree.
 *
 * This node type is used for accumulation and synchronization.
 */
class AccumulatorDescriptorNode : public ModelDescriptorNode {
public:
  /**
   * @brief Construct a new AccumulatorDescriptorNode.
   * @param kind The kind of node.
   * @param name The name of the node.
   * @param params The params associated with the node.
   */
  AccumulatorDescriptorNode(const std::string &kind, const std::string &name,
                            const json &params);

  /**
   * @brief Accepts a visitor for the Visitor pattern.
   * @param visitor The visitor to process this node.
   */
  void accept(ModelDescriptorVisitor &visitor) override;
};

/**
 * @brief Visitor interface for traversing model descriptor nodes.
 *
 * Implements the Visitor pattern to allow operations on different node types.
 */
class ModelDescriptorVisitor {
public:
  /**
   * @brief Virtual destructor for proper cleanup.
   */
  virtual ~ModelDescriptorVisitor() = default;

  /**
   * @brief Visits a TaskDescriptorNode.
   * @param node The task node being visited.
   */
  virtual void visit(TaskDescriptorNode &node) = 0;

  /**
   * @brief Visits an AccumulatorDescriptorNode.
   * @param node The accumulator node being visited.
   */
  virtual void visit(AccumulatorDescriptorNode &node) = 0;
};

/**
 * @brief Represents a model descriptor containing tasks, accumulators, and
 * their relationships.
 */
class ModelDescriptor {
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
   * @brief Map from name to (kind, params).
   */
  using TaskMap = std::unordered_map<std::string, std::pair<std::string, json>>;

  /**
   * @brief Map from name to (kind, params).
   */
  using AccumulatorMap =
      std::unordered_map<std::string, std::pair<std::string, json>>;

  /**
   * @brief Adds a task factory.
   * @param kind The kind of the task.
   * @param factory Unique pointer to the task factory.
   * @return Error code indicating success or failure.
   */
  Error add_task_factory(const std::string &kind,
                         std::unique_ptr<TaskFactory> factory);

  /**
   * @brief Adds an accumulator factory.
   * @param kind The kind of the accumulator.
   * @param factory Unique pointer to the accumulator factory.
   * @return Error code indicating success or failure.
   */
  Error add_accumulator_factory(const std::string &kind,
                                std::unique_ptr<AccumulatorFactory> factory);

  /**
   * @brief Adds a task to the model.
   * @param kind The kind of the task.
   * @param name The name of the task.
   * @param params The parameters of the task.
   * @return Error code indicating success or failure.
   */
  Error add_task(const std::string &kind, const std::string &name,
                 const json &params);

  /**
   * @brief Adds an accumulator to the model.
   * @param kind The kind of the accumulator.
   * @param name The name of the accumulator.
   * @param params The parameters of the accumulator.
   * @return Error code indicating success or failure.
   */
  Error add_accumulator(const std::string &kind, const std::string &name,
                        const json &params);

  /**
   * @brief Sets the root accumulator of the model.
   * @param name The name of the root accumulator.
   * @return Error code indicating success or failure.
   */
  Error set_root_accumulator(const std::string &name);

  /**
   * @brief Adds a child relationship between nodes.
   * @param parent_name The name of the parent node.
   * @param child_name The name of the child node.
   * @return Error code indicating success or failure.
   */
  Error add_child(const std::string &parent_name,
                  const std::string &child_name);

  /**
   * @brief Retrieves the task factories.
   * @return A reference to the task factory map.
   */
  const TaskFactoryMap &task_factories() const;

  /**
   * @brief Retrieves the accumulator factories.
   * @return A reference to the accumulator factory map.
   */
  const AccumulatorFactoryMap &accumulator_factories() const;

  /**
   * @brief Retrieves the root node.
   * @return Pointer to the root node.
   */
  ModelDescriptorNode *root() const;

private:
  /// Maps task kinds to factories.
  TaskFactoryMap task_factories_map_;

  /// Maps accumulator kinds to factories.
  AccumulatorFactoryMap accumulator_factories_map_;

  /// Maps task names to (kind, params).
  TaskMap tasks_;

  /// Maps accumulator names to (kind, params).
  AccumulatorMap accumulators_;

  /// Root node (non-owning).
  ModelDescriptorNode *root_ = nullptr;

  /// Stores task factories.
  std::vector<std::unique_ptr<TaskFactory>> task_factories_;

  /// Stores accumulator factories.
  std::vector<std::unique_ptr<AccumulatorFactory>> accumulator_factories_;

  /// Stores model nodes.
  std::vector<std::unique_ptr<ModelDescriptorNode>> nodes_;
};

} // namespace dh
