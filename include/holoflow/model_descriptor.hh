#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

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
   *
   * @return Reference to the node's name.
   */
  std::string &name();

  /**
   * @brief Gets the immutable name of the node.
   *
   * @return Constant reference to the node's name.
   */
  const std::string &name() const;

  /**
   * @brief Gets the mutable kind of the node.
   *
   * @return Reference to the node's kind.
   */
  std::string &kind();

  /**
   * @brief Gets the immutable kind of the node.
   *
   * @return Constant reference to the node's kind.
   */
  const std::string &kind() const;

  /**
   * @brief Gets the mutable parameters of the node.
   *
   * @return Reference to the JSON object representing node parameters.
   */
  json &params();

  /**
   * @brief Gets the immutable parameters of the node.
   *
   * @return Constant reference to the JSON object representing node parameters.
   */
  const json &params() const;

  /**
   * @brief Adds a child node.
   *
   * Transfers ownership of the given node to this parent node.
   *
   * @param child Unique pointer to the child node.
   */
  void add_child(std::unique_ptr<ModelDescriptorNode> child);

  /**
   * @brief Removes a child node by pointer.
   *
   * @param child Pointer to the child node to remove.
   * @return True if the node was found and removed, false otherwise.
   */
  bool remove_child(const ModelDescriptorNode *child);

  /**
   * @brief Removes a child node by index.
   *
   * @param index The index of the child node to remove.
   * @return True if the node was successfully removed, false otherwise.
   */
  bool remove_child(size_t index);

  /**
   * @brief Returns a modifiable view of child nodes.
   *
   * Allows modification of children but does not allow adding/removing them.
   *
   * @return A span of raw pointers to child nodes.
   */
  std::span<ModelDescriptorNode *> children();

  /**
   * @brief Returns a read-only view of child nodes.
   *
   * Provides access to child nodes without allowing modification.
   *
   * @return A span of constant raw pointers to child nodes.
   */
  std::span<const ModelDescriptorNode *> children() const;

private:
  std::string name_; ///< The name of the node.
  std::string kind_; ///< The kind/type of the node.
  json params_;      ///< JSON object storing the node parameters.
  std::vector<std::unique_ptr<ModelDescriptorNode>>
      children_; ///< List of child nodes.
};

/**
 * @brief Represents a task node in the model descriptor tree.
 *
 * This node type corresponds to computational tasks.
 */
class TaskDescriptorNode : public ModelDescriptorNode {
public:
  /**
   * @brief Accepts a visitor for the Visitor pattern.
   *
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
   * @brief Accepts a visitor for the Visitor pattern.
   *
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
   *
   * @param node The task node being visited.
   */
  virtual void visit(TaskDescriptorNode &node) = 0;

  /**
   * @brief Visits an AccumulatorDescriptorNode.
   *
   * @param node The accumulator node being visited.
   */
  virtual void visit(AccumulatorDescriptorNode &node) = 0;
};

} // namespace dh
