#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <tl/expected.hpp>

#include "curaii/cuda_runtime.hh"
#include "holoflow/v2/error.hh"
#include "holoflow/v2/model.hh"

using json = nlohmann::json;

namespace dh::v2 {

class ModelTransaction {
public:
  enum class RemoveNodeBehavior {
    RemoveSubtree,
    OrphanChildren,
    ReparentChildren,
  };

  enum class DisconnectNodeBehavior {
    OrphanChild,
    ReparentChildren,
  };

  ModelTransaction &add_source(const std::string &name, const std::string &kind,
                               const json &params);

  ModelTransaction &add_sink(const std::string &name, const std::string &kind,
                             const json &params);

  ModelTransaction &add_task(const std::string &name, const std::string &kind,
                             const json &params);

  ModelTransaction &add_accumulator(const std::string &name,
                                    const std::string &kind,
                                    const json &params);

  ModelTransaction &update_node_params(const std::string &name,
                                       const json &params);

  ModelTransaction &remove_node(const std::string &name,
                                RemoveNodeBehavior behavior);

  ModelTransaction &connect(const std::string &parent_name,
                            const std::string &child_name);

  ModelTransaction &disconnect(const std::string &parent_name,
                               const std::string &child_name,
                               DisconnectNodeBehavior behavior);

  bool has_source(const std::string &name) const;
  bool has_sink(const std::string &name) const;
  bool has_task(const std::string &name) const;
  bool has_accumulator(const std::string &name) const;
  bool has_node(const std::string &name) const;

  tl::expected<void, Error> commit();

private:
  struct Node {
    std::optional<std::string> name_;
    std::optional<std::string> kind_;
    std::optional<json> params_;
    std::optional<CudaStreamRef> stream_;
    std::vector<std::reference_wrapper<Node>> children_;
  };

  struct TaskNode : Node {
    std::optional<int> itens_id_;
    std::optional<int> otens_id_;
    std::unique_ptr<Task> task_;
    std::optional<TaskMeta> task_meta_;
  };

  struct AccumulatorNode : Node {
    std::optional<int> itens_id_;
    std::optional<int> otens_id_;
    std::unique_ptr<Accumulator> accumulator_;
    std::optional<AccumulatorMeta> accumulator_meta_;
  };

  struct SourceNode : Node {
    std::optional<int> otens_id_;
    std::unique_ptr<Source> source_;
    std::optional<SourceMeta> source_meta_;
  };

  struct SinkNode : Node {
    std::optional<int> itens_id_;
    std::unique_ptr<Sink> sink_;
    std::optional<SinkMeta> sink_meta_;
  };

  explicit ModelTransaction(Model &model);

  void validate_orphan_nodes();
  void validate_childless_nodes();
  void validate_source_nodes();
  void validate_sink_nodes();

  Model &model_;

  Node *root_;
  std::vector<std::unique_ptr<Node>> nodes_;

  std::vector<Error> errors_;

  friend class Model;
};

} // namespace dh::v2