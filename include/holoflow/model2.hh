#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoflow/accumulator.hh"
#include "holoflow/error.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/tensor.hh"

namespace dh::v2 {

enum class ErrorType {
  NotFound,
  InvalidArgument,
  Validation,
  Internal,
  NotImplemented,
};

// Forward declaration
class Model;
class ModelTransaction;

class ModelTransaction {
public:
  enum class RemoveNodeBehavior {
    RemoveSubtree,
    OrphanChildren,
    ReparentChildren,
  };

  enum class DisconnectNodeBehavior {
    RemoveSubtree,
    OrphanChildren,
    ReparentChildren,
  };

  ModelTransaction(Model &model);

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

  tl::expected<void, Error> commit();

private:
  Model &model_;
};

} // namespace dh::v2