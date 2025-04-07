#include "holoflow/v2/model.hh"

#include <memory>
#include <tl/expected.hpp>

#include "holoflow/accumulator.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v2/error.hh"
#include "holoflow/v2/model_transaction.hh"

namespace dh::v2 {

// ==========================================================================
//                     Model Implementation
// ==========================================================================

Model::Model() {}

tl::expected<std::unique_ptr<Model>, Error> Model::create() {
  auto *model = new Model();
  auto model_ptr = std::unique_ptr<Model>(model);
  return model_ptr;
}

tl::expected<void, Error>
Model::register_source_factory(const std::string &kind,
                               std::unique_ptr<SourceFactory> factory) {
  holoflow_logger()->trace(
      "[Model::register_source_factory] Registering source factory: {}", kind);

  if (has_factory(kind)) {
    holoflow_logger()->warn(
        "[Model::register_source_factory] Factory {} is already "
        "registered",
        kind);

    return tl::unexpected(
        Error::make(ErrorType::InternalError, "Factory already registered"));
  }

  source_factories_map_.emplace(kind, std::ref(*factory));
  source_factories_.push_back(std::move(factory));
  return {};
}

tl::expected<void, Error>
Model::register_sink_factory(const std::string &kind,
                             std::unique_ptr<SinkFactory> factory) {
  holoflow_logger()->trace(
      "[Model::register_sink_factory] Registering sink factory: {}", kind);

  if (has_factory(kind)) {
    holoflow_logger()->warn(
        "[Model::register_sink_factory] Factory {} is already "
        "registered",
        kind);

    return tl::unexpected(
        Error::make(ErrorType::InternalError, "Factory already registered"));
  }

  sink_factories_map_.emplace(kind, std::ref(*factory));
  sink_factories_.push_back(std::move(factory));
  return {};
}

tl::expected<void, Error>
Model::register_task_factory(const std::string &kind,
                             std::unique_ptr<TaskFactory> factory) {
  holoflow_logger()->trace(
      "[Model::register_task_factory] Registering task factory: {}", kind);

  if (has_factory(kind)) {
    holoflow_logger()->warn(
        "[Model::register_task_factory] Factory {} is already "
        "registered",
        kind);

    return tl::unexpected(
        Error::make(ErrorType::InternalError, "Factory already registered"));
  }

  task_factories_map_.emplace(kind, std::ref(*factory));
  task_factories_.push_back(std::move(factory));
  return {};
}

tl::expected<void, Error> Model::register_accumulator_factory(
    const std::string &kind, std::unique_ptr<AccumulatorFactory> factory) {
  holoflow_logger()->trace(
      "[Model::register_accumulator_factory] Registering accumulator "
      "factory: {}",
      kind);

  if (has_factory(kind)) {
    holoflow_logger()->warn(
        "[Model::register_accumulator_factory] Factory {} is already "
        "registered",
        kind);

    return tl::unexpected(
        Error::make(ErrorType::InternalError, "Factory already registered"));
  }

  accumulator_factories_map_.emplace(kind, std::ref(*factory));
  accumulator_factories_.push_back(std::move(factory));
  return {};
}

bool Model::has_task_factory(const std::string &kind) const {
  return task_factories_map_.contains(kind);
}

bool Model::has_accumulator_factory(const std::string &kind) const {
  return accumulator_factories_map_.contains(kind);
}

bool Model::has_source_factory(const std::string &kind) const {
  return source_factories_map_.contains(kind);
}

bool Model::has_sink_factory(const std::string &kind) const {
  return sink_factories_map_.contains(kind);
}

bool Model::has_factory(const std::string &kind) const {
  return has_task_factory(kind) || has_accumulator_factory(kind) ||
         has_source_factory(kind) || has_sink_factory(kind);
}

ModelTransaction Model::begin_transaction() {
  // TODO: Check if an existing transaction is in progress
  holoflow_logger()->trace("[Model::begin_transaction] Creating transaction");

  return ModelTransaction(*this);
}

} // namespace dh::v2