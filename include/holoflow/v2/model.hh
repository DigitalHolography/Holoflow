#pragma once

#include <memory>
#include <tl/expected.hpp>

#include "holoflow/accumulator.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v2/error.hh"

namespace dh::v2 {

class ModelTransaction;

class Model {
public:
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(const Model &&) = delete;
  Model &operator=(const Model &&) = delete;

  [[nodiscard]]
  static tl::expected<std::unique_ptr<Model>, Error> create();

  [[nodiscard]]
  tl::expected<void, Error>
  register_task_factory(const std::string &kind,
                        std::unique_ptr<TaskFactory> factory);

  [[nodiscard]]
  tl::expected<void, Error>
  register_accumulator_factory(const std::string &kind,
                               std::unique_ptr<AccumulatorFactory> factory);

  [[nodiscard]]
  tl::expected<void, Error>
  register_source_factory(const std::string &kind,
                          std::unique_ptr<SourceFactory> factory);

  [[nodiscard]]
  tl::expected<void, Error>
  register_sink_factory(const std::string &kind,
                        std::unique_ptr<SinkFactory> factory);

  bool has_task_factory(const std::string &kind) const;
  bool has_accumulator_factory(const std::string &kind) const;
  bool has_source_factory(const std::string &kind) const;
  bool has_sink_factory(const std::string &kind) const;
  bool has_factory(const std::string &kind) const;

  void start();
  void stop();

  ModelTransaction begin_transaction();

private:
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
   * @brief Map from kind to source factory.
   */
  using SourceFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SourceFactory>>;

  /**
   * @brief Map from kind to sink factory.
   */
  using SinkFactoryMap =
      std::unordered_map<std::string, std::reference_wrapper<SinkFactory>>;

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
   * @brief Map from name to (kind, params).
   */
  using SourceMap =
      std::unordered_map<std::string, std::pair<std::string, json>>;

  /**
   * @brief Map from name to (kind, params).
   */
  using SinkMap = std::unordered_map<std::string, std::pair<std::string, json>>;

  Model();

  TaskFactoryMap task_factories_map_;
  AccumulatorFactoryMap accumulator_factories_map_;
  SourceFactoryMap source_factories_map_;
  SinkFactoryMap sink_factories_map_;

  std::vector<std::unique_ptr<TaskFactory>> task_factories_;
  std::vector<std::unique_ptr<AccumulatorFactory>> accumulator_factories_;
  std::vector<std::unique_ptr<SourceFactory>> source_factories_;
  std::vector<std::unique_ptr<SinkFactory>> sink_factories_;

  Model::TaskMap tasks_map_;
  Model::AccumulatorMap accumulators_map_;
  Model::SourceMap sources_map_;
  Model::SinkMap sinks_map_;

  std::vector<std::unique_ptr<Task>> tasks_;
  std::vector<std::unique_ptr<Accumulator>> accumulators_;
  std::vector<std::unique_ptr<Source>> sources_;
  std::vector<std::unique_ptr<Sink>> sinks_;

  friend class ModelTransaction;
};

} // namespace dh::v2