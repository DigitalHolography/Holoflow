#pragma once

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "holoflow/accumulator.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/graph.hh"
#include "holoflow/v3/model/model.hh"

namespace holoflow::model {

template <typename T>
concept FactoryType = std::derived_from<T, dh::SourceFactory> ||
                      std::derived_from<T, dh::SinkFactory> ||
                      std::derived_from<T, dh::TaskFactory> ||
                      std::derived_from<T, dh::AccumulatorFactory>;

class ModelCompiler {
public:
  using EventListener = std::function<void(const nlohmann::json &)>;
  using EventListenerMap = std::map<std::string, std::vector<EventListener>>;

  template <FactoryType Factory, typename... Args>
  void add_factory(const std::string &type, Args &&...args) {
    dh::holoflow_logger()->trace(
        "[ModelCompiler::add_factory] Adding factory of type: {}", type);

    if (source_factories_.contains(type) || task_factories_.contains(type) ||
        sink_factories_.contains(type) ||
        accumulator_factories_.contains(type)) {
      throw std::runtime_error("Factory already exists for type: " + type);
    }

    auto factory = std::make_unique<Factory>(std::forward<Args>(args)...);

    if constexpr (std::derived_from<Factory, dh::SourceFactory>) {
      source_factories_[type] = std::move(factory);
    } else if constexpr (std::derived_from<Factory, dh::SinkFactory>) {
      sink_factories_[type] = std::move(factory);
    } else if constexpr (std::derived_from<Factory, dh::TaskFactory>) {
      task_factories_[type] = std::move(factory);
    } else if constexpr (std::derived_from<Factory, dh::AccumulatorFactory>) {
      accumulator_factories_[type] = std::move(factory);
    } else {
      static_assert(sizeof(Factory) == 0,
                    "Unknown factory type passed to add_factory.");
    }
  }

  Model compile(const DescriptorGraph &graph,
                EventListenerMap &event_listeners);

private:
  void build_compiler_graph(const DescriptorGraph &graph);

  void validate_no_orphan_nodes() const;

  void validate_single_parent_node() const;

  void validate_no_childless_nodes() const;

  void validate_sources_are_orphan() const;

  void validate_sinks_are_childless() const;

  void validate_source_is_unique() const;

  void assign_cuda_streams();

  void perform_type_checking();

  void validate_single_inlined_or_accumulator_child() const;

  void validate_non_inlined_child_between_accumulators() const;

  void reset_non_accumulator_tensor_ids();

  void assign_accumulator_tensor_ids();

  void assign_inlined_tensor_ids();

  void assign_non_inlined_tensor_ids();

  void call_factories();

  void create_tensor_slots();

  bool is_accumulator_tensor(int tens_id) const;

  void allocate_non_accumulator_tensor_slots();

  void select_pes_roots();

  Model model_;

  std::map<std::string, std::unique_ptr<dh::SourceFactory>> source_factories_;
  std::map<std::string, std::unique_ptr<dh::SinkFactory>> sink_factories_;
  std::map<std::string, std::unique_ptr<dh::TaskFactory>> task_factories_;
  std::map<std::string, std::unique_ptr<dh::AccumulatorFactory>>
      accumulator_factories_;
};

} // namespace holoflow::model