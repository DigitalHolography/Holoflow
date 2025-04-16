#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "curaii/cuda_runtime.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/graph.hh"
#include "holoflow/v3/model/model.hh"

namespace holoflow::model {

class ModelCompiler {
public:
  void add_factory(const std::string &type,
                   std::unique_ptr<dh::SourceFactory> factory);

  void add_factory(const std::string &type,
                   std::unique_ptr<dh::SinkFactory> factory);

  void add_factory(const std::string &type,
                   std::unique_ptr<dh::TaskFactory> factory);

  void add_factory(const std::string &type,
                   std::unique_ptr<dh::AccumulatorFactory> factory);

  Model compile(const DescriptorGraph &graph);

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