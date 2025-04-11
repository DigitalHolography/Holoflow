#pragma once

#include <atomic>
#include <boost/graph/adjacency_list.hpp>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "curaii/cuda_runtime.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v3/model/descriptor.hh"

namespace holoflow::model {

struct MetricsProperties {
  std::shared_ptr<std::atomic<int>> num_executions_{0};
  std::chrono::steady_clock::time_point last_reset_time_;
  std::shared_ptr<std::atomic<int>> execution_throughput_{0};
};

struct CommonProperties {
  std::optional<dh::CudaStreamRef> stream_;
};

struct SourceProperties {
  std::optional<int> otens_id_;
  dh::Source *source_;
  std::optional<dh::SourceMeta> source_meta_;
};

struct SinkProperties {
  std::optional<int> itens_id_;
  dh::Sink *sink_;
  std::optional<dh::SinkMeta> sink_meta_;
};

struct TaskProperties {
  std::optional<int> itens_id_;
  std::optional<int> otens_id_;
  dh::Task *task_;
  std::optional<dh::TaskMeta> task_meta_;
};

struct AccumulatorProperties {
  std::optional<int> itens_id_;
  std::optional<int> otens_id_;
  dh::Accumulator *accumulator_;
  std::optional<dh::AccumulatorMeta> accumulator_meta_;
};

using TypeSpecificProperties =
    std::variant<SourceProperties, SinkProperties, TaskProperties,
                 AccumulatorProperties>;

enum class NodeKind {
  Source,
  Sink,
  Task,
  Accumulator,
};

struct NodeProperties {
  DescriptorNodeProperties descriptor_;
  NodeKind kind_;
  CommonProperties common_;
  MetricsProperties metrics_;
  TypeSpecificProperties type_specific_;
};

struct EdgeProperties {};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              NodeProperties, EdgeProperties>
    Graph;

typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;

} // namespace holoflow::model