#pragma once

#include "curaii/curaii.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/tensor.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/graph.hh"

namespace holoflow::model {

struct TensorSlot {

  TensorSlot(dh::TensorMeta meta, dh::unique_host_ptr<uint8_t> h = nullptr,
             dh::unique_device_ptr<uint8_t> d = nullptr);

  dh::TensorView view();

  dh::TensorMeta meta;
  dh::unique_host_ptr<uint8_t> host_data;
  dh::unique_device_ptr<uint8_t> device_data;
  uint8_t *data;
};

class Model {
private:
  Model() = default;

  std::vector<dh::CudaStream> streams_;
  std::vector<std::unique_ptr<dh::Task>> tasks_;
  std::vector<std::unique_ptr<dh::Accumulator>> accumulators_;
  std::vector<std::unique_ptr<dh::Source>> sources_;
  std::vector<std::unique_ptr<dh::Sink>> sinks_;
  Graph graph_;
  std::map<std::string, Vertex> node_map_;
  std::map<int, TensorSlot> tensor_slots_;
  std::vector<size_t> pes_roots_;
  int next_tens_id_ = 0;

  friend class ModelCompiler;
  friend class Runner;
};

} // namespace holoflow::model