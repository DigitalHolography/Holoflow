#pragma once

#include "curaii/curaii.hh"
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

  Graph graph_;
  std::map<std::string, Vertex> node_map_;
  std::vector<dh::CudaStream> streams_;
  int next_tens_id_ = 0;
  std::map<int, TensorSlot> tensor_slots_;
  std::vector<size_t> pes_roots_;

  friend class ModelCompiler;
  friend class Runner;
};

} // namespace holoflow::model