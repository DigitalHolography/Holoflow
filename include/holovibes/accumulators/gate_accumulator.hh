#pragma once

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/tensor.hh"

namespace holovibes::accumulators {

struct GateParams {
  bool is_on;
  std::optional<size_t> target;
};

void to_json(nlohmann::json &j, const GateParams &p);
void from_json(const nlohmann::json &j, GateParams &p);

class Gate : public dh::Accumulator {
public:
  std::optional<dh::TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<dh::TensorView> read_tensor() override;

  void commit_read() override;

  void handle_event(const json &event) override;

  friend class GateFactory;

private:
  Gate(const dh::AccumulatorMeta &meta, cudaStream_t stream,
       dh::Accumulator::EventListeners event_listeners, bool is_on,
       std::optional<size_t> target,
       curaii::cuda::unique_host_ptr<uint8_t> h_buffer,
       curaii::cuda::unique_device_ptr<uint8_t> d_buffer);

  std::atomic<bool> write_;
  std::atomic<bool> is_on_;
  std::optional<size_t> target_;
  size_t frames_passed_;

  uint8_t *buffer_;
  curaii::cuda::unique_host_ptr<uint8_t> h_buffer_;
  curaii::cuda::unique_device_ptr<uint8_t> d_buffer_;
};

class GateFactory : public dh::AccumulatorFactory {
  dh::AccumulatorMeta type_check(const dh::TensorMeta &imeta,
                                 const json &params) override;

  std::unique_ptr<dh::Accumulator>
  create(const dh::TensorMeta &imeta, const json &params, cudaStream_t stream,
         dh::Accumulator::EventListeners event_listeners) override;
};

} // namespace holovibes::accumulators