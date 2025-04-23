#pragma once

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <optional>

#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/tensor.hh"

namespace dh {

class GateAccumulator : public Accumulator {
public:
  std::optional<TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<TensorView> read_tensor() override;

  void commit_read() override;

  friend class GateAccumulatorFactory;

private:
  std::atomic<bool> is_on_;
  std::optional<size_t> target_;
  size_t frames_passed_;

  uint8_t *buffer_;
  curaii::cuda::unique_host_ptr<uint8_t> h_buffer_;
  curaii::cuda::unique_device_ptr<uint8_t> d_buffer_;
};

} // namespace dh