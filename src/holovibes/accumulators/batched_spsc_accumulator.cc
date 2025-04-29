#include "holovibes/accumulators/batched_spsc_accumulator.hh"

#include <cassert>
#include <cstdlib>
#include <numeric>
#include <spdlog/spdlog.h>
#include <vector>

#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     BatchedSPSCAccumulator Implementation
// ==========================================================================

BatchedSPSCAccumulator::BatchedSPSCAccumulator(
    const AccumulatorMeta &meta, cudaStream_t stream,
    Accumulator::EventListeners event_listeners, size_t nb_slots,
    curaii::cuda::unique_host_ptr<uint8_t> host_buffer,
    curaii::cuda::unique_device_ptr<uint8_t> device_buffer)
    : Accumulator(meta, stream, event_listeners) {
  nb_slots_ = nb_slots;
  enqueue_batch_size_ = meta_.imeta().shape().at(0);
  dequeue_batch_size_ = meta_.ometa().shape().at(0);
  element_size_ = meta_.imeta().size_in_bytes() / enqueue_batch_size_;
  switch (meta_.imeta().memory_location()) {
  case MemoryLocation::HOST:
    buffer_ = host_buffer.get();
    break;
  case MemoryLocation::DEVICE:
    buffer_ = device_buffer.get();
    break;
  }
  host_buffer_ = std::move(host_buffer);
  device_buffer_ = std::move(device_buffer);
  write_idx_ = 0;
  read_idx_ = 0;

  holovibes_logger()->trace(
      "Constructed BatchedSPSCAccumulator: nb_slots={}, enqueue_batch_size={}, "
      "dequeue_batch_size={}, element_size={}",
      nb_slots_, enqueue_batch_size_, dequeue_batch_size_, element_size_);
}

std::optional<TensorView> BatchedSPSCAccumulator::write_tensor() {
  if (nb_slots_ - writer_size() < enqueue_batch_size_ + 1) {
    return std::nullopt;
  }

  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *data = buffer_ + write_idx * element_size_;
  return TensorView(data, meta_.imeta());
}

void BatchedSPSCAccumulator::commit_write() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t next_write_idx = write_idx + enqueue_batch_size_;
  if (next_write_idx == nb_slots_)
    next_write_idx = 0;

  write_idx_.store(next_write_idx, std::memory_order_release);
}

std::optional<TensorView> BatchedSPSCAccumulator::read_tensor() {
  if (reader_size() < dequeue_batch_size_)
    return std::nullopt;

  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  uint8_t *data = buffer_ + read_idx * element_size_;
  return TensorView(data, meta_.ometa());
}

void BatchedSPSCAccumulator::commit_read() {
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + dequeue_batch_size_;
  if (next_read_idx == nb_slots_)
    next_read_idx = 0;

  read_idx_.store(next_read_idx, std::memory_order_release);
}

size_t BatchedSPSCAccumulator::size() {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx = read_idx_.load(std::memory_order_acquire);

  size_t diff = write_idx - read_idx;
  if (write_idx < read_idx)
    diff += nb_slots_;

  return diff;
}

void BatchedSPSCAccumulator::reset() {
  write_idx_.store(0, std::memory_order_release);
  read_idx_.store(0, std::memory_order_release);
}

void BatchedSPSCAccumulator::fill() {
  write_idx_.store(nb_slots_, std::memory_order_release);
  read_idx_.store(0, std::memory_order_release);
}

size_t BatchedSPSCAccumulator::writer_size() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t read_idx = read_idx_.load(std::memory_order_acquire);

  size_t diff = write_idx - read_idx;
  if (write_idx < read_idx)
    diff += nb_slots_;

  return diff;
}

size_t BatchedSPSCAccumulator::reader_size() {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);

  size_t diff = write_idx - read_idx;
  if (write_idx < read_idx)
    diff += nb_slots_;

  return diff;
}

// ==========================================================================
//                     BatchedSPSCAccumulatorFactory Implementation
// ==========================================================================

AccumulatorMeta
BatchedSPSCAccumulatorFactory::type_check(const TensorMeta &imeta,
                                          const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.nb_slots > 0, "nb_slots <= 0");
  check(params.dequeue_batch_size > 0, "dequeue_batch_size <= 0");
  check(params.nb_slots % params.dequeue_batch_size == 0,
        "dequeue_batch_size is not a factor of nb_slots");

  // 2) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(params.nb_slots % imeta.shape().at(0) == 0,
        "input dim 0 is not a factor of nb_slots");

  // 3) Success
  auto oshape = imeta.shape();
  oshape.at(0) = params.dequeue_batch_size;
  TensorMeta ometa(imeta.data_type(), imeta.memory_location(), oshape,
                   imeta.strides());

  return AccumulatorMeta(imeta, ometa);
}

std::unique_ptr<Accumulator> BatchedSPSCAccumulatorFactory::create(
    const TensorMeta &imeta, const json &jparams, cudaStream_t stream,
    Accumulator::EventListeners event_listeners) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  // 2) Buffer size
  auto enqueue_batch_size = meta.imeta().shape().at(0);
  auto element_size = meta.imeta().size_in_bytes() / enqueue_batch_size;
  auto buffer_size = params.nb_slots * element_size;

  // 3) Allocation
  curaii::cuda::unique_host_ptr<uint8_t> host_buffer = nullptr;
  curaii::cuda::unique_device_ptr<uint8_t> device_buffer = nullptr;
  switch (meta.imeta().memory_location()) {
  case MemoryLocation::HOST:
    host_buffer = curaii::cuda::make_unique_host_ptr<uint8_t>(buffer_size);
    break;
  case MemoryLocation::DEVICE:
    device_buffer =
        curaii::cuda::make_unique_device_ptr<uint8_t>(buffer_size, stream);
    break;
  }

  // 4) Assemble accumulator
  auto *accumulator = new BatchedSPSCAccumulator(
      meta, stream, event_listeners, params.nb_slots, std::move(host_buffer),
      std::move(device_buffer));

  return std::unique_ptr<BatchedSPSCAccumulator>(accumulator);
}

} // namespace dh
