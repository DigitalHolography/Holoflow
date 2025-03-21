#include "holovibes/accumulators/batched_spsc_accumulator.hh"

namespace dh {

// ==========================================================================
//                     BatchedSPSCAccumulator Implementation
// ==========================================================================

BatchedSPSCAccumulator::BatchedSPSCAccumulator(
    const AccumulatorMeta &meta, cudaStream_t stream, size_t nb_slots,
    unique_host_ptr<uint8_t> host_buffer,
    unique_device_ptr<uint8_t> device_buffer)
    : Accumulator(meta, stream) {
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
}

tl::expected<std::optional<TensorView>, Error>
BatchedSPSCAccumulator::write_tensor() {
  if (nb_slots_ - writer_size() < enqueue_batch_size_ + 1) {
    return std::nullopt;
  }

  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *data = buffer_ + write_idx * element_size_;
  return TensorView(data, meta_.imeta());
}

tl::expected<void, Error> BatchedSPSCAccumulator::commit_write() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t next_write_idx = write_idx + enqueue_batch_size_;
  if (next_write_idx == nb_slots_)
    next_write_idx = 0;

  write_idx_.store(next_write_idx, std::memory_order_release);
  return {};
}

tl::expected<std::optional<TensorView>, Error>
BatchedSPSCAccumulator::read_tensor() {
  if (reader_size() < dequeue_batch_size_)
    return std::nullopt;

  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  uint8_t *data = buffer_ + read_idx * element_size_;
  return TensorView(data, meta_.ometa());
}

tl::expected<void, Error> BatchedSPSCAccumulator::commit_read() {
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + dequeue_batch_size_;
  if (next_read_idx == nb_slots_)
    next_read_idx = 0;

  read_idx_.store(next_read_idx, std::memory_order_release);
  return {};
}

tl::expected<size_t, Error> BatchedSPSCAccumulator::size() {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx = read_idx_.load(std::memory_order_acquire);

  size_t diff = write_idx - read_idx;

  if (write_idx < read_idx)
    diff += nb_slots_;

  return diff;
}

tl::expected<void, Error> BatchedSPSCAccumulator::reset() {
  write_idx_.store(0, std::memory_order_release);
  read_idx_.store(0, std::memory_order_release);
  return {};
}

tl::expected<void, Error> BatchedSPSCAccumulator::fill() {
  write_idx_.store(nb_slots_, std::memory_order_release);
  read_idx_.store(0, std::memory_order_release);
  return {};
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

tl::expected<AccumulatorMeta, Error>
BatchedSPSCAccumulatorFactory::type_check(const TensorMeta &imeta,
                                          const json &jparams) {
  auto params = jparams.get<Params>();

  if (params.nb_slots <= 0) {
    LOG(WARNING) << "Invalid nb_slots: " << params.nb_slots;
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.dequeue_batch_size <= 0) {
    LOG(WARNING) << "Invalid dequeue_batch_size: " << params.dequeue_batch_size;
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().size() < 2) {
    LOG(WARNING) << "imeta.shape().size() < 2: " << imeta.shape().size();
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.nb_slots % imeta.shape().at(0) != 0) {
    LOG(WARNING) << "params.nb_slots %% imeta.shape.at(0) != 0: "
                 << params.nb_slots << " " << imeta.shape().at(0);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.nb_slots % params.dequeue_batch_size != 0) {
    LOG(WARNING) << "params.nb_slots %% params.dequeue_batch_size != 0: "
                 << params.nb_slots << " " << params.dequeue_batch_size;
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto oshape = imeta.shape();
  oshape.at(0) = params.dequeue_batch_size;
  TensorMeta ometa(imeta.data_type(), imeta.memory_location(), oshape,
                   imeta.strides());

  return AccumulatorMeta(imeta, ometa);
}

tl::expected<std::unique_ptr<Accumulator>, Error>
BatchedSPSCAccumulatorFactory::create(const TensorMeta &imeta,
                                      const json &jparams,
                                      cudaStream_t stream) {

  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    LOG(WARNING) << "type check failed";
    return tl::unexpected(meta_result.error());
  }

  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  auto enqueue_batch_size = meta.imeta().shape().at(0);
  auto element_size = meta.imeta().size_in_bytes() / enqueue_batch_size;
  auto buffer_size = params.nb_slots * element_size;
  unique_host_ptr<uint8_t> host_buffer = nullptr;
  unique_device_ptr<uint8_t> device_buffer = nullptr;

  switch (meta.imeta().memory_location()) {
  case MemoryLocation::HOST:
    host_buffer = make_unique_host_ptr<uint8_t>(buffer_size);
    break;
  case MemoryLocation::DEVICE:
    device_buffer = make_unique_device_ptr<uint8_t>(buffer_size);
    break;
  }

  auto *accumulator = new BatchedSPSCAccumulator(meta, stream, params.nb_slots,
                                                 std::move(host_buffer),
                                                 std::move(device_buffer));

  return std::unique_ptr<BatchedSPSCAccumulator>(accumulator);
}

} // namespace dh