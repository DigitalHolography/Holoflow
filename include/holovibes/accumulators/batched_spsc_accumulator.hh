#pragma once

#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/error.hh"
#include "holoflow/tensor.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 128
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

namespace holovibes::accumulators {

struct BatchedSPSCParams {
  size_t nb_slots{2};
  size_t dequeue_batch_size{1};
  std::optional<size_t> stride{std::nullopt};
};

void to_json(nlohmann::json &j, const BatchedSPSCParams &p);
void from_json(const nlohmann::json &j, BatchedSPSCParams &p);

class BatchedSPSC : public dh::Accumulator {
public:
  ~BatchedSPSC() = default;

  std::optional<dh::TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<dh::TensorView> read_tensor() override;

  void commit_read() override;

  size_t size();

  void reset();

  void fill();

  friend class BatchedSPSCFactory;

private:
  BatchedSPSC(const dh::AccumulatorMeta &meta, cudaStream_t stream,
              dh::Accumulator::EventListeners event_listeners, size_t nb_slots,
              size_t stride, curaii::cuda::unique_host_ptr<uint8_t> host_buffer,
              curaii::cuda::unique_device_ptr<uint8_t> device_buffer);

  size_t writer_size();

  size_t reader_size();

  /// The number of slots in the circular buffer.
  size_t nb_slots_;

  /// The number of elements to be enqueued in a single batch.
  size_t enqueue_batch_size_;

  /// The number of elements to be dequeued in a single batch.
  size_t dequeue_batch_size_;

  /// The size of each element in bytes.
  size_t element_size_;

  /// The dequeue stride.
  size_t stride_;

  /// A pre-allocated memory block for storing elements.
  uint8_t *buffer_;

  curaii::cuda::unique_host_ptr<uint8_t> host_buffer_;

  curaii::cuda::unique_device_ptr<uint8_t> device_buffer_;

  /// The current write index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;

  /// The current read index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

class BatchedSPSCFactory : public dh::AccumulatorFactory {
public:
  dh::AccumulatorMeta type_check(const dh::TensorMeta &imeta,
                                 const json &params) override;

  std::unique_ptr<dh::Accumulator>
  create(const dh::TensorMeta &imeta, const json &params, cudaStream_t stream,
         dh::Accumulator::EventListeners event_listeners) override;
};

} // namespace holovibes::accumulators

#ifdef _MSC_VER
#pragma warning(pop)
#endif