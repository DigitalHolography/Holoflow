#pragma once

#include "curaii/curaii.hh"
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

namespace dh {

class BatchedSPSCAccumulator : public Accumulator {
public:
  ~BatchedSPSCAccumulator() = default;

  std::optional<TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<TensorView> read_tensor() override;

  void commit_read() override;

  size_t size();

  void reset();

  void fill();

  friend class BatchedSPSCAccumulatorFactory;

private:
  BatchedSPSCAccumulator(const AccumulatorMeta &meta, CudaStreamRef stream,
                         size_t nb_slots, unique_host_ptr<uint8_t> host_buffer,
                         unique_device_ptr<uint8_t> device_buffer);

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

  /// A pre-allocated memory block for storing elements.
  uint8_t *buffer_;

  unique_host_ptr<uint8_t> host_buffer_;

  unique_device_ptr<uint8_t> device_buffer_;

  /// The current write index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;

  /// The current read index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

class BatchedSPSCAccumulatorFactory : public AccumulatorFactory {
public:
  AccumulatorMeta type_check(const TensorMeta &imeta,
                             const json &params) override;

  std::unique_ptr<Accumulator> create(const TensorMeta &imeta,
                                      const json &params,
                                      CudaStreamRef stream) override;

private:
  struct Params {
    size_t nb_slots;
    size_t dequeue_batch_size;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, nb_slots, dequeue_batch_size);
  };
};

} // namespace dh

#ifdef _MSC_VER
#pragma warning(pop)
#endif