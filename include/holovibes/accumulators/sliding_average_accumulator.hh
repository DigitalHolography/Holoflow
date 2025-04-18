#pragma once

#include <tl/expected.hpp>

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

class SlidingAverageAccumulator : public Accumulator {
public:
  std::optional<TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<TensorView> read_tensor() override;

  void commit_read() override;

  friend class SlidingAverageAccumulatorFactory;

private:
  SlidingAverageAccumulator(const AccumulatorMeta &meta, CudaStreamRef stream,
                            size_t nb_slots, size_t window_size,
                            unique_device_ptr<uint8_t> d_buffer,
                            unique_device_ptr<uint8_t> d_avg_frame);

  size_t writer_size();

  size_t reader_size();

  size_t window_size_;

  /// The number of slots in the circular buffer.
  size_t nb_slots_;

  /// The size of each element in bytes.
  size_t element_size_;

  unique_device_ptr<uint8_t> d_buffer_;

  unique_device_ptr<uint8_t> d_running_avg_;

  alignas(CACHE_LINE_SIZE) std::atomic<size_t> avg_idx_;

  /// The current write index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;

  /// The current read index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

class SlidingAverageAccumulatorFactory : public AccumulatorFactory {
public:
  AccumulatorMeta type_check(const TensorMeta &imeta,
                             const json &params) override;

  std::unique_ptr<Accumulator> create(const TensorMeta &imeta,
                                      const json &params,
                                      CudaStreamRef stream) override;

private:
  struct Params {
    size_t nb_slots;
    size_t window_size;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, nb_slots, window_size);
  };
};

} // namespace dh

#ifdef _MSC_VER
#pragma warning(pop)
#endif