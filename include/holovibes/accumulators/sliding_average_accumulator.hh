#pragma once

#include <tl/expected.hpp>

#include "curaii/v2/cublas.hh"
#include "curaii/v2/cuda.hh"
#include "curaii/v2/cufft.hh"
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

using json                          = nlohmann::json;
template <typename T> using DevPtr  = curaii::cuda::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::cuda::unique_host_ptr<T>;

namespace dh {

class SlidingAverageAccumulator : public Accumulator {
public:
  std::optional<TensorView> write_tensor() override;
  void                      commit_write() override;

  std::optional<TensorView> read_tensor() override;
  void                      commit_read() override;

  friend class SlidingAverageAccumulatorFactory;

private:
  SlidingAverageAccumulator(
      const AccumulatorMeta &meta, cudaStream_t stream,
      Accumulator::EventListeners event_listeners, size_t nb_slots,
      size_t window_size, DevPtr<uint8_t> d_buffer, DevPtr<uint8_t> d_avg_frame,
      size_t freq_size, curaii::cufft::Handle r2c_handle,
      curaii::cufft::Handle c2r_handle, curaii::cublas::Handle handle,
      DevPtr<cuFloatComplex> d_freq1, DevPtr<cuFloatComplex> d_freq2,
      DevPtr<float> d_xcorr, DevPtr<float> d_shifted, DevPtr<float> d_ref,
      DevPtr<uint8_t> d_cub_sum_tmp, DevPtr<float> d_cub_sum,
      size_t cub_sum_tmp_bytes);

  size_t writer_size();
  size_t reader_size();
  void   cross_correlation(float *odata, float *idata1, float *idata2);
  void   registration(float *odata, float *idata);
  void   center_mean(float *iodata);

  // Config
  size_t window_size_;
  size_t nb_slots_;
  size_t element_size_;
  size_t freq_size_;
  bool   d_ref_initialized_;
  size_t cub_sum_tmp_bytes_;

  // Handles
  curaii::cufft::Handle  r2c_handle_;
  curaii::cufft::Handle  c2r_handle_;
  curaii::cublas::Handle handle_;

  // Buffers
  DevPtr<uint8_t>        d_buffer_;
  DevPtr<uint8_t>        d_running_avg_;
  DevPtr<cuFloatComplex> d_freq1_;
  DevPtr<cuFloatComplex> d_freq2_;
  DevPtr<float>          d_xcorr_;
  DevPtr<float>          d_shifted_;
  DevPtr<float>          d_ref_;
  DevPtr<uint8_t>        d_cub_sum_tmp_;
  DevPtr<float>          d_cub_sum_;

  // Atomics
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> avg_idx_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

class SlidingAverageAccumulatorFactory : public AccumulatorFactory {
public:
  AccumulatorMeta type_check(const TensorMeta &imeta,
                             const json       &params) override;

  std::unique_ptr<Accumulator>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream,
         Accumulator::EventListeners event_listeners) override;

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