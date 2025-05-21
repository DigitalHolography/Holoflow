#pragma once

#include <cuComplex.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <utility>

#include "curaii/v2/cuda.hh"
#include "curaii/v2/cufft.hh"
#include "holoflow/task.hh"
#include "holoflow/tensor.hh"

using json                          = nlohmann::json;
using CufftHandle                   = curaii::cufft::Handle;
template <typename T> using DevPtr  = curaii::cuda::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::cuda::unique_host_ptr<T>;

namespace holovibes::tasks {

struct RegistrationParams {
  float radius;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RegistrationParams, radius);
};

class Registration : public dh::Task {
public:
  void run(dh::TensorView input, dh::TensorView output) override;

  friend class RegistrationFactory;

private:
  Registration(const dh::TaskMeta &meta, cudaStream_t stream,
               // Config
               float radius,
               // Center mean
               DevPtr<float> d_mean_centered,
               // Cross correlation
               bool ref_initialized, size_t freq_size, CufftHandle r2c_handle,
               CufftHandle c2r_handle, DevPtr<float> d_ref,
               DevPtr<float> d_xcorr, DevPtr<cuFloatComplex> d_freq1,
               DevPtr<cuFloatComplex> d_freq2,
               // CUB sum
               size_t sum_tmp_bytes, DevPtr<uint8_t> d_sum_tmp,
               DevPtr<float> d_sum,
               // CUB argmax
               size_t amax_tmp_bytes, DevPtr<uint8_t> d_amax_tmp,
               DevPtr<float> d_max, DevPtr<int64_t> d_max_idx,
               // CUB select
               size_t select_tmp_bytes, DevPtr<uint8_t> d_select_tmp,
               DevPtr<int> d_select_count, DevPtr<uint8_t> d_select_roi,
               DevPtr<float> d_selected);
  void cross_correlation(float *odata, const float *idata);
  void mask_circle(float *odata, const float *idata);
  void center_mean(float *odata, const float *idata);
  std::pair<int64_t, int64_t> get_shifts(float *xcorr);
  void apply_shifts(float *odata, const float *idata, int64_t shift_x,
                    int64_t shift_y);

  // Config
  float radius_;

  // Center mean
  DevPtr<float> d_mean_centered_;

  // Cross correlation
  bool                   ref_initialized_;
  size_t                 freq_size_;
  CufftHandle            r2c_handle_;
  CufftHandle            c2r_handle_;
  DevPtr<float>          d_ref_;
  DevPtr<float>          d_xcorr_;
  DevPtr<cuFloatComplex> d_freq1_;
  DevPtr<cuFloatComplex> d_freq2_;

  // CUB sum
  size_t          sum_tmp_bytes_;
  DevPtr<uint8_t> d_sum_tmp_;
  DevPtr<float>   d_sum_;

  // CUB argmax
  size_t          amax_tmp_bytes_;
  DevPtr<uint8_t> d_amax_tmp_;
  DevPtr<float>   d_max_;
  DevPtr<int64_t> d_max_idx_;

  // CUB select
  size_t          select_tmp_bytes_;
  DevPtr<uint8_t> d_select_tmp_;
  DevPtr<int>     d_select_count_;
  DevPtr<uint8_t> d_select_roi_;
  DevPtr<float>   d_selected_;
};

class RegistrationFactory : public dh::TaskFactory {
public:
  dh::TaskMeta type_check(const dh::TensorMeta &imeta,
                          const json           &params) override;

  std::unique_ptr<dh::Task> create(const dh::TensorMeta &imeta,
                                   const json           &params,
                                   cudaStream_t          stream) override;
};

} // namespace holovibes::tasks