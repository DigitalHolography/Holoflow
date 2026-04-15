// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holonp/cross_correlation2.hh"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuComplex.h>
#include <math_constants.h>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"

namespace holonp {

void to_json(nlohmann::json &j, const CrossCorrelation2Settings::Ellipse &e) {
  j = nlohmann::json{
      {"cx", e.cx}, {"cy", e.cy}, {"rx", e.rx}, {"ry", e.ry}, {"angle", e.angle},
  };
}

void from_json(const nlohmann::json &j, CrossCorrelation2Settings::Ellipse &e) {
  if (j.contains("cx"))
    j.at("cx").get_to(e.cx);
  if (j.contains("cy"))
    j.at("cy").get_to(e.cy);
  if (j.contains("rx"))
    j.at("rx").get_to(e.rx);
  if (j.contains("ry"))
    j.at("ry").get_to(e.ry);
  if (j.contains("angle"))
    j.at("angle").get_to(e.angle);
}

void to_json(nlohmann::json &j, const CrossCorrelation2Settings &s) {
  j = nlohmann::json{{"axes", s.axes}, {"norm", s.norm}, {"roi", s.roi}};
}

void from_json(const nlohmann::json &j, CrossCorrelation2Settings &s) {
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
  else
    s.axes.clear();
  if (j.contains("norm"))
    j.at("norm").get_to(s.norm);
  else
    s.norm = FftNorm::Backward;
  if (j.contains("roi"))
    j.at("roi").get_to(s.roi);
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("CrossCorrelation2: " + msg);
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0)
    axis += ndim;
  return axis;
}

inline float inverse_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward)
    return static_cast<float>(1.0 / static_cast<double>(n_fft));
  if (norm == FftNorm::Forward)
    return 1.0f;
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft)));
}

std::array<int, 2> resolve_axes(const std::vector<int> &axes, int ndim) {
  check(ndim >= 2, "ndim must be >= 2");

  std::array<int, 2> resolved{};
  if (axes.empty()) {
    resolved[0] = ndim - 2;
    resolved[1] = ndim - 1;
  } else {
    check(axes.size() == 2, "expected exactly 2 axes");
    resolved[0] = normalize_axis(axes[0], ndim);
    resolved[1] = normalize_axis(axes[1], ndim);
  }

  check(resolved[0] >= 0 && resolved[0] < ndim, "axis 0 out of range");
  check(resolved[1] >= 0 && resolved[1] < ndim, "axis 1 out of range");
  check(resolved[0] < resolved[1], "axes must be ordered");
  check(resolved[0] == ndim - 2 && resolved[1] == ndim - 1,
        "only trailing axes (-2, -1) are supported");
  return resolved;
}

std::vector<size_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty())
    return desc.strides;

  std::vector<size_t> strides(desc.shape.size());
  size_t              acc = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
}

void generate_offsets_recursive(const std::vector<size_t> &shape,
                                const std::vector<size_t> &strides, int dim_idx,
                                size_t current_offset, std::vector<size_t> &out_offsets) {
  if (dim_idx < 0) {
    out_offsets.push_back(current_offset);
    return;
  }

  for (size_t i = 0; i < shape[static_cast<size_t>(dim_idx)]; ++i) {
    generate_offsets_recursive(shape, strides, dim_idx - 1,
                               current_offset + i * strides[static_cast<size_t>(dim_idx)],
                               out_offsets);
  }
}

size_t product(const std::vector<size_t> &shape, size_t start, size_t end) {
  size_t result = 1;
  for (size_t i = start; i < end; ++i)
    result *= shape[i];
  return result;
}

struct TensorLayout {
  std::vector<size_t> offsets;
  size_t              inner_batches;
  long long           idist;
  int                 istride;
  int                 inembed_w;
};

TensorLayout get_tensor_layout(const holoflow::core::TDesc &idesc) {
  const int    ndim  = static_cast<int>(idesc.shape.size());
  const size_t esize = holoflow::core::size_of(idesc.dtype);

  auto strides_bytes = get_strides_bytes(idesc);

  check(strides_bytes[static_cast<size_t>(ndim - 1)] % esize == 0, "unsupported stride on W axis");
  check(strides_bytes[static_cast<size_t>(ndim - 2)] %
                strides_bytes[static_cast<size_t>(ndim - 1)] ==
            0,
        "H axis stride must be a multiple of W stride");

  const int istride   = static_cast<int>(strides_bytes[static_cast<size_t>(ndim - 1)] / esize);
  const int inembed_w = static_cast<int>(strides_bytes[static_cast<size_t>(ndim - 2)] /
                                         strides_bytes[static_cast<size_t>(ndim - 1)]);

  const size_t h = idesc.shape[static_cast<size_t>(ndim - 2)];
  const size_t w = idesc.shape[static_cast<size_t>(ndim - 1)];

  long long idist         = static_cast<long long>(h * w);
  size_t    inner_batches = 1;
  int       first_outer   = ndim - 2;

  if (first_outer > 0) {
    int k = first_outer - 1;
    idist = static_cast<long long>(strides_bytes[static_cast<size_t>(k)] / esize);
    inner_batches *= idesc.shape[static_cast<size_t>(k)];
    first_outer = k;
    for (int i = k - 1; i >= 0; --i) {
      size_t expected =
          strides_bytes[static_cast<size_t>(i + 1)] * idesc.shape[static_cast<size_t>(i + 1)];
      if (strides_bytes[static_cast<size_t>(i)] == expected) {
        inner_batches *= idesc.shape[static_cast<size_t>(i)];
        first_outer = i;
      } else {
        break;
      }
    }
  }

  std::vector<size_t> offsets;
  if (first_outer > 0) {
    generate_offsets_recursive(idesc.shape, strides_bytes, first_outer - 1, 0, offsets);
  } else {
    offsets.push_back(0);
  }

  return {offsets, inner_batches, idist, istride, inembed_w};
}
struct ForwardPlanInfo {
  curaii::CufftHandle plan;
  std::vector<size_t> input_offsets;
  size_t              output_stride_bytes = 0;
};

ForwardPlanInfo make_cfft2_forward_plan(const holoflow::core::TDesc &idesc, cudaStream_t stream) {
  const int    ndim  = static_cast<int>(idesc.shape.size());
  const size_t h     = idesc.shape[static_cast<size_t>(ndim - 2)];
  const size_t w     = idesc.shape[static_cast<size_t>(ndim - 1)];
  const size_t n_fft = h * w;

  auto layout = get_tensor_layout(idesc);

  const auto ll_max = std::numeric_limits<long long>::max();
  check(n_fft <= static_cast<size_t>(ll_max), "FFT size exceeds cuFFT limits");
  check(layout.inner_batches > 0, "invalid batch size");
  check(layout.inner_batches <= static_cast<size_t>(ll_max), "batch size exceeds cuFFT limits");

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));
  size_t work_size = 0;

  // The planner acts on the densely packed cuFloatComplex buffers
  long long n[2]       = {static_cast<long long>(h), static_cast<long long>(w)};
  long long inembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};
  long long onembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, n, inembed, 1, static_cast<long long>(n_fft),
                                  CUDA_C_32F, onembed, 1, static_cast<long long>(n_fft), CUDA_C_32F,
                                  static_cast<long long>(layout.inner_batches), &work_size,
                                  CUDA_C_32F));

  return ForwardPlanInfo{
      .plan                = std::move(plan),
      .input_offsets       = std::move(layout.offsets),
      .output_stride_bytes = layout.inner_batches * n_fft * sizeof(cuFloatComplex),
  };
}

curaii::CufftHandle make_cfft2_inverse_plan(const holoflow::core::TDesc &desc,
                                            cudaStream_t                 stream) {
  const int    ndim       = static_cast<int>(desc.shape.size());
  const size_t h          = desc.shape[static_cast<size_t>(ndim - 2)];
  const size_t w          = desc.shape[static_cast<size_t>(ndim - 1)];
  const size_t n_fft      = h * w;
  const size_t batch_size = product(desc.shape, 0, desc.shape.size() - 2);

  const auto ll_max = std::numeric_limits<long long>::max();
  check(n_fft <= static_cast<size_t>(ll_max), "FFT size exceeds cuFFT limits");
  check(batch_size > 0, "invalid batch size");
  check(batch_size <= static_cast<size_t>(ll_max), "batch size exceeds cuFFT limits");

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));
  size_t work_size = 0;

  long long n[2]       = {static_cast<long long>(h), static_cast<long long>(w)};
  long long inembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};
  long long onembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, n, inembed, 1, static_cast<long long>(n_fft),
                                  CUDA_C_32F, onembed, 1, static_cast<long long>(n_fft), CUDA_C_32F,
                                  static_cast<long long>(batch_size), &work_size, CUDA_C_32F));

  return plan;
}

std::vector<size_t> make_reference_batch_map(const holoflow::core::TDesc &moving,
                                             const holoflow::core::TDesc &reference) {
  const size_t moving_leading_rank = moving.shape.size() - 2;
  const size_t ref_leading_rank    = reference.shape.size() - 2;
  const size_t moving_batches      = product(moving.shape, 0, moving_leading_rank);

  std::vector<size_t> moving_leading_shape(moving.shape.begin(), moving.shape.end() - 2);
  std::vector<size_t> ref_leading_shape(reference.shape.begin(), reference.shape.end() - 2);
  std::vector<size_t> ref_leading_strides(ref_leading_rank, 1);

  for (size_t i = ref_leading_rank; i-- > 1;) {
    ref_leading_strides[i - 1] = ref_leading_strides[i] * ref_leading_shape[i];
  }

  std::vector<size_t> coords(moving_leading_rank, 0);
  std::vector<size_t> batch_map(moving_batches, 0);

  for (size_t moving_batch = 0; moving_batch < moving_batches; ++moving_batch) {
    size_t       ref_batch   = 0;
    const size_t rank_offset = moving_leading_rank - ref_leading_rank;

    for (size_t axis = 0; axis < moving_leading_rank; ++axis) {
      if (axis < rank_offset)
        continue;

      const size_t ref_axis  = axis - rank_offset;
      const size_t ref_dim   = ref_leading_shape[ref_axis];
      const size_t ref_coord = ref_dim == 1 ? 0 : coords[axis];
      ref_batch += ref_coord * ref_leading_strides[ref_axis];
    }

    batch_map[moving_batch] = ref_batch;

    for (size_t axis = moving_leading_rank; axis-- > 0;) {
      coords[axis] += 1;
      if (coords[axis] < moving_leading_shape[axis])
        break;
      coords[axis] = 0;
    }
  }

  return batch_map;
}

__device__ float compute_ellipse_sq_dist(int x, int y, int W, int H,
                                         CrossCorrelation2Settings::Ellipse roi) {
  if (W <= 0 || H <= 0 || roi.rx <= 0.f || roi.ry <= 0.f) {
    return 2.0f; // Safely outside
  }

  float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
  float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);

  float dx = xn - roi.cx;
  float dy = yn - roi.cy;

  float th = roi.angle * (CUDART_PI_F / 180.0f);
  float c  = cosf(th);
  float s  = sinf(th);
  float xr = c * dx + s * dy;
  float yr = -s * dx + c * dy;

  return (xr * xr) / (roi.rx * roi.rx) + (yr * yr) / (roi.ry * roi.ry);
}

__global__ void compute_means_kernel(const float *__restrict__ in_base,
                                     float *__restrict__ means_out, size_t inner_batches,
                                     size_t idist, size_t istride, size_t inembed_w, size_t h,
                                     size_t w, size_t global_batch_offset,
                                     CrossCorrelation2Settings::Ellipse roi) {
  size_t batch_idx = blockIdx.x;
  if (batch_idx >= inner_batches)
    return;

  const float *img_in       = in_base + batch_idx * idist;
  float        sum          = 0.0f;
  int          count        = 0;
  size_t       total_pixels = h * w;

  for (size_t i = threadIdx.x; i < total_pixels; i += blockDim.x) {
    size_t y = i / w;
    size_t x = i % w;

    // Only accumulate pixels inside the ellipse
    if (compute_ellipse_sq_dist(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w),
                                static_cast<int>(h), roi) <= 1.0f) {
      sum += img_in[y * inembed_w * istride + x * istride];
      count++;
    }
  }

  __shared__ float sdata_sum[256];
  __shared__ int   sdata_count[256];

  sdata_sum[threadIdx.x]   = sum;
  sdata_count[threadIdx.x] = count;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata_sum[threadIdx.x] += sdata_sum[threadIdx.x + s];
      sdata_count[threadIdx.x] += sdata_count[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (sdata_count[0] > 0) {
      means_out[global_batch_offset + batch_idx] =
          sdata_sum[0] / static_cast<float>(sdata_count[0]);
    } else {
      means_out[global_batch_offset + batch_idx] = 0.0f;
    }
  }
}

// Write to cuFloatComplex setting .x to the real pixel value and .y to 0
__global__ void
subtract_and_pack_kernel(const float *__restrict__ in_base, const float *__restrict__ means,
                         cuFloatComplex *__restrict__ out_contiguous, size_t inner_batches,
                         size_t idist, size_t istride, size_t inembed_w, size_t h, size_t w,
                         size_t global_batch_offset, CrossCorrelation2Settings::Ellipse roi) {
  size_t idx         = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_elems = inner_batches * h * w;
  if (idx >= total_elems)
    return;

  size_t batch_idx = idx / (h * w);
  size_t pixel_idx = idx % (h * w);
  size_t y         = pixel_idx / w;
  size_t x         = pixel_idx % w;

  const float *img_in  = in_base + batch_idx * idist;
  float        val     = img_in[y * inembed_w * istride + x * istride];
  float        mean    = means[global_batch_offset + batch_idx];
  size_t       out_idx = (global_batch_offset + batch_idx) * (h * w) + pixel_idx;

  float d = compute_ellipse_sq_dist(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w),
                                    static_cast<int>(h), roi);

  if (d <= 1.0f) {
    float w_val               = 0.5f * (1.0f + cosf(CUDART_PI_F * sqrtf(d)));
    out_contiguous[out_idx].x = (val - mean) * w_val;
    out_contiguous[out_idx].y = 0.0f;
  } else {
    out_contiguous[out_idx].x = 0.0f;
    out_contiguous[out_idx].y = 0.0f;
  }
}

__global__ void phase_correlate_kernel(cuFloatComplex *__restrict__ moving_freq,
                                       const cuFloatComplex *__restrict__ reference_freq,
                                       const size_t *__restrict__ reference_batch_map,
                                       size_t total_freq_elems, size_t freq_elems_per_batch) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_freq_elems)
    return;

  const size_t batch_idx = idx / freq_elems_per_batch;
  const size_t freq_idx  = idx % freq_elems_per_batch;
  const size_t ref_batch = reference_batch_map[batch_idx];

  const cuFloatComplex moving_val    = moving_freq[idx];
  const cuFloatComplex reference_val = reference_freq[ref_batch * freq_elems_per_batch + freq_idx];
  const cuFloatComplex product       = cuCmulf(moving_val, cuConjf(reference_val));
  const float          mag           = sqrtf(product.x * product.x + product.y * product.y);

  if (mag > 1e-12f) {
    moving_freq[idx].x = product.x / mag;
    moving_freq[idx].y = product.y / mag;
  } else {
    moving_freq[idx].x = 0.0f;
    moving_freq[idx].y = 0.0f;
  }
}

// Extract real component from cuFloatComplex buffer to float buffer and scale
__global__ void extract_real_and_scale_kernel(const cuFloatComplex *__restrict__ data_in,
                                              float *__restrict__ data_out, size_t n, float scale) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data_out[idx] = data_in[idx].x * scale;
  }
}

class CrossCorrelation2 : public holoflow::core::ISyncTask {
public:
  CrossCorrelation2(CrossCorrelation2Settings settings, holoflow::core::TDesc moving_desc,
                    holoflow::core::TDesc reference_desc, holoflow::core::TDesc moving_freq_desc,
                    curaii::CufftHandle &&moving_fwd_plan,
                    curaii::CufftHandle &&reference_fwd_plan,
                    curaii::CufftHandle &&inverse_plan, TensorLayout moving_layout,
                    TensorLayout reference_layout, size_t h, size_t w, size_t freq_elems_per_batch,
                    size_t total_moving_freq_elems, size_t total_out_elems,
                    DevPtr<cuFloatComplex> d_moving_spatial,
                    DevPtr<cuFloatComplex> d_reference_spatial, DevPtr<float> d_moving_means,
                    DevPtr<float> d_reference_means, DevPtr<cuFloatComplex> d_moving_freq,
                    DevPtr<cuFloatComplex> d_reference_freq, DevPtr<size_t> d_reference_batch_map,
                    cudaStream_t stream)
      : settings_(std::move(settings)), moving_desc_(std::move(moving_desc)),
        reference_desc_(std::move(reference_desc)), moving_freq_desc_(std::move(moving_freq_desc)),
        moving_fwd_plan_(std::move(moving_fwd_plan)),
        reference_fwd_plan_(std::move(reference_fwd_plan)), inverse_plan_(std::move(inverse_plan)),
        moving_layout_(std::move(moving_layout)), reference_layout_(std::move(reference_layout)),
        h_(h), w_(w), freq_elems_per_batch_(freq_elems_per_batch),
        total_moving_freq_elems_(total_moving_freq_elems), total_out_elems_(total_out_elems),
        d_moving_spatial_(std::move(d_moving_spatial)),
        d_reference_spatial_(std::move(d_reference_spatial)),
        d_moving_means_(std::move(d_moving_means)),
        d_reference_means_(std::move(d_reference_means)),
        d_moving_freq_(std::move(d_moving_freq)),
        d_reference_freq_(std::move(d_reference_freq)),
        d_reference_batch_map_(std::move(d_reference_batch_map)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const CrossCorrelation2Settings &settings() const { return settings_; }
  const holoflow::core::TDesc     &moving_desc() const { return moving_desc_; }
  const holoflow::core::TDesc     &reference_desc() const { return reference_desc_; }
  void                             update_stream(cudaStream_t stream);

private:
  CrossCorrelation2Settings settings_;
  holoflow::core::TDesc     moving_desc_;
  holoflow::core::TDesc     reference_desc_;
  holoflow::core::TDesc     moving_freq_desc_;
  curaii::CufftHandle       moving_fwd_plan_;
  curaii::CufftHandle       reference_fwd_plan_;
  curaii::CufftHandle       inverse_plan_;
  TensorLayout              moving_layout_;
  TensorLayout              reference_layout_;
  size_t                    h_;
  size_t                    w_;
  size_t                    freq_elems_per_batch_;
  size_t                    total_moving_freq_elems_;
  size_t                    total_out_elems_;
  DevPtr<cuFloatComplex>    d_moving_spatial_;
  DevPtr<cuFloatComplex>    d_reference_spatial_;
  DevPtr<float>             d_moving_means_;
  DevPtr<float>             d_reference_means_;
  DevPtr<cuFloatComplex>    d_moving_freq_;
  DevPtr<cuFloatComplex>    d_reference_freq_;
  DevPtr<size_t>            d_reference_batch_map_;
  cudaStream_t              stream_;
};

} // namespace

void CrossCorrelation2::update_stream(cudaStream_t stream) {
  if (stream_ != stream) {
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(moving_fwd_plan_.get(), stream_));
    CUFFT_CHECK(cufftSetStream(reference_fwd_plan_.get(), stream_));
    CUFFT_CHECK(cufftSetStream(inverse_plan_.get(), stream_));
  }
}

holoflow::core::OpResult CrossCorrelation2::execute(holoflow::core::SyncCtx &ctx) {
  auto *moving_input_base    = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *reference_input_base = reinterpret_cast<uint8_t *>(ctx.inputs[1].data());

  size_t global_batch = 0;
  for (size_t off : moving_layout_.offsets) {
    const float *in_ptr = reinterpret_cast<const float *>(moving_input_base + off);

    unsigned int means_grid = static_cast<unsigned int>(moving_layout_.inner_batches);
    compute_means_kernel<<<means_grid, 256, 0, stream_>>>(
        in_ptr, d_moving_means_.get(), moving_layout_.inner_batches, moving_layout_.idist,
        moving_layout_.istride, moving_layout_.inembed_w, h_, w_, global_batch, settings_.roi);

    unsigned int threads = 256;
    unsigned int blocks =
        static_cast<unsigned int>((moving_layout_.inner_batches * h_ * w_ + threads - 1) / threads);
    subtract_and_pack_kernel<<<blocks, threads, 0, stream_>>>(
        in_ptr, d_moving_means_.get(), d_moving_spatial_.get(), moving_layout_.inner_batches,
        moving_layout_.idist, moving_layout_.istride, moving_layout_.inembed_w, h_, w_,
        global_batch, settings_.roi);

    global_batch += moving_layout_.inner_batches;
  }

  global_batch = 0;
  for (size_t off : reference_layout_.offsets) {
    const float *in_ptr = reinterpret_cast<const float *>(reference_input_base + off);

    unsigned int means_grid = static_cast<unsigned int>(reference_layout_.inner_batches);
    compute_means_kernel<<<means_grid, 256, 0, stream_>>>(
        in_ptr, d_reference_means_.get(), reference_layout_.inner_batches, reference_layout_.idist,
        reference_layout_.istride, reference_layout_.inembed_w, h_, w_, global_batch,
        settings_.roi);

    unsigned int threads = 256;
    unsigned int blocks  = static_cast<unsigned int>(
        (reference_layout_.inner_batches * h_ * w_ + threads - 1) / threads);
    subtract_and_pack_kernel<<<blocks, threads, 0, stream_>>>(
        in_ptr, d_reference_means_.get(), d_reference_spatial_.get(),
        reference_layout_.inner_batches, reference_layout_.idist, reference_layout_.istride,
        reference_layout_.inembed_w, h_, w_, global_batch, settings_.roi);

    global_batch += reference_layout_.inner_batches;
  }

  CUFFT_CHECK(cufftXtExec(moving_fwd_plan_.get(), d_moving_spatial_.get(), d_moving_freq_.get(),
                          CUFFT_FORWARD));
  CUFFT_CHECK(cufftXtExec(reference_fwd_plan_.get(), d_reference_spatial_.get(),
                          d_reference_freq_.get(), CUFFT_FORWARD));

  if (total_moving_freq_elems_ > 0) {
    constexpr unsigned int block = 256;
    const unsigned int     grid =
        static_cast<unsigned int>((total_moving_freq_elems_ + block - 1) / block);
    phase_correlate_kernel<<<grid, block, 0, stream_>>>(
        d_moving_freq_.get(), d_reference_freq_.get(), d_reference_batch_map_.get(),
        total_moving_freq_elems_, freq_elems_per_batch_);
  }

  // Inverse FFT writes directly over the complex moving spatial buffer (saving memory)
  CUFFT_CHECK(cufftXtExec(inverse_plan_.get(), d_moving_freq_.get(), d_moving_spatial_.get(),
                          CUFFT_INVERSE));

  auto        *out_ptr = reinterpret_cast<float *>(ctx.outputs[0].data());
  const size_t n_fft   = moving_desc_.shape[moving_desc_.shape.size() - 2] *
                       moving_desc_.shape[moving_desc_.shape.size() - 1];
  const float scale = inverse_scale(settings_.norm, n_fft);

  if (total_out_elems_ > 0) {
    constexpr unsigned int block = 256;
    const unsigned int     grid = static_cast<unsigned int>((total_out_elems_ + block - 1) / block);
    extract_real_and_scale_kernel<<<grid, block, 0, stream_>>>(d_moving_spatial_.get(), out_ptr,
                                                               total_out_elems_, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
CrossCorrelation2Factory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<CrossCorrelation2Settings>();
  check(input_descs.size() == 2, "expected exactly 2 input tensors");

  const auto &moving    = input_descs[0];
  const auto &reference = input_descs[1];

  check(moving.mem_loc == holoflow::core::MemLoc::Device, "moving input must be in device memory");
  check(reference.mem_loc == holoflow::core::MemLoc::Device,
        "reference input must be in device memory");
  check(moving.dtype == holoflow::core::DType::F32, "moving input must be F32");
  check(reference.dtype == holoflow::core::DType::F32, "reference input must be F32");
  check(moving.rank() >= 2, "moving input rank must be >= 2");
  check(reference.rank() >= 2, "reference input rank must be >= 2");
  check(reference.rank() <= moving.rank(), "reference input rank cannot exceed moving input rank");

  const int moving_ndim = static_cast<int>(moving.rank());
  const int ref_ndim    = static_cast<int>(reference.rank());
  (void)resolve_axes(settings.axes, moving_ndim);
  (void)resolve_axes(settings.axes, ref_ndim);

  const size_t moving_h = moving.shape[moving.shape.size() - 2];
  const size_t moving_w = moving.shape[moving.shape.size() - 1];
  const size_t ref_h    = reference.shape[reference.shape.size() - 2];
  const size_t ref_w    = reference.shape[reference.shape.size() - 1];

  check(moving_h == ref_h && moving_w == ref_w,
        "moving and reference FFT axes must have identical shapes");

  const size_t moving_leading_rank = moving.rank() - 2;
  const size_t ref_leading_rank    = reference.rank() - 2;
  const size_t rank_offset         = moving_leading_rank - ref_leading_rank;

  for (size_t axis = 0; axis < ref_leading_rank; ++axis) {
    const size_t moving_dim = moving.shape[rank_offset + axis];
    const size_t ref_dim    = reference.shape[axis];
    check(ref_dim == 1 || ref_dim == moving_dim,
          "reference leading dimensions must be 1 or match moving");
  }

  holoflow::core::TDesc odesc(moving.shape, holoflow::core::DType::F32,
                              holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {moving, reference},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CrossCorrelation2Factory::create(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings,
                                 const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = infer(input_descs, jsettings);
  const auto settings  = jsettings.get<CrossCorrelation2Settings>();

  const auto &moving    = input_descs[0];
  const auto &reference = input_descs[1];
  const auto &out_desc  = infer_res.output_descs[0];

  const size_t h = moving.shape[moving.shape.size() - 2];
  const size_t w = moving.shape[moving.shape.size() - 1];

  auto moving_layout    = get_tensor_layout(moving);
  auto reference_layout = get_tensor_layout(reference);

  size_t moving_batches = product(moving.shape, 0, moving.shape.size() - 2);
  size_t ref_batches    = product(reference.shape, 0, reference.shape.size() - 2);

  // Full spatial dimensions for C2C Transforms
  holoflow::core::TDesc moving_freq_desc(moving.shape, holoflow::core::DType::CF32,
                                         holoflow::core::MemLoc::Device);
  holoflow::core::TDesc reference_freq_desc(reference.shape, holoflow::core::DType::CF32,
                                            holoflow::core::MemLoc::Device);

  auto moving_plan_info    = make_cfft2_forward_plan(moving, ctx.stream);
  auto reference_plan_info = make_cfft2_forward_plan(reference, ctx.stream);
  auto inverse_plan        = make_cfft2_inverse_plan(moving, ctx.stream);

  // Allocate complex spatial buffers to support the padded C2C Pipeline
  auto d_moving_spatial =
      curaii::make_unique_device_ptr<cuFloatComplex>(moving_batches * h * w, ctx.stream);
  auto d_reference_spatial =
      curaii::make_unique_device_ptr<cuFloatComplex>(ref_batches * h * w, ctx.stream);

  auto d_moving_means    = curaii::make_unique_device_ptr<float>(moving_batches, ctx.stream);
  auto d_reference_means = curaii::make_unique_device_ptr<float>(ref_batches, ctx.stream);

  auto reference_batch_map_h = make_reference_batch_map(moving, reference);
  auto d_reference_batch_map =
      curaii::make_unique_device_ptr<size_t>(reference_batch_map_h.size(), ctx.stream);
  CUDA_CHECK(cudaMemcpyAsync(d_reference_batch_map.get(), reference_batch_map_h.data(),
                             reference_batch_map_h.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                             ctx.stream));

  auto d_moving_freq =
      curaii::make_unique_device_ptr<cuFloatComplex>(moving_freq_desc.num_elements(), ctx.stream);
  auto d_reference_freq = curaii::make_unique_device_ptr<cuFloatComplex>(
      reference_freq_desc.num_elements(), ctx.stream);

  const size_t freq_elems_per_batch = h * w;

  return std::unique_ptr<holoflow::core::ISyncTask>(new CrossCorrelation2(
      settings, moving, reference, moving_freq_desc, std::move(moving_plan_info.plan),
      std::move(reference_plan_info.plan), std::move(inverse_plan), std::move(moving_layout),
      std::move(reference_layout), h, w, freq_elems_per_batch, moving_freq_desc.num_elements(),
      out_desc.num_elements(), std::move(d_moving_spatial), std::move(d_reference_spatial),
      std::move(d_moving_means), std::move(d_reference_means), std::move(d_moving_freq),
      std::move(d_reference_freq), std::move(d_reference_batch_map), ctx.stream));
}

std::unique_ptr<holoflow::core::ISyncTask>
CrossCorrelation2Factory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                 std::span<const holoflow::core::TDesc>     input_descs,
                                 const nlohmann::json                      &jsettings,
                                 const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old_xcorr = dynamic_cast<CrossCorrelation2 *>(old_task.get());
  if (old_xcorr != nullptr && input_descs.size() == 2) {
    const auto  new_settings  = jsettings.get<CrossCorrelation2Settings>();
    const auto &old_moving    = old_xcorr->moving_desc();
    const auto &old_reference = old_xcorr->reference_desc();
    const auto &new_moving    = input_descs[0];
    const auto &new_reference = input_descs[1];

    const bool can_reuse = (new_settings == old_xcorr->settings()) &&
                           same_desc(new_moving, old_moving) &&
                           same_desc(new_reference, old_reference);

    if (can_reuse) {
      old_xcorr->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
