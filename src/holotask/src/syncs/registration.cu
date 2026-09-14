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

#include "holotask/syncs/registration.hh"

#include <cub/cub.cuh>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "logger.hh"
#include "syncs/phase_correlation.cuh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

using curaii::make_unique_device_ptr;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const RegistrationSettings &s) {
  j = nlohmann::json{{"radius", s.radius}};
}

void from_json(const nlohmann::json &j, RegistrationSettings &s) {
  j.at("radius").get_to(s.radius);
}

namespace {

void check(bool condition, const std::string &message) {
  if (!condition) {
    logger()->error("[RegistrationFactory::infer] error: {}", message);
    throw std::invalid_argument("RegistrationFactory inference error: " + message);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

struct RegistrationEstimate {
  float shift_x;
  float shift_y;
  float rotation_degrees;
  float rotation_score;
  float psr;
  int   accepted;
};

struct RegistrationTelemetry {
  HostPtr<RegistrationEstimate> estimate;
  cudaEvent_t                   ready = nullptr;
  size_t                        frame_index{};
  bool                          pending = false;
};

inline constexpr int   kRotationCandidateCount = 41;
inline constexpr float kRotationStepDegrees    = 0.25f;
// Real recordings have a much flatter Fourier-magnitude correlation than the
// synthetic registration fixtures. Translation PSR and motion bounds remain
// the primary rejection criteria; this threshold only rejects unusable angular
// estimates.
inline constexpr float kMinimumRotationScore = 0.03f;
// Rotation is more sensitive to weak spectral structure than translation. Keep
// accepting frames whose translation is reliable, but only apply a rotation
// when the angular correlation itself is strong enough to justify interpolation.
inline constexpr float kMinimumRotationApplicationScore = 0.10f;

__global__ void spectrum_magnitude_kernel(float *magnitude, const cuFloatComplex *frequency,
                                          int frequency_width, int width, int height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  const int  x         = idx % width;
  const int  y         = idx / width;
  const int  kx        = x - width / 2;
  const int  ky        = y - height / 2;
  const int  source_kx = kx >= 0 ? kx : -kx;
  const int  source_ky = kx >= 0 ? ky : -ky;
  const int  source_y  = source_ky >= 0 ? source_ky : source_ky + height;
  const auto value     = frequency[source_y * frequency_width + source_kx];
  magnitude[idx]       = log1pf(sqrtf(value.x * value.x + value.y * value.y));
}

__global__ void rotation_scores_kernel(float *scores, const float *current, const float *reference,
                                       int width, int height) {
  __shared__ float sum_reference[256];
  __shared__ float sum_current[256];
  __shared__ float sum_reference_sq[256];
  __shared__ float sum_current_sq[256];
  __shared__ float sum_product[256];
  __shared__ int   sample_count[256];

  const int   candidate = blockIdx.x;
  const float degrees =
      (static_cast<float>(candidate) - 0.5f * (kRotationCandidateCount - 1)) * kRotationStepDegrees;
  const float radians = degrees * (CUDART_PI_F / 180.0f);
  const float cosine  = cosf(radians);
  const float sine    = sinf(radians);
  const float cx      = 0.5f * static_cast<float>(width - 1);
  const float cy      = 0.5f * static_cast<float>(height - 1);
  const float min_dim = static_cast<float>(min(width, height));
  const float min_r2  = 0.01f * min_dim * min_dim;
  const float max_r2  = 0.20f * min_dim * min_dim;

  float         sr            = 0.0f;
  float         sc            = 0.0f;
  float         srr           = 0.0f;
  float         scc           = 0.0f;
  float         src           = 0.0f;
  int           count         = 0;
  constexpr int sample_stride = 4;
  const int     sample_width  = (width + sample_stride - 1) / sample_stride;
  const int     sample_height = (height + sample_stride - 1) / sample_stride;
  for (int sample = threadIdx.x; sample < sample_width * sample_height; sample += blockDim.x) {
    const int   x   = (sample % sample_width) * sample_stride;
    const int   y   = (sample / sample_width) * sample_stride;
    const int   idx = y * width + x;
    const float dx  = static_cast<float>(x) - cx;
    const float dy  = static_cast<float>(y) - cy;
    const float r2  = dx * dx + dy * dy;
    if (r2 < min_r2 || r2 > max_r2)
      continue;

    const float sx = cx + cosine * dx - sine * dy;
    const float sy = cy + sine * dx + cosine * dy;
    const int   x0 = static_cast<int>(floorf(sx));
    const int   y0 = static_cast<int>(floorf(sy));
    if (x0 < 0 || x0 + 1 >= width || y0 < 0 || y0 + 1 >= height)
      continue;
    const float fx  = sx - static_cast<float>(x0);
    const float fy  = sy - static_cast<float>(y0);
    const float c00 = current[y0 * width + x0];
    const float c10 = current[y0 * width + x0 + 1];
    const float c01 = current[(y0 + 1) * width + x0];
    const float c11 = current[(y0 + 1) * width + x0 + 1];
    const float cv  = (1.0f - fx) * (1.0f - fy) * c00 + fx * (1.0f - fy) * c10 +
                      (1.0f - fx) * fy * c01 + fx * fy * c11;
    const float rv  = reference[idx];
    sr += rv;
    sc += cv;
    srr += rv * rv;
    scc += cv * cv;
    src += rv * cv;
    ++count;
  }

  sum_reference[threadIdx.x]    = sr;
  sum_current[threadIdx.x]      = sc;
  sum_reference_sq[threadIdx.x] = srr;
  sum_current_sq[threadIdx.x]   = scc;
  sum_product[threadIdx.x]      = src;
  sample_count[threadIdx.x]     = count;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_reference[threadIdx.x] += sum_reference[threadIdx.x + stride];
      sum_current[threadIdx.x] += sum_current[threadIdx.x + stride];
      sum_reference_sq[threadIdx.x] += sum_reference_sq[threadIdx.x + stride];
      sum_current_sq[threadIdx.x] += sum_current_sq[threadIdx.x + stride];
      sum_product[threadIdx.x] += sum_product[threadIdx.x + stride];
      sample_count[threadIdx.x] += sample_count[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float n         = static_cast<float>(sample_count[0]);
    const float numerator = n * sum_product[0] - sum_reference[0] * sum_current[0];
    const float denominator =
        sqrtf(fmaxf(0.0f, n * sum_reference_sq[0] - sum_reference[0] * sum_reference[0]) *
              fmaxf(0.0f, n * sum_current_sq[0] - sum_current[0] * sum_current[0]));
    scores[candidate] = denominator > 1e-12f ? numerator / denominator : -1.0f;
  }
}

__global__ void finalize_rotation_kernel(RegistrationEstimate *estimate, const float *scores) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  int best = 0;
  for (int i = 1; i < kRotationCandidateCount; ++i) {
    if (scores[i] > scores[best])
      best = i;
  }
  constexpr int zero_rotation = (kRotationCandidateCount - 1) / 2;
  // FFT magnitudes can have a broad angular maximum. Do not introduce a small,
  // interpolating rotation unless it improves the zero-rotation hypothesis by
  // a meaningful margin.
  if (!isfinite(scores[best]) || scores[best] < kMinimumRotationApplicationScore ||
      scores[best] - scores[zero_rotation] < 0.01f) {
    estimate->rotation_degrees = 0.0f;
    estimate->rotation_score   = scores[zero_rotation];
    return;
  }
  float offset = 0.0f;
  if (best > 0 && best + 1 < kRotationCandidateCount) {
    const float lower       = scores[best - 1];
    const float center      = scores[best];
    const float upper       = scores[best + 1];
    const float denominator = lower - 2.0f * center + upper;
    if (isfinite(denominator) && fabsf(denominator) > 1e-12f)
      offset = fminf(0.5f, fmaxf(-0.5f, 0.5f * (lower - upper) / denominator));
  }
  estimate->rotation_degrees =
      (static_cast<float>(best) + offset - 0.5f * (kRotationCandidateCount - 1)) *
      kRotationStepDegrees;
  estimate->rotation_score = scores[best];
}

__global__ void rotate_image_kernel(float *output, const float *input,
                                    const RegistrationEstimate *estimate, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const float radians = estimate->rotation_degrees * (CUDART_PI_F / 180.0f);
  const float cosine  = cosf(radians);
  const float sine    = sinf(radians);
  const float cx      = 0.5f * static_cast<float>(width - 1);
  const float cy      = 0.5f * static_cast<float>(height - 1);
  const float dx      = static_cast<float>(x) - cx;
  const float dy      = static_cast<float>(y) - cy;
  const float sx      = cx + cosine * dx - sine * dy;
  const float sy      = cy + sine * dx + cosine * dy;
  const int   x0      = static_cast<int>(floorf(sx));
  const int   y0      = static_cast<int>(floorf(sy));
  if (x0 < 0 || x0 + 1 >= width || y0 < 0 || y0 + 1 >= height) {
    output[y * width + x] = 0.0f;
    return;
  }
  const float fx        = sx - static_cast<float>(x0);
  const float fy        = sy - static_cast<float>(y0);
  output[y * width + x] = (1.0f - fx) * (1.0f - fy) * input[y0 * width + x0] +
                          fx * (1.0f - fy) * input[y0 * width + x0 + 1] +
                          (1.0f - fx) * fy * input[(y0 + 1) * width + x0] +
                          fx * fy * input[(y0 + 1) * width + x0 + 1];
}

__global__ void phase_cross_power_kernel(cuFloatComplex *current, const cuFloatComplex *reference,
                                         int frequency_width, int width, int height, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  const int   kx                = idx % frequency_width;
  const int   row               = idx / frequency_width;
  const int   ky                = row <= height / 2 ? row : row - height;
  const float nx                = 2.0f * static_cast<float>(kx) / static_cast<float>(width);
  const float ny                = 2.0f * static_cast<float>(ky) / static_cast<float>(height);
  const float normalized_radius = sqrtf(nx * nx + ny * ny);
  const float cycles            = sqrtf(static_cast<float>(kx * kx + ky * ky));
  if (cycles < 2.0f || normalized_radius > 0.9f) {
    current[idx] = make_cuFloatComplex(0.0f, 0.0f);
    return;
  }

  current[idx] = detail::normalized_cross_power(current[idx], reference[idx]);
}

__global__ void psr_stats_kernel(const float *correlation, const int64_t *peak_index, float *stats,
                                 int width, int height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height)
    return;

  const int peak = static_cast<int>(*peak_index);
  const int px   = peak % width;
  const int py   = peak / width;
  const int x    = idx % width;
  const int y    = idx / width;
  const int dx   = min(abs(x - px), width - abs(x - px));
  const int dy   = min(abs(y - py), height - abs(y - py));
  if (dx <= 5 && dy <= 5)
    return;

  const float value = correlation[idx];
  atomicAdd(&stats[0], value);
  atomicAdd(&stats[1], value * value);
}

__global__ void finalize_estimate_kernel(RegistrationEstimate *estimate, const float *correlation,
                                         const float *peak_value, const int64_t *peak_index,
                                         const float *stats, int width, int height,
                                         float max_translation) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  const int   peak    = static_cast<int>(*peak_index);
  const int   px      = peak % width;
  const int   py      = peak / width;
  const int   xm      = (px + width - 1) % width;
  const int   xp      = (px + 1) % width;
  const int   ym      = (py + height - 1) % height;
  const int   yp      = (py + 1) % height;
  const float center  = correlation[peak];
  float       shift_x = detail::circular_signed_coordinate(px, width);
  float       shift_y = detail::circular_signed_coordinate(py, height);
  shift_x += detail::parabolic_peak_offset(correlation[py * width + xm], center,
                                           correlation[py * width + xp]);
  shift_y += detail::parabolic_peak_offset(correlation[ym * width + px], center,
                                           correlation[yp * width + px]);

  const int   sidelobe_count = width * height - 121;
  const float mean = sidelobe_count > 0 ? stats[0] / static_cast<float>(sidelobe_count) : 0.0f;
  const float variance =
      sidelobe_count > 0 ? fmaxf(0.0f, stats[1] / static_cast<float>(sidelobe_count) - mean * mean)
                         : 0.0f;
  const float stddev   = sqrtf(variance);
  const float psr      = stddev > 1e-12f ? (*peak_value - mean) / stddev : 0.0f;
  const bool  accepted = isfinite(shift_x) && isfinite(shift_y) && isfinite(psr) && psr >= 8.0f &&
                         isfinite(estimate->rotation_degrees) &&
                         estimate->rotation_score >= kMinimumRotationScore &&
                         fabsf(shift_x) <= max_translation && fabsf(shift_y) <= max_translation;
  estimate->shift_x    = shift_x;
  estimate->shift_y    = shift_y;
  estimate->psr        = psr;
  estimate->accepted   = accepted ? 1 : 0;
}

__global__ void f32_shift_subpixel_kernel(float *odata, uint8_t *valid, const float *idata,
                                          const RegistrationEstimate *estimate, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  const float shift_x = estimate->accepted ? estimate->shift_x : 0.0f;
  const float shift_y = estimate->accepted ? estimate->shift_y : 0.0f;
  const float angle =
      estimate->accepted ? estimate->rotation_degrees * (CUDART_PI_F / 180.0f) : 0.0f;
  const float cosine = cosf(angle);
  const float sine   = sinf(angle);
  const float cx     = 0.5f * static_cast<float>(w - 1);
  const float cy     = 0.5f * static_cast<float>(h - 1);
  const float px     = static_cast<float>(x) - cx;
  const float py     = static_cast<float>(y) - cy;
  float       src_x  = cx + cosine * px - sine * py + shift_x;
  float       src_y  = cy + sine * px + cosine * py + shift_y;

  int x0 = static_cast<int>(floor(src_x));
  int y0 = static_cast<int>(floor(src_y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = src_x - static_cast<float>(x0);
  float dy = src_y - static_cast<float>(y0);

  int dst_idx = y * w + x;

  auto wrap_coord = [](int coord, int size) {
    coord = coord % size;
    if (coord < 0)
      coord += size;
    return coord;
  };

  int x0_wrap = wrap_coord(x0, w);
  int x1_wrap = wrap_coord(x1, w);
  int y0_wrap = wrap_coord(y0, h);
  int y1_wrap = wrap_coord(y1, h);

  float i00 = idata[y0_wrap * w + x0_wrap];
  float i10 = idata[y0_wrap * w + x1_wrap];
  float i01 = idata[y1_wrap * w + x0_wrap];
  float i11 = idata[y1_wrap * w + x1_wrap];

  odata[dst_idx] = (1.0f - dx) * (1.0f - dy) * i00 + dx * (1.0f - dy) * i10 +
                   (1.0f - dx) * dy * i01 + dx * dy * i11;
  if (x == 0 && y == 0)
    valid[0] = estimate->accepted ? uint8_t{1} : uint8_t{0};
}

__device__ bool in_ellipse(int x, int y, int width, int height, float radius) {
  const float cx     = 0.5f * (width - 1);
  const float cy     = 0.5f * (height - 1);
  const float minDim = static_cast<float>(min(width, height));
  const float r      = radius * 0.5f * minDim;
  const float r2     = r * r;
  const float sx     = minDim / static_cast<float>(width);
  const float sy     = minDim / static_cast<float>(height);
  const float dx     = static_cast<float>(x) - cx;
  const float dy     = static_cast<float>(y) - cy;
  const float dxs    = dx * sx;
  const float dys    = dy * sy;
  return (dxs * dxs + dys * dys) <= r2;
}

__global__ void roi_sum_kernel(const float *idata, const uint8_t *roi, float *sum, int count) {
  __shared__ float partial[256];
  const int        idx = blockIdx.x * blockDim.x + threadIdx.x;
  partial[threadIdx.x] = idx < count && roi[idx] ? idata[idx] : 0.0f;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(sum, partial[0]);
}

__global__ void f32_center_mask_kernel(float *odata, const float *idata, const uint8_t *roi,
                                       int width, int height, float radius, const float *sum,
                                       const int *roi_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width * height) {
    return;
  }
  if (!roi[idx]) {
    odata[idx] = 0.0f;
    return;
  }

  const int   x                = idx % width;
  const int   y                = idx / width;
  const float cx               = 0.5f * static_cast<float>(width - 1);
  const float cy               = 0.5f * static_cast<float>(height - 1);
  const float rx               = 0.5f * radius * static_cast<float>(width);
  const float ry               = 0.5f * radius * static_cast<float>(height);
  const float dx               = (static_cast<float>(x) - cx) / rx;
  const float dy               = (static_cast<float>(y) - cy) / ry;
  const float squared_distance = dx * dx + dy * dy;
  const float mean             = *roi_count > 0 ? *sum / static_cast<float>(*roi_count) : 0.0f;
  odata[idx] = detail::preprocess_phase_correlation_value(idata[idx], mean, squared_distance);
}

__global__ void ellipse_mask_kernel(uint8_t *oroi, int width, int height, float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const bool   inside = in_ellipse(x, y, width, height, radius);
  const size_t idx    = static_cast<size_t>(y) * width + static_cast<size_t>(x);
  oroi[idx]           = inside;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Registration task implementation
// -------------------------------------------------------------------------------------------------

class Registration : public holoflow::core::ISyncTask {
public:
  Registration(RegistrationSettings settings, holoflow::core::TDesc input_desc,
               holoflow::core::TDesc output_desc, cudaStream_t stream,
               DevPtr<float> d_mean_centered, bool ref_initialized, size_t freq_size,
               curaii::CufftHandle r2c_handle, curaii::CufftHandle c2r_handle, DevPtr<float> d_ref,
               DevPtr<float> d_xcorr, DevPtr<cuFloatComplex> d_freq1,
               DevPtr<cuFloatComplex> d_freq2, size_t sum_tmp_bytes, DevPtr<uint8_t> d_sum_tmp,
               DevPtr<float> d_sum, size_t amax_tmp_bytes, DevPtr<uint8_t> d_amax_tmp,
               DevPtr<float> d_max, DevPtr<int64_t> d_max_idx, size_t select_tmp_bytes,
               DevPtr<uint8_t> d_select_tmp, DevPtr<int> d_select_count,
               DevPtr<uint8_t> d_select_roi, DevPtr<float> d_selected)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)),
        output_desc_(std::move(output_desc)), stream_(stream),
        d_mean_centered_(std::move(d_mean_centered)), ref_initialized_(ref_initialized),
        freq_size_(freq_size), r2c_handle_(std::move(r2c_handle)),
        c2r_handle_(std::move(c2r_handle)), d_ref_(std::move(d_ref)), d_xcorr_(std::move(d_xcorr)),
        d_freq1_(std::move(d_freq1)), d_freq2_(std::move(d_freq2)), sum_tmp_bytes_(sum_tmp_bytes),
        d_sum_tmp_(std::move(d_sum_tmp)), d_sum_(std::move(d_sum)), amax_tmp_bytes_(amax_tmp_bytes),
        d_amax_tmp_(std::move(d_amax_tmp)), d_max_(std::move(d_max)),
        d_max_idx_(std::move(d_max_idx)), select_tmp_bytes_(select_tmp_bytes),
        d_select_tmp_(std::move(d_select_tmp)), d_select_count_(std::move(d_select_count)),
        d_select_roi_(std::move(d_select_roi)), d_selected_(std::move(d_selected)),
        d_psr_stats_(make_unique_device_ptr<float>(2)),
        d_estimate_(make_unique_device_ptr<RegistrationEstimate>(1)) {
    const auto pixels = input_desc_.shape[input_desc_.shape.size() - 2] * input_desc_.shape.back();
    d_rotated_        = make_unique_device_ptr<float>(pixels);
    d_reference_magnitude_ = make_unique_device_ptr<float>(pixels);
    d_current_magnitude_   = make_unique_device_ptr<float>(pixels);
    d_rotation_scores_     = make_unique_device_ptr<float>(kRotationCandidateCount);
    for (auto &slot : telemetry_) {
      slot.estimate = curaii::make_unique_host_ptr<RegistrationEstimate>(1);
      CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
    }
  }

  ~Registration() override {
    for (auto &slot : telemetry_)
      if (slot.ready)
        cudaEventDestroy(slot.ready);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_recording_events(ctx);
    flush_telemetry();
    if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
      return holoflow::core::OpResult::Cancelled;
    }

    if (ctx.inputs.empty() || ctx.outputs.size() < 2) {
      return holoflow::core::OpResult::NotReady;
    }

    auto &input_view  = ctx.inputs[0];
    auto &output_view = ctx.outputs[0];

    float *input_data  = reinterpret_cast<float *>(input_view.data());
    float *output_data = reinterpret_cast<float *>(output_view.data());
    auto  *valid_data  = reinterpret_cast<uint8_t *>(ctx.outputs[1].data());

    // Registration is part of the live graph, but only needs to do the expensive
    // phase-correlation work while an image recording is active.
    if (!recording_) {
      CUDA_CHECK(cudaMemcpyAsync(output_data, input_data, input_desc_.num_bytes(),
                                 cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemsetAsync(valid_data, 0, sizeof(uint8_t), stream_));
      return holoflow::core::OpResult::Ok;
    }

    const auto width  = input_desc_.shape.back();
    const auto height = input_desc_.shape[input_desc_.shape.size() - 2];
    const auto batch  = input_desc_.rank() == 3 ? input_desc_.shape[0] : 1;

    center_mean(d_mean_centered_.get(), input_data, batch, height, width);

    if (!ref_initialized_) {
      CUDA_CHECK(cudaMemcpyAsync(d_ref_.get(), d_mean_centered_.get(), input_desc_.num_bytes(),
                                 cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(output_data, input_data, input_desc_.num_bytes(),
                                 cudaMemcpyDeviceToDevice, stream_));
      CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), d_ref_.get(), d_freq2_.get()));
      build_spectrum_magnitude(d_reference_magnitude_.get(), d_freq2_.get(), width, height);
      CUDA_CHECK(cudaMemsetAsync(valid_data, 1, sizeof(uint8_t), stream_));
      ref_initialized_ = true;
      logger()->debug("[Registration] Frame {}: captured recording reference, accepted=true",
                      frame_index_++);
      return holoflow::core::OpResult::Ok;
    }

    estimate_rotation(d_mean_centered_.get(), width, height);
    rotate_for_correlation(d_rotated_.get(), d_mean_centered_.get(), width, height);
    xcorr(d_xcorr_.get(), d_rotated_.get());
    estimate_shift(width, height);
    apply_shifts(output_data, valid_data, input_data, batch, height, width);

    schedule_telemetry();
    ++frame_index_;
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) {
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(r2c_handle_.get(), stream_));
    CUFFT_CHECK(cufftSetStream(c2r_handle_.get(), stream_));
  }

  const RegistrationSettings  &settings() const { return settings_; }
  const holoflow::core::TDesc &input_desc() const { return input_desc_; }

private:
  void flush_telemetry() {
    for (auto &slot : telemetry_) {
      if (!slot.pending)
        continue;
      const auto status = cudaEventQuery(slot.ready);
      if (status == cudaErrorNotReady)
        continue;
      CUDA_CHECK(status);
      const auto &estimate = *slot.estimate;
      logger()->debug(
          "[Registration] Frame {}: shift=({:.3f}, {:.3f}), rotation={:.3f}, score={:.3f}, "
          "PSR={:.2f}, accepted={}",
          slot.frame_index, estimate.shift_x, estimate.shift_y, estimate.rotation_degrees,
          estimate.rotation_score, estimate.psr, estimate.accepted != 0);
      slot.pending = false;
    }
  }

  void schedule_telemetry() {
    if (!logger()->should_log(spdlog::level::debug))
      return;
    for (auto &slot : telemetry_) {
      if (slot.pending)
        continue;
      slot.frame_index = frame_index_;
      slot.pending     = true;
      CUDA_CHECK(cudaMemcpyAsync(slot.estimate.get(), d_estimate_.get(), sizeof(*slot.estimate),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaEventRecord(slot.ready, stream_));
      return;
    }
    logger()->warn("[Registration] Telemetry queue full; frame {} telemetry dropped", frame_index_);
  }

  void handle_recording_events(holoflow::core::SyncCtx &ctx) {
    if (!ctx.event_reader)
      return;
    while (auto event = ctx.event_reader->try_pop()) {
      const auto type = event->data.value("type", std::string{});
      if (type == "start_recording") {
        ref_initialized_ = false;
        recording_       = true;
        frame_index_     = 0;
        logger()->debug("[Registration] Received start_recording; reference reset");
      } else if (type == "stop_recording") {
        ref_initialized_ = false;
        recording_       = false;
        logger()->debug("[Registration] Received stop_recording; reference reset");
      }
    }
  }

  void xcorr(float *odata, const float *idata) {
    auto *idata_nc = const_cast<float *>(idata);
    CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata_nc, d_freq1_.get()));

    int block_size = 256;
    int grid_size  = (static_cast<int>(freq_size_) + block_size - 1) / block_size;

    const auto width           = static_cast<int>(input_desc_.shape.back());
    const auto height          = static_cast<int>(input_desc_.shape[input_desc_.shape.size() - 2]);
    const auto frequency_width = width / 2 + 1;
    phase_cross_power_kernel<<<grid_size, block_size, 0, stream_>>>(d_freq1_.get(), d_freq2_.get(),
                                                                    frequency_width, width, height,
                                                                    static_cast<int>(freq_size_));

    CUFFT_CHECK(cufftExecC2R(c2r_handle_.get(), d_freq1_.get(), odata));
    CUDA_CHECK(cudaGetLastError());
  }

  void build_spectrum_magnitude(float *output, const cuFloatComplex *frequency, std::size_t width,
                                std::size_t height) {
    constexpr int block_size = 256;
    const int     grid_size  = (static_cast<int>(width * height) + block_size - 1) / block_size;
    spectrum_magnitude_kernel<<<grid_size, block_size, 0, stream_>>>(
        output, frequency, static_cast<int>(width / 2 + 1), static_cast<int>(width),
        static_cast<int>(height));
  }

  void estimate_rotation(const float *input, std::size_t width, std::size_t height) {
    auto *input_nc = const_cast<float *>(input);
    CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), input_nc, d_freq1_.get()));
    build_spectrum_magnitude(d_current_magnitude_.get(), d_freq1_.get(), width, height);
    rotation_scores_kernel<<<kRotationCandidateCount, 256, 0, stream_>>>(
        d_rotation_scores_.get(), d_current_magnitude_.get(), d_reference_magnitude_.get(),
        static_cast<int>(width), static_cast<int>(height));
    finalize_rotation_kernel<<<1, 1, 0, stream_>>>(d_estimate_.get(), d_rotation_scores_.get());
    CUDA_CHECK(cudaGetLastError());
  }

  void rotate_for_correlation(float *output, const float *input, std::size_t width,
                              std::size_t height) {
    const dim3 block_size(16, 16);
    const dim3 grid_size(static_cast<unsigned int>((width + block_size.x - 1) / block_size.x),
                         static_cast<unsigned int>((height + block_size.y - 1) / block_size.y));
    rotate_image_kernel<<<grid_size, block_size, 0, stream_>>>(
        output, input, d_estimate_.get(), static_cast<int>(width), static_cast<int>(height));
  }

  void estimate_shift(std::size_t width, std::size_t height) {
    CUDA_CHECK(cub::DeviceReduce::ArgMax(d_amax_tmp_.get(), amax_tmp_bytes_, d_xcorr_.get(),
                                         d_max_.get(), d_max_idx_.get(), width * height, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_psr_stats_.get(), 0, 2 * sizeof(float), stream_));
    constexpr int block_size = 256;
    const int     grid_size  = (static_cast<int>(width * height) + block_size - 1) / block_size;
    psr_stats_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_xcorr_.get(), d_max_idx_.get(), d_psr_stats_.get(), static_cast<int>(width),
        static_cast<int>(height));
    const float max_translation = 0.15f * static_cast<float>(std::min(width, height));
    finalize_estimate_kernel<<<1, 1, 0, stream_>>>(
        d_estimate_.get(), d_xcorr_.get(), d_max_.get(), d_max_idx_.get(), d_psr_stats_.get(),
        static_cast<int>(width), static_cast<int>(height), max_translation);
    CUDA_CHECK(cudaGetLastError());
  }

  void center_mean(float *odata, const float *idata, std::size_t b, std::size_t h, std::size_t w) {
    if (b != 1) {
      return;
    }

    const size_t  num_pixels = w * h;
    constexpr int block_dim  = 256;
    const size_t  grid_dim   = (num_pixels + block_dim - 1) / block_dim;

    CUDA_CHECK(cudaMemsetAsync(d_sum_.get(), 0, sizeof(float), stream_));
    roi_sum_kernel<<<static_cast<unsigned int>(grid_dim), block_dim, 0, stream_>>>(
        idata, d_select_roi_.get(), d_sum_.get(), static_cast<int>(num_pixels));
    f32_center_mask_kernel<<<static_cast<unsigned int>(grid_dim), block_dim, 0, stream_>>>(
        odata, idata, d_select_roi_.get(), static_cast<int>(w), static_cast<int>(h),
        settings_.radius, d_sum_.get(), d_select_count_.get());
    CUDA_CHECK(cudaGetLastError());
  }

  void apply_shifts(float *odata, uint8_t *valid, const float *idata, std::size_t b, std::size_t h,
                    std::size_t w) {
    if (b != 1) {
      return;
    }

    dim3 block_size(16, 16);
    dim3 grid_size(static_cast<unsigned int>((w + block_size.x - 1) / block_size.x),
                   static_cast<unsigned int>((h + block_size.y - 1) / block_size.y));
    f32_shift_subpixel_kernel<<<grid_size, block_size, 0, stream_>>>(
        odata, valid, idata, d_estimate_.get(), static_cast<int>(w), static_cast<int>(h));
    CUDA_CHECK(cudaGetLastError());
  }

  RegistrationSettings                  settings_;
  holoflow::core::TDesc                 input_desc_;
  holoflow::core::TDesc                 output_desc_;
  cudaStream_t                          stream_;
  DevPtr<float>                         d_mean_centered_;
  bool                                  ref_initialized_;
  bool                                  recording_ = false;
  size_t                                freq_size_;
  curaii::CufftHandle                   r2c_handle_;
  curaii::CufftHandle                   c2r_handle_;
  DevPtr<float>                         d_ref_;
  DevPtr<float>                         d_xcorr_;
  DevPtr<cuFloatComplex>                d_freq1_;
  DevPtr<cuFloatComplex>                d_freq2_;
  size_t                                sum_tmp_bytes_;
  DevPtr<uint8_t>                       d_sum_tmp_;
  DevPtr<float>                         d_sum_;
  size_t                                amax_tmp_bytes_;
  DevPtr<uint8_t>                       d_amax_tmp_;
  DevPtr<float>                         d_max_;
  DevPtr<int64_t>                       d_max_idx_;
  size_t                                select_tmp_bytes_;
  DevPtr<uint8_t>                       d_select_tmp_;
  DevPtr<int>                           d_select_count_;
  DevPtr<uint8_t>                       d_select_roi_;
  DevPtr<float>                         d_selected_;
  DevPtr<float>                         d_psr_stats_;
  DevPtr<RegistrationEstimate>          d_estimate_;
  DevPtr<float>                         d_rotated_;
  DevPtr<float>                         d_reference_magnitude_;
  DevPtr<float>                         d_current_magnitude_;
  DevPtr<float>                         d_rotation_scores_;
  std::array<RegistrationTelemetry, 16> telemetry_;
  size_t                                frame_index_ = 0;
};

// -------------------------------------------------------------------------------------------------
// RegistrationFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
RegistrationFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings) const {
  check(!input_descs.empty(), "No input descriptors provided");
  check(input_descs.size() == 1, "Registration expects exactly one input");

  const auto &input_desc = input_descs[0];
  const auto  settings   = jsettings.get<RegistrationSettings>();

  check(input_desc.rank() == 2 || input_desc.rank() == 3, "Input must be rank 2 or 3");
  check(input_desc.dtype == holoflow::core::DType::F32, "Input must be F32 type");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "Input must be in device memory");
  check(settings.radius >= 0.0f && settings.radius <= 1.0f, "radius not in [0, 1]");
  check(is_c_contiguous(input_desc), "Input must be C-contiguous");
  if (input_desc.rank() == 3) {
    check(input_desc.shape[0] == 1, "Only batch size 1 is supported");
  }

  holoflow::core::TDesc output_desc = input_desc;
  holoflow::core::TDesc valid_desc({1}, holoflow::core::DType::U8, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {input_desc},
      .output_descs  = {output_desc, valid_desc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false, false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
RegistrationFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings,
                            const holoflow::core::SyncCreateCtx   &ctx) const {
  logger()->info("Creating Registration sync task");
  auto result   = infer(input_descs, jsettings);
  auto settings = jsettings.get<RegistrationSettings>();

  const auto &input_desc = input_descs[0];

  const auto H = input_desc.shape[input_desc.shape.size() - 2];
  const auto W = input_desc.shape.back();

  auto d_mean_centered = make_unique_device_ptr<float>(W * H);

  bool   ref_initialized = false;
  size_t freq_size       = H * (W / 2 + 1);

  auto r2c_handle = curaii::CufftHandle();
  auto c2r_handle = curaii::CufftHandle();

  size_t r2c_ws = 0;
  size_t c2r_ws = 0;
  CUFFT_CHECK(cufftMakePlan2d(r2c_handle.get(), static_cast<int>(H), static_cast<int>(W), CUFFT_R2C,
                              &r2c_ws));
  CUFFT_CHECK(cufftMakePlan2d(c2r_handle.get(), static_cast<int>(H), static_cast<int>(W), CUFFT_C2R,
                              &c2r_ws));
  CUFFT_CHECK(cufftSetStream(r2c_handle.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(c2r_handle.get(), ctx.stream));

  auto d_ref   = make_unique_device_ptr<float>(W * H);
  auto d_xcorr = make_unique_device_ptr<float>(W * H);
  auto d_freq1 = make_unique_device_ptr<cuFloatComplex>(freq_size);
  auto d_freq2 = make_unique_device_ptr<cuFloatComplex>(freq_size);

  size_t amax_tmp_bytes = 0;
  auto   d_max          = make_unique_device_ptr<float>(1);
  auto   d_max_idx      = make_unique_device_ptr<int64_t>(1);
  CUDA_CHECK(cub::DeviceReduce::ArgMax((void *)nullptr, amax_tmp_bytes, (float *)nullptr,
                                       d_max.get(), d_max_idx.get(), W * H, ctx.stream));
  auto d_amax_tmp = make_unique_device_ptr<uint8_t>(amax_tmp_bytes);

  size_t select_tmp_bytes = 0;
  auto   d_select_count   = make_unique_device_ptr<int>(1);
  auto   d_select_roi     = make_unique_device_ptr<uint8_t>(W * H);
  auto   d_selected       = make_unique_device_ptr<float>(W * H);

  dim3 block_size(16, 16);
  dim3 grid_size(static_cast<unsigned int>((W + block_size.x - 1) / block_size.x),
                 static_cast<unsigned int>((H + block_size.y - 1) / block_size.y));

  ellipse_mask_kernel<<<grid_size, block_size, 0, ctx.stream>>>(
      d_select_roi.get(), static_cast<int>(W), static_cast<int>(H), settings.radius);
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cub::DeviceSelect::Flagged(static_cast<void *>(nullptr), select_tmp_bytes,
                                        static_cast<float *>(nullptr), d_select_roi.get(),
                                        d_selected.get(), d_select_count.get(), W * H, ctx.stream));
  auto d_select_tmp = make_unique_device_ptr<uint8_t>(select_tmp_bytes);

  CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp.get(), select_tmp_bytes, d_mean_centered.get(),
                                        d_select_roi.get(), d_selected.get(), d_select_count.get(),
                                        W * H, ctx.stream));

  int select_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&select_count, d_select_count.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, ctx.stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  size_t sum_tmp_bytes = 0;
  auto   d_sum         = make_unique_device_ptr<float>(1);
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, sum_tmp_bytes, static_cast<float *>(nullptr),
                                    d_sum.get(), select_count, ctx.stream));
  auto d_sum_tmp = make_unique_device_ptr<uint8_t>(sum_tmp_bytes);

  return std::make_unique<Registration>(
      settings, input_desc, result.output_descs[0], ctx.stream, std::move(d_mean_centered),
      ref_initialized, freq_size, std::move(r2c_handle), std::move(c2r_handle), std::move(d_ref),
      std::move(d_xcorr), std::move(d_freq1), std::move(d_freq2), sum_tmp_bytes,
      std::move(d_sum_tmp), std::move(d_sum), amax_tmp_bytes, std::move(d_amax_tmp),
      std::move(d_max), std::move(d_max_idx), select_tmp_bytes, std::move(d_select_tmp),
      std::move(d_select_count), std::move(d_select_roi), std::move(d_selected));
}

std::unique_ptr<holoflow::core::ISyncTask>
RegistrationFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                            std::span<const holoflow::core::TDesc>     input_descs,
                            const nlohmann::json                      &jsettings,
                            const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_registration = dynamic_cast<Registration *>(old_task.get());
  if (old_registration == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_input_desc = input_descs[0];
  const auto &old_input_desc = old_registration->input_desc();
  const auto  settings       = jsettings.get<RegistrationSettings>();
  const bool  can_reuse      = settings == old_registration->settings() &&
                               new_input_desc.shape == old_input_desc.shape &&
                               new_input_desc.strides == old_input_desc.strides &&
                               new_input_desc.dtype == old_input_desc.dtype &&
                               new_input_desc.mem_loc == old_input_desc.mem_loc;

  if (can_reuse) {
    old_registration->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
