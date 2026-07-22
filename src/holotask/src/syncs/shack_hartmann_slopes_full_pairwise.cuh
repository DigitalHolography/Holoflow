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

#pragma once

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holotask/syncs/shack_hartmann_slopes.hh"
#include "syncs/phase_correlation.cuh"
#include "syncs/shack_hartmann_geometry.hh"

namespace holotask::syncs::detail {

template <typename T> using PairwiseDevPtr = curaii::unique_device_ptr<T>;

class ShackHartmannSlopesTaskBase : public holoflow::core::ISyncTask {
public:
  virtual const ShackHartmannSlopeSettings &settings() const                   = 0;
  virtual const holoflow::core::TDesc      &input_desc() const                 = 0;
  virtual void                              update_stream(cudaStream_t stream) = 0;
};

inline curaii::CufftHandle make_dense_cfft2_plan(size_t height, size_t width, size_t batch,
                                                 cudaStream_t stream) {
  const auto ll_max = static_cast<size_t>(std::numeric_limits<long long>::max());
  if (height > ll_max || width > ll_max || batch == 0 || batch > ll_max ||
      (width != 0 && height > ll_max / width)) {
    throw std::invalid_argument(
        "ShackHartmannSlopes: full-pairwise FFT dimensions exceed cuFFT limits");
  }

  const size_t        n_fft = height * width;
  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));

  long long dimensions[2] = {static_cast<long long>(height), static_cast<long long>(width)};
  long long embed[2]      = {static_cast<long long>(height), static_cast<long long>(width)};
  size_t    work_size     = 0;
  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, dimensions, embed, 1,
                                  static_cast<long long>(n_fft), CUDA_C_32F, embed, 1,
                                  static_cast<long long>(n_fft), CUDA_C_32F,
                                  static_cast<long long>(batch), &work_size, CUDA_C_32F));
  return plan;
}

__global__ void active_image_means(const std::byte *__restrict__ input,
                                   const size_t *__restrict__ active_dense_indices,
                                   float *__restrict__ means, size_t active_count, size_t sx,
                                   size_t height, size_t width, size_t sy_stride, size_t sx_stride,
                                   size_t y_stride, size_t x_stride,
                                   CrossCorrelation2Settings::Ellipse roi) {
  const size_t active_index = blockIdx.x;
  if (active_index >= active_count) {
    return;
  }

  const size_t dense_index = active_dense_indices[active_index];
  const auto  *image = input + (dense_index / sx) * sy_stride + (dense_index % sx) * sx_stride;
  float        sum   = 0.0f;
  int          count = 0;

  for (size_t pixel = threadIdx.x; pixel < height * width; pixel += blockDim.x) {
    const size_t y = pixel / width;
    const size_t x = pixel % width;
    if (ellipse_sq_distance(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width),
                            static_cast<int>(height), roi) <= 1.0f) {
      sum += *reinterpret_cast<const float *>(image + y * y_stride + x * x_stride);
      ++count;
    }
  }

  __shared__ float shared_sum[256];
  __shared__ int   shared_count[256];
  shared_sum[threadIdx.x]   = sum;
  shared_count[threadIdx.x] = count;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
      shared_count[threadIdx.x] += shared_count[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    means[active_index] =
        shared_count[0] == 0 ? 0.0f : shared_sum[0] / static_cast<float>(shared_count[0]);
  }
}

__global__ void preprocess_active_images(const std::byte *__restrict__ input,
                                         const size_t *__restrict__ active_dense_indices,
                                         const float *__restrict__ means,
                                         cuFloatComplex *__restrict__ spectra, size_t active_count,
                                         size_t sx, size_t height, size_t width, size_t sy_stride,
                                         size_t sx_stride, size_t y_stride, size_t x_stride,
                                         CrossCorrelation2Settings::Ellipse roi) {
  const size_t index            = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t pixels_per_image = height * width;
  if (index >= active_count * pixels_per_image) {
    return;
  }

  const size_t active_index = index / pixels_per_image;
  const size_t pixel        = index % pixels_per_image;
  const size_t y            = pixel / width;
  const size_t x            = pixel % width;
  const size_t dense_index  = active_dense_indices[active_index];
  const auto  *image = input + (dense_index / sx) * sy_stride + (dense_index % sx) * sx_stride;
  const float  value = *reinterpret_cast<const float *>(image + y * y_stride + x * x_stride);
  const float  distance =
      ellipse_sq_distance(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width),
                          static_cast<int>(height), roi);

  const float tapered = preprocess_phase_correlation_value(value, means[active_index], distance);
  spectra[index]      = make_cuFloatComplex(tapered, 0.0f);
}

__global__ void generate_complete_graph_pairs(size_t edge_offset, size_t active_count,
                                              size_t pair_count, size_t *__restrict__ sources,
                                              size_t *__restrict__ destinations) {
  const size_t batch_edge = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch_edge >= pair_count) {
    return;
  }

  size_t edge = edge_offset + batch_edge;
  size_t src  = 0;
  size_t row  = active_count - 1;
  while (edge >= row) {
    edge -= row;
    ++src;
    --row;
  }
  sources[batch_edge]      = src;
  destinations[batch_edge] = src + 1 + edge;
}

__global__ void construct_pair_cross_power(const cuFloatComplex *__restrict__ spectra,
                                           const size_t *__restrict__ sources,
                                           const size_t *__restrict__ destinations,
                                           cuFloatComplex *__restrict__ pair_buffer,
                                           size_t pair_count, size_t pixels_per_image) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pair_count * pixels_per_image) {
    return;
  }

  const size_t pair  = index / pixels_per_image;
  const size_t pixel = index % pixels_per_image;
  pair_buffer[index] =
      normalized_cross_power(spectra[sources[pair] * pixels_per_image + pixel],
                             spectra[destinations[pair] * pixels_per_image + pixel]);
}

// measurement(i, j) = g[i] - g[j]. A moving image i translated positively from reference j
// produces a positive periodic peak, matching the SingleReference path.
__global__ void recover_complex_phase_correlation_peaks(const cuFloatComplex *__restrict__ maps,
                                                        float2 *__restrict__ shifts,
                                                        size_t map_count, size_t height,
                                                        size_t width, float inverse_fft_scale) {
  const size_t map_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (map_index >= map_count) {
    return;
  }

  const cuFloatComplex *map        = maps + map_index * height * width;
  float                 best_value = -FLT_MAX;
  size_t                peak_y     = 0;
  size_t                peak_x     = 0;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const float value = map[y * width + x].x * inverse_fft_scale;
      if (value > best_value) {
        best_value = value;
        peak_y     = y;
        peak_x     = x;
      }
    }
  }

  const size_t x_minus = (peak_x + width - 1) % width;
  const size_t x_plus  = (peak_x + 1) % width;
  const size_t y_minus = (peak_y + height - 1) % height;
  const size_t y_plus  = (peak_y + 1) % height;
  const float  center  = map[peak_y * width + peak_x].x * inverse_fft_scale;
  const float  dx =
      parabolic_peak_offset(map[peak_y * width + x_minus].x * inverse_fft_scale, center,
                            map[peak_y * width + x_plus].x * inverse_fft_scale);
  const float dy =
      parabolic_peak_offset(map[y_minus * width + peak_x].x * inverse_fft_scale, center,
                            map[y_plus * width + peak_x].x * inverse_fft_scale);
  shifts[map_index] = {circular_signed_coordinate(peak_x, width) + dx,
                       circular_signed_coordinate(peak_y, height) + dy};
}

// Uniform complete-graph accumulation. Atomic ordering can introduce small run-to-run roundoff,
// but each edge contributes exactly opposite values to its two incident nodes.
__global__ void accumulate_complete_graph_rhs(const size_t *__restrict__ sources,
                                              const size_t *__restrict__ destinations,
                                              const float2 *__restrict__ shifts,
                                              float2 *__restrict__ rhs, size_t pair_count,
                                              float slope_per_pixel_x, float slope_per_pixel_y) {
  const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) {
    return;
  }

  const float sx = shifts[pair].x * slope_per_pixel_x;
  const float sy = shifts[pair].y * slope_per_pixel_y;
  atomicAdd(&rhs[sources[pair]].x, sx);
  atomicAdd(&rhs[sources[pair]].y, sy);
  atomicAdd(&rhs[destinations[pair]].x, -sx);
  atomicAdd(&rhs[destinations[pair]].y, -sy);
}

__global__ void solve_complete_graph_and_scatter(const float2 *__restrict__ rhs,
                                                 const size_t *__restrict__ active_dense_indices,
                                                 float *__restrict__ slopes, size_t active_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float mean_x = 0.0f;
  float mean_y = 0.0f;
  for (size_t i = 0; i < active_count; ++i) {
    mean_x += rhs[i].x / static_cast<float>(active_count);
    mean_y += rhs[i].y / static_cast<float>(active_count);
  }
  mean_x /= static_cast<float>(active_count);
  mean_y /= static_cast<float>(active_count);

  for (size_t i = 0; i < active_count; ++i) {
    const size_t dense    = active_dense_indices[i];
    slopes[2 * dense]     = rhs[i].x / static_cast<float>(active_count) - mean_x;
    slopes[2 * dense + 1] = rhs[i].y / static_cast<float>(active_count) - mean_y;
  }
}

class FullPairwiseShackHartmannSlopes final : public ShackHartmannSlopesTaskBase {
public:
  FullPairwiseShackHartmannSlopes(
      ShackHartmannSlopeSettings settings, holoflow::core::TDesc input_desc, size_t active_count,
      size_t edge_count, size_t pair_capacity, PairwiseDevPtr<size_t> active_dense_indices,
      PairwiseDevPtr<float> means, PairwiseDevPtr<cuFloatComplex> active_spectra,
      PairwiseDevPtr<size_t> pair_sources, PairwiseDevPtr<size_t> pair_destinations,
      PairwiseDevPtr<cuFloatComplex> pair_buffer, PairwiseDevPtr<float2> pair_shifts,
      PairwiseDevPtr<float2> rhs, curaii::CufftHandle forward_plan,
      curaii::CufftHandle inverse_plan, std::unique_ptr<curaii::CufftHandle> tail_inverse_plan,
      cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)),
        active_count_(active_count), edge_count_(edge_count), pair_capacity_(pair_capacity),
        active_dense_indices_(std::move(active_dense_indices)), means_(std::move(means)),
        active_spectra_(std::move(active_spectra)), pair_sources_(std::move(pair_sources)),
        pair_destinations_(std::move(pair_destinations)), pair_buffer_(std::move(pair_buffer)),
        pair_shifts_(std::move(pair_shifts)), rhs_(std::move(rhs)),
        forward_plan_(std::move(forward_plan)), inverse_plan_(std::move(inverse_plan)),
        tail_inverse_plan_(std::move(tail_inverse_plan)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const size_t           sx               = input_desc_.shape[2];
    const size_t           height           = input_desc_.shape[3];
    const size_t           width            = input_desc_.shape[4];
    const size_t           pixels_per_image = height * width;
    const size_t           active_elements  = active_count_ * pixels_per_image;
    constexpr unsigned int block            = 256;
    const auto             grid_for         = [](size_t count) {
      return static_cast<unsigned int>((count + block - 1) / block);
    };

    CUDA_CHECK(cudaMemsetAsync(rhs_.get(), 0, active_count_ * sizeof(float2), stream_));
    CUDA_CHECK(cudaMemsetAsync(ctx.outputs[0].data(), 0,
                               input_desc_.shape[1] * sx * 2 * sizeof(float), stream_));

    active_image_means<<<static_cast<unsigned int>(active_count_), block, 0, stream_>>>(
        ctx.inputs[0].data(), active_dense_indices_.get(), means_.get(), active_count_, sx, height,
        width, input_desc_.strides[1], input_desc_.strides[2], input_desc_.strides[3],
        input_desc_.strides[4], settings_.correlation_roi);
    preprocess_active_images<<<grid_for(active_elements), block, 0, stream_>>>(
        ctx.inputs[0].data(), active_dense_indices_.get(), means_.get(), active_spectra_.get(),
        active_count_, sx, height, width, input_desc_.strides[1], input_desc_.strides[2],
        input_desc_.strides[3], input_desc_.strides[4], settings_.correlation_roi);
    CUFFT_CHECK(cufftXtExec(forward_plan_.get(), active_spectra_.get(), active_spectra_.get(),
                            CUFFT_FORWARD));

    const float delta_out_x =
        settings_.lambda * settings_.z / (static_cast<float>(width) * settings_.dx);
    const float delta_out_y =
        settings_.lambda * settings_.z / (static_cast<float>(height) * settings_.dy);
    const float slope_per_pixel_x = delta_out_x / settings_.z;
    const float slope_per_pixel_y = delta_out_y / settings_.z;
    const float inverse_fft_scale = 1.0f / static_cast<float>(pixels_per_image);

    for (size_t edge_offset = 0; edge_offset < edge_count_; edge_offset += pair_capacity_) {
      const size_t pair_count = std::min(pair_capacity_, edge_count_ - edge_offset);
      generate_complete_graph_pairs<<<grid_for(pair_count), block, 0, stream_>>>(
          edge_offset, active_count_, pair_count, pair_sources_.get(), pair_destinations_.get());
      construct_pair_cross_power<<<grid_for(pair_count * pixels_per_image), block, 0, stream_>>>(
          active_spectra_.get(), pair_sources_.get(), pair_destinations_.get(), pair_buffer_.get(),
          pair_count, pixels_per_image);

      const cufftHandle inverse =
          pair_count == pair_capacity_ ? inverse_plan_.get() : tail_inverse_plan_->get();
      CUFFT_CHECK(cufftXtExec(inverse, pair_buffer_.get(), pair_buffer_.get(), CUFFT_INVERSE));
      recover_complex_phase_correlation_peaks<<<grid_for(pair_count), block, 0, stream_>>>(
          pair_buffer_.get(), pair_shifts_.get(), pair_count, height, width, inverse_fft_scale);
      accumulate_complete_graph_rhs<<<grid_for(pair_count), block, 0, stream_>>>(
          pair_sources_.get(), pair_destinations_.get(), pair_shifts_.get(), rhs_.get(), pair_count,
          slope_per_pixel_x, slope_per_pixel_y);
    }

    solve_complete_graph_and_scatter<<<1, 1, 0, stream_>>>(
        rhs_.get(), active_dense_indices_.get(), reinterpret_cast<float *>(ctx.outputs[0].data()),
        active_count_);
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const ShackHartmannSlopeSettings &settings() const override { return settings_; }
  const holoflow::core::TDesc      &input_desc() const override { return input_desc_; }

  void update_stream(cudaStream_t stream) override {
    if (stream_ == stream) {
      return;
    }
    CUFFT_CHECK(cufftSetStream(forward_plan_.get(), stream));
    CUFFT_CHECK(cufftSetStream(inverse_plan_.get(), stream));
    if (tail_inverse_plan_) {
      CUFFT_CHECK(cufftSetStream(tail_inverse_plan_->get(), stream));
    }
    stream_ = stream;
  }

private:
  ShackHartmannSlopeSettings           settings_;
  holoflow::core::TDesc                input_desc_;
  size_t                               active_count_;
  size_t                               edge_count_;
  size_t                               pair_capacity_;
  PairwiseDevPtr<size_t>               active_dense_indices_;
  PairwiseDevPtr<float>                means_;
  PairwiseDevPtr<cuFloatComplex>       active_spectra_;
  PairwiseDevPtr<size_t>               pair_sources_;
  PairwiseDevPtr<size_t>               pair_destinations_;
  PairwiseDevPtr<cuFloatComplex>       pair_buffer_;
  PairwiseDevPtr<float2>               pair_shifts_;
  PairwiseDevPtr<float2>               rhs_;
  curaii::CufftHandle                  forward_plan_;
  curaii::CufftHandle                  inverse_plan_;
  std::unique_ptr<curaii::CufftHandle> tail_inverse_plan_;
  cudaStream_t                         stream_;
};

inline std::unique_ptr<holoflow::core::ISyncTask> make_full_pairwise_shack_hartmann_slopes(
    ShackHartmannSlopeSettings settings, const holoflow::core::TDesc &input,
    const ShackHartmannGeometry &geometry, const holoflow::core::SyncCreateCtx &ctx) {
  std::vector<size_t> active_dense_host;
  active_dense_host.reserve(geometry.samples.size());
  for (size_t i = 0; i < geometry.samples.size(); ++i) {
    if (geometry.samples[i].active) {
      active_dense_host.push_back(i);
    }
  }

  const size_t active_count  = active_dense_host.size();
  const size_t edge_count    = active_count * (active_count - 1) / 2;
  const size_t pair_capacity = std::min(settings.pair_batch_size, edge_count);
  const size_t pixels        = input.shape[3] * input.shape[4];
  const size_t tail_count    = edge_count % pair_capacity;

  auto active_dense = curaii::make_unique_device_ptr<size_t>(active_count, ctx.stream);
  CUDA_CHECK(cudaMemcpyAsync(active_dense.get(), active_dense_host.data(),
                             active_count * sizeof(size_t), cudaMemcpyHostToDevice, ctx.stream));
  auto means   = curaii::make_unique_device_ptr<float>(active_count, ctx.stream);
  auto spectra = curaii::make_unique_device_ptr<cuFloatComplex>(active_count * pixels, ctx.stream);
  auto sources = curaii::make_unique_device_ptr<size_t>(pair_capacity, ctx.stream);
  auto destinations = curaii::make_unique_device_ptr<size_t>(pair_capacity, ctx.stream);
  auto pair_buffer =
      curaii::make_unique_device_ptr<cuFloatComplex>(pair_capacity * pixels, ctx.stream);
  auto shifts = curaii::make_unique_device_ptr<float2>(pair_capacity, ctx.stream);
  auto rhs    = curaii::make_unique_device_ptr<float2>(active_count, ctx.stream);

  auto forward_plan =
      make_dense_cfft2_plan(input.shape[3], input.shape[4], active_count, ctx.stream);
  auto inverse_plan =
      make_dense_cfft2_plan(input.shape[3], input.shape[4], pair_capacity, ctx.stream);
  std::unique_ptr<curaii::CufftHandle> tail_plan;
  if (tail_count != 0) {
    tail_plan = std::make_unique<curaii::CufftHandle>(
        make_dense_cfft2_plan(input.shape[3], input.shape[4], tail_count, ctx.stream));
  }

  return std::make_unique<FullPairwiseShackHartmannSlopes>(
      std::move(settings), input, active_count, edge_count, pair_capacity, std::move(active_dense),
      std::move(means), std::move(spectra), std::move(sources), std::move(destinations),
      std::move(pair_buffer), std::move(shifts), std::move(rhs), std::move(forward_plan),
      std::move(inverse_plan), std::move(tail_plan), ctx.stream);
}

} // namespace holotask::syncs::detail
