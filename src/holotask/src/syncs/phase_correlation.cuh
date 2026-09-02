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

#include <cstddef>

#include <cuComplex.h>
#include <math_constants.h>

#include "holotask/syncs/cross_correlation2.hh"

namespace holotask::syncs::detail {

inline constexpr unsigned int kPhaseCorrelationPeakBlockSize = 256;

struct PhaseCorrelationPeak {
  float  value;
  size_t index;
};

__device__ inline PhaseCorrelationPeak select_phase_correlation_peak(
    PhaseCorrelationPeak lhs, PhaseCorrelationPeak rhs) {
  return rhs.value > lhs.value || (rhs.value == lhs.value && rhs.index < lhs.index) ? rhs : lhs;
}

__device__ inline PhaseCorrelationPeak reduce_phase_correlation_peak(
    PhaseCorrelationPeak candidate, PhaseCorrelationPeak *__restrict__ shared_peaks) {
  const unsigned int thread = threadIdx.x;
  shared_peaks[thread]       = candidate;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      shared_peaks[thread] =
          select_phase_correlation_peak(shared_peaks[thread], shared_peaks[thread + stride]);
    }
    __syncthreads();
  }
  return shared_peaks[0];
}

__device__ inline float ellipse_sq_distance(int x, int y, int width, int height,
                                            CrossCorrelation2Settings::Ellipse roi) {
  if (width <= 0 || height <= 0 || roi.rx <= 0.0f || roi.ry <= 0.0f) {
    return 2.0f;
  }

  const float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
  const float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
  const float dx = xn - roi.cx;
  const float dy = yn - roi.cy;
  const float th = roi.angle * (CUDART_PI_F / 180.0f);
  const float c  = cosf(th);
  const float s  = sinf(th);
  const float xr = c * dx + s * dy;
  const float yr = -s * dx + c * dy;

  return (xr * xr) / (roi.rx * roi.rx) + (yr * yr) / (roi.ry * roi.ry);
}

__device__ inline float cosine_ellipse_taper(float squared_distance) {
  return 0.5f * (1.0f + cosf(CUDART_PI_F * sqrtf(squared_distance)));
}

__device__ inline float preprocess_phase_correlation_value(float value, float mean,
                                                           float squared_distance) {
  return squared_distance <= 1.0f ? (value - mean) * cosine_ellipse_taper(squared_distance) : 0.0f;
}

__device__ inline cuFloatComplex normalized_cross_power(cuFloatComplex moving,
                                                        cuFloatComplex reference) {
  const cuFloatComplex product   = cuCmulf(moving, cuConjf(reference));
  const float          magnitude = sqrtf(product.x * product.x + product.y * product.y);
  if (magnitude <= 1e-12f) {
    return make_cuFloatComplex(0.0f, 0.0f);
  }
  return make_cuFloatComplex(product.x / magnitude, product.y / magnitude);
}

__device__ inline float parabolic_peak_offset(float value_minus, float value_zero,
                                              float value_plus) {
  const float denominator = value_minus - 2.0f * value_zero + value_plus;
  return fabsf(denominator) <= 1e-9f ? 0.0f : 0.5f * (value_minus - value_plus) / denominator;
}

__device__ inline float circular_signed_coordinate(size_t peak, size_t size) {
  const size_t positive_limit = (size + 1) / 2;
  return peak < positive_limit ? static_cast<float>(peak)
                               : static_cast<float>(peak) - static_cast<float>(size);
}

} // namespace holotask::syncs::detail
