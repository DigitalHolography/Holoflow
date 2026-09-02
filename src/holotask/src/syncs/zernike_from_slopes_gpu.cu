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

#include "syncs/zernike_from_slopes_gpu.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace holotask::syncs::detail {

namespace {

__device__ void solve_linear_system(float *matrix, float *rhs, float *solution, size_t size) {
  constexpr float singular_epsilon = 1e-12f;

  for (size_t column = 0; column < size; ++column) {
    size_t pivot      = column;
    float  best_value = fabsf(matrix[column * size + column]);
    for (size_t row = column + 1; row < size; ++row) {
      const float candidate = fabsf(matrix[row * size + column]);
      if (candidate > best_value) {
        best_value = candidate;
        pivot      = row;
      }
    }

    if (best_value < singular_epsilon) {
      return;
    }

    if (pivot != column) {
      for (size_t j = column; j < size; ++j) {
        const float value         = matrix[pivot * size + j];
        matrix[pivot * size + j]  = matrix[column * size + j];
        matrix[column * size + j] = value;
      }
      const float value = rhs[pivot];
      rhs[pivot]        = rhs[column];
      rhs[column]       = value;
    }

    const float pivot_value = matrix[column * size + column];
    for (size_t j = column; j < size; ++j) {
      matrix[column * size + j] /= pivot_value;
    }
    rhs[column] /= pivot_value;

    for (size_t row = column + 1; row < size; ++row) {
      const float factor = matrix[row * size + column];
      for (size_t j = column; j < size; ++j) {
        matrix[row * size + j] -= factor * matrix[column * size + j];
      }
      rhs[row] -= factor * rhs[column];
    }
  }

  for (int row = static_cast<int>(size) - 1; row >= 0; --row) {
    float value = rhs[static_cast<size_t>(row)];
    for (size_t column = static_cast<size_t>(row) + 1; column < size; ++column) {
      value -= matrix[static_cast<size_t>(row) * size + column] * solution[column];
    }
    solution[static_cast<size_t>(row)] = value;
  }
}

__global__ void zernike_from_slopes_kernel(const float *slopes, float *output,
                                           const size_t *active_samples, const float *derivatives_x,
                                           const float                 *derivatives_y,
                                           const float                 *regularized_gram,
                                           ZernikeFromSlopesGpuSettings settings) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (size_t position = 0; position < settings.output_count; ++position) {
    output[position] = 0.0f;
  }
  if (settings.observable_count == 0) {
    return;
  }

  float matrix[kMaxZernikeModes * kMaxZernikeModes]{};
  float rhs[kMaxZernikeModes]{};
  float solution[kMaxZernikeModes]{};

  for (size_t sample = 0; sample < settings.active_count; ++sample) {
    const size_t sample_index = active_samples[sample];
    const size_t sy           = sample_index / settings.sx_count;
    const size_t sx           = sample_index % settings.sx_count;
    const auto  *sample_data  = reinterpret_cast<const std::uint8_t *>(slopes) +
                              sy * settings.stride_y + sx * settings.stride_x;
    const float slope_x = *reinterpret_cast<const float *>(sample_data);
    const float slope_y = *reinterpret_cast<const float *>(sample_data + settings.stride_component);

    const size_t derivative_offset = sample * settings.observable_count;
    for (size_t mode = 0; mode < settings.observable_count; ++mode) {
      rhs[mode] += derivatives_x[derivative_offset + mode] * slope_x +
                   derivatives_y[derivative_offset + mode] * slope_y;
    }
  }

  const size_t matrix_size = settings.observable_count * settings.observable_count;
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix[i] = regularized_gram[i];
  }
  solve_linear_system(matrix, rhs, solution, settings.observable_count);

  for (size_t mode = 0; mode < settings.observable_count; ++mode) {
    output[settings.observable_positions[mode]] = solution[mode] * settings.radians_per_meter;
  }
}

} // namespace

cudaError_t launch_zernike_from_slopes_gpu(const float *slopes, float *output,
                                           const size_t *active_samples, const float *derivatives_x,
                                           const float                        *derivatives_y,
                                           const float                        *regularized_gram,
                                           const ZernikeFromSlopesGpuSettings &settings,
                                           cudaStream_t                        stream) {
  zernike_from_slopes_kernel<<<1, 1, 0, stream>>>(slopes, output, active_samples, derivatives_x,
                                                  derivatives_y, regularized_gram, settings);
  return cudaGetLastError();
}

} // namespace holotask::syncs::detail
