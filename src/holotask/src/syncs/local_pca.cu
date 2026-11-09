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

#include "local_pca.hh"

#include <cstddef>

#include "curaii/cuda.hh"

namespace holotask::syncs {

namespace {

constexpr unsigned int kBlockSize = 256;

__global__ void gather_patches_kernel(const float *__restrict__ input,
                                      float *__restrict__ patches, size_t pixel_begin,
                                      size_t tile_pixels, int features, int height, int width,
                                      int window_size) {
  const size_t patch_sample = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int    samples      = window_size * window_size;
  if (patch_sample >= tile_pixels * static_cast<size_t>(samples)) {
    return;
  }

  const size_t tile_pixel = patch_sample / samples;
  const int    sample     = static_cast<int>(patch_sample % samples);
  const size_t pixel      = pixel_begin + tile_pixel;
  const int    raw_x      = static_cast<int>(pixel % width) + sample % window_size - window_size / 2;
  const int    raw_y      = static_cast<int>(pixel / width) + sample / window_size - window_size / 2;
  const int    x_mod      = raw_x % width;
  const int    y_mod      = raw_y % height;
  const int    x          = x_mod < 0 ? x_mod + width : x_mod;
  const int    y          = y_mod < 0 ? y_mod + height : y_mod;
  const size_t source     = static_cast<size_t>(y) * width + x;
  const size_t frame_size = static_cast<size_t>(height) * width;
  const size_t base       = tile_pixel * features * samples + sample;

  for (int feature = 0; feature < features; ++feature) {
    patches[base + static_cast<size_t>(feature) * samples] =
        input[static_cast<size_t>(feature) * frame_size + source];
  }
}

__global__ void project_center_kernel(const float *__restrict__ input,
                                      const float *__restrict__ eigenvectors,
                                      float *__restrict__ output, size_t pixel_begin,
                                      size_t tile_pixels, int features, int height, int width,
                                      int begin, int end) {
  const size_t tile_pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile_pixel >= tile_pixels) {
    return;
  }

  const size_t frame_size = static_cast<size_t>(height) * width;
  const size_t pixel      = pixel_begin + tile_pixel;
  const float *vectors    = eigenvectors + tile_pixel * features * features;

  for (int component = begin; component < end; ++component) {
    const float *vector = vectors + static_cast<size_t>(component) * features;
    float        value  = 0.0f;
    for (int feature = 0; feature < features; ++feature) {
      value += input[static_cast<size_t>(feature) * frame_size + pixel] * vector[feature];
    }
    output[static_cast<size_t>(component - begin) * frame_size + pixel] = value;
  }
}

} // namespace

void enqueue_local_pca_patches(const float *input, float *patches, size_t batch,
                               size_t pixel_begin, size_t tile_pixels, int features, int height,
                               int width, int window_size, cudaStream_t stream) {
  const size_t frame_size = static_cast<size_t>(height) * width;
  const size_t samples    = static_cast<size_t>(window_size) * window_size;
  const size_t threads    = tile_pixels * samples;
  const auto  blocks     = static_cast<unsigned int>((threads + kBlockSize - 1) / kBlockSize);
  gather_patches_kernel<<<blocks, kBlockSize, 0, stream>>>(
      input + batch * features * frame_size, patches, pixel_begin, tile_pixels, features, height,
      width, window_size);
  CUDA_CHECK(cudaGetLastError());
}

void enqueue_local_pca_projection(const float *input, const float *eigenvectors, float *output,
                                  size_t batch, size_t pixel_begin, size_t tile_pixels,
                                  int features, int height, int width, int begin, int end,
                                  cudaStream_t stream) {
  const size_t frame_size = static_cast<size_t>(height) * width;
  const auto blocks = static_cast<unsigned int>((tile_pixels + kBlockSize - 1) / kBlockSize);
  project_center_kernel<<<blocks, kBlockSize, 0, stream>>>(
      input + batch * features * frame_size, eigenvectors,
      output + batch * static_cast<size_t>(end - begin) * frame_size, pixel_begin, tile_pixels,
      features, height, width, begin, end);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace holotask::syncs
