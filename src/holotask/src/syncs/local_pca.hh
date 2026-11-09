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

#include <cuda_runtime.h>

#include <cstddef>

namespace holotask::syncs {

// Each tile entry is a column-major window_size² x features matrix. Pixels outside the image
// wrap around both image dimensions, so every neighborhood has window_size² samples.
void enqueue_local_pca_patches(const float *input, float *patches, size_t batch,
                               size_t pixel_begin, size_t tile_pixels, int features, int height,
                               int width, int window_size, cudaStream_t stream);

// Project each center pixel using its tile entry's eigenvectors (columns in ascending order).
void enqueue_local_pca_projection(const float *input, const float *eigenvectors, float *output,
                                  size_t batch, size_t pixel_begin, size_t tile_pixels,
                                  int features, int height, int width, int begin, int end,
                                  cudaStream_t stream);

} // namespace holotask::syncs
