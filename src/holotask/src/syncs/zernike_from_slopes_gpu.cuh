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

#include <cuda_runtime_api.h>

namespace holotask::syncs::detail {

constexpr size_t kMaxZernikeModes = 9;

struct ZernikeFromSlopesGpuSettings {
  size_t active_count      = 0;
  size_t observable_count  = 0;
  size_t output_count      = 0;
  size_t sx_count          = 0;
  size_t stride_y          = 0;
  size_t stride_x          = 0;
  size_t stride_component  = 0;
  float  radians_per_meter = 0.0f;
  int    observable_positions[kMaxZernikeModes]{};
};

cudaError_t launch_zernike_from_slopes_gpu(const float *slopes, float *output,
                                           const size_t *active_samples, const float *derivatives_x,
                                           const float                        *derivatives_y,
                                           const float                        *regularized_gram,
                                           const ZernikeFromSlopesGpuSettings &settings,
                                           cudaStream_t                        stream);

} // namespace holotask::syncs::detail
