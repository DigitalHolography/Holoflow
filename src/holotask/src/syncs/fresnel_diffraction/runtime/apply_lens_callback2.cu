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

#include <cuComplex.h>

struct ApplyLensCallerInfo {
  cuFloatComplex *lens;
  float          *magnitude_output;
};

__device__ cuFloatComplex apply_lens_callback2(void *data, size_t offset, void *callerInfo,
                                               void *sharedPtr) {
  auto *info = static_cast<ApplyLensCallerInfo *>(callerInfo);
#if FRESNEL_REAL_INPUT
  auto val = make_cuComplex(static_cast<float *>(data)[offset], 0.f);
#else
  auto val = static_cast<cuFloatComplex *>(data)[offset];
#endif
#if FRESNEL_CONTIGUOUS
  size_t lens_idx = offset % IDIST;
#else
  size_t local_offset = offset % IDIST;
  size_t row          = local_offset / STRIDE_H;
  size_t col          = (local_offset % STRIDE_H) / ISTRIDE;
  size_t lens_idx     = row * WIDTH + col;
#endif
  return cuCmulf(val, info->lens[lens_idx]);
}

__device__ void store_magnitude_callback(void *data, unsigned long long offset,
                                         cuFloatComplex element, void *callerInfo,
                                         void *sharedPtr) {
  auto *info                     = static_cast<ApplyLensCallerInfo *>(callerInfo);
  info->magnitude_output[offset] = cuCabsf(element);
}
