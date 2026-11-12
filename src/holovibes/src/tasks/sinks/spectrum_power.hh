// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime_api.h>

namespace holovibes::tasks::sinks {

inline constexpr size_t kSpectrumPowerPartials = 64;

// Input is contiguous [batch, frequency, height, width]. Scratch holds bins * 64 floats;
// output holds bins floats. Both remain on the device, and work is enqueued on stream.
void compute_spectrum_power(const cuFloatComplex *input, float *scratch, float *output,
                            size_t batches, size_t bins, size_t pixels, cudaStream_t stream);

} // namespace holovibes::tasks::sinks
