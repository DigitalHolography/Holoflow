// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "curaii/cuda.hh"
#include "tasks/sinks/spectrum_power.hh"

namespace holovibes::tasks::sinks {
namespace {

TEST(SpectrumPower, AveragesSquaredComplexMagnitudeAcrossBatchesAndPixels) {
  for (size_t pixels : {1ULL, 15ULL, 20003ULL}) {
    constexpr size_t            batches = 3;
    constexpr size_t            bins    = 5;
    std::vector<cuFloatComplex> input(batches * bins * pixels);
    std::vector<double>         reference(bins, 0.0);
    for (size_t batch = 0; batch < batches; ++batch) {
      for (size_t bin = 0; bin < bins; ++bin) {
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
          const float real                             = static_cast<float>(batch + 2 * bin);
          const float imag                             = static_cast<float>(pixel % 7) - 3.0f;
          input[(batch * bins + bin) * pixels + pixel] = make_cuFloatComplex(real, imag);
          reference[bin] += real * real + imag * imag;
        }
      }
    }
    auto device_input = curaii::make_unique_device_ptr<cuFloatComplex>(input.size());
    auto scratch      = curaii::make_unique_device_ptr<float>(bins * kSpectrumPowerPartials);
    auto output       = curaii::make_unique_device_ptr<float>(bins);
    curaii::CudaStream stream;
    CUDA_CHECK(cudaMemcpyAsync(device_input.get(), input.data(),
                               input.size() * sizeof(cuFloatComplex), cudaMemcpyHostToDevice,
                               stream.get()));
    compute_spectrum_power(device_input.get(), scratch.get(), output.get(), batches, bins, pixels,
                           stream.get());
    std::vector<float> actual(bins);
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.get(), bins * sizeof(float),
                               cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    for (size_t bin = 0; bin < bins; ++bin) {
      const double expected = reference[bin] / static_cast<double>(batches * pixels);
      EXPECT_NEAR(actual[bin], expected, std::max(1.0, expected) * 1e-5)
          << "bin=" << bin << ", pixels=" << pixels;
    }
  }
}

} // namespace
} // namespace holovibes::tasks::sinks
