// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "holotask/syncs/focus_stack.hh"
#include "holotask/syncs/temporal_fft.hh"
#include "sync_task_runner.hh"

namespace {
template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
struct Complex {
  float real, imaginary;
};
} // namespace

TEST(FocusVolumeTest, ConstantHologramProducesFiniteNonblackPlanes) {
  using namespace holoflow::core;
  const TDesc                         input({1, 4, 4}, DType::U8, MemLoc::Host);
  const auto                          data = as_bytes(std::vector<std::uint8_t>(16, 255));
  holotask::syncs::FocusStackSettings settings;
  settings.planes  = 2;
  settings.padding = false;
  const auto run =
      holonp_test::run_sync_factory(holotask::syncs::FocusStackFactory{}, std::vector<TDesc>{input},
                                    std::vector<std::vector<std::byte>>{data}, settings);
  ASSERT_EQ(run.output_bytes.size(), 1);
  ASSERT_EQ(run.output_bytes[0].size(), 32 * sizeof(float));
  const auto *pixels = reinterpret_cast<const float *>(run.output_bytes[0].data());
  for (std::size_t i = 0; i < 32; ++i) {
    EXPECT_TRUE(std::isfinite(pixels[i]));
    EXPECT_NEAR(pixels[i], 1.F, 1e-3F);
  }
}

TEST(FocusVolumeTest, TemporalImpulseHasFlatComplexSpectrum) {
  using namespace holoflow::core;
  const TDesc          input({4, 2, 2}, DType::CF32, MemLoc::Device);
  std::vector<Complex> impulse(16, {0.F, 0.F});
  for (std::size_t pixel = 0; pixel < 4; ++pixel)
    impulse[pixel].real = 1.F;
  const auto run = holonp_test::run_sync_factory(
      holotask::syncs::TemporalFFTFactory{}, std::vector<TDesc>{input},
      std::vector<std::vector<std::byte>>{as_bytes(impulse)}, nlohmann::json::object());
  ASSERT_EQ(run.output_bytes.size(), 1);
  ASSERT_EQ(run.output_bytes[0].size(), 16 * sizeof(Complex));
  const auto *values = reinterpret_cast<const Complex *>(run.output_bytes[0].data());
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_NEAR(values[i].real, 1.F, 1e-4F);
    EXPECT_NEAR(values[i].imaginary, 0.F, 1e-4F);
  }
}
