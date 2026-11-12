// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "ui/widgets/spectrum.hh"

namespace holovibes::ui {
namespace {

TEST(Spectrum, EvenFftShiftsPowerAndPlacesNyquistAtNegativeFrequency) {
  const std::array<float, 8>  power{10, 11, 12, 13, 14, 15, 16, 17};
  const auto                  samples = make_spectrum(power, 8, 800.0, true);
  const std::array<double, 8> expected_power{14, 15, 16, 17, 10, 11, 12, 13};
  ASSERT_EQ(samples.size(), 8);
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_DOUBLE_EQ(samples[i].frequency_hz, -400.0 + 100.0 * i);
    EXPECT_DOUBLE_EQ(samples[i].power, expected_power[i]);
  }
}

TEST(Spectrum, OddFftShiftsPowerWithDcInTheMiddle) {
  const std::array<float, 5>  power{10, 11, 12, 13, 14};
  const auto                  samples = make_spectrum(power, 5, 500.0, true);
  const std::array<double, 5> expected_power{13, 14, 10, 11, 12};
  ASSERT_EQ(samples.size(), 5);
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_DOUBLE_EQ(samples[i].frequency_hz, -200.0 + 100.0 * i);
    EXPECT_DOUBLE_EQ(samples[i].power, expected_power[i]);
  }
}

TEST(Spectrum, RfftKeepsNonnegativeOrderAndEvenNyquist) {
  const std::array<float, 5> power{10, 11, 12, 13, 14};
  const auto                 samples = make_spectrum(power, 8, 800.0, false);
  ASSERT_EQ(samples.size(), 5);
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_DOUBLE_EQ(samples[i].frequency_hz, 100.0 * i);
    EXPECT_DOUBLE_EQ(samples[i].power, power[i]);
  }
}

TEST(Spectrum, OddRfftUsesOriginalWindowSizeForBinSpacing) {
  const std::array<float, 3> power{10, 11, 12};
  const auto                 samples = make_spectrum(power, 5, 500.0, false);
  ASSERT_EQ(samples.size(), 3);
  EXPECT_DOUBLE_EQ(samples.back().frequency_hz, 200.0);
  EXPECT_DOUBLE_EQ(samples.back().power, 12.0);
}

TEST(Spectrum, SingleSampleHasOnlyDc) {
  const std::array<float, 1> power{42};
  for (bool full_fft : {false, true}) {
    const auto samples = make_spectrum(power, 1, 100.0, full_fft);
    ASSERT_EQ(samples.size(), 1);
    EXPECT_DOUBLE_EQ(samples[0].frequency_hz, 0.0);
    EXPECT_DOUBLE_EQ(samples[0].power, 42.0);
  }
}

TEST(Spectrum, RejectsInvalidTransformMetadata) {
  const std::array<float, 3> power{10, 11, 12};
  EXPECT_THROW(make_spectrum(power, 0, 500.0, false), std::invalid_argument);
  EXPECT_THROW(make_spectrum(power, 5, 0.0, false), std::invalid_argument);
  EXPECT_THROW(make_spectrum(power, 5, -1.0, false), std::invalid_argument);
  EXPECT_THROW(make_spectrum(power, 5, std::numeric_limits<double>::infinity(), false),
               std::invalid_argument);
  EXPECT_THROW(make_spectrum(power, 5, std::numeric_limits<double>::quiet_NaN(), false),
               std::invalid_argument);
  EXPECT_THROW(make_spectrum(power, 5, 500.0, true), std::invalid_argument);
}

} // namespace
} // namespace holovibes::ui
