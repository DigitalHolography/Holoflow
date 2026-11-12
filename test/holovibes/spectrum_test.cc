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

TEST(Spectrum, Log10PowerIncludesPowersBelowOneAndOmitsZero) {
  EXPECT_DOUBLE_EQ(*spectrum_plot_value({100.0, 1000.0}, true, false), 3.0);
  EXPECT_DOUBLE_EQ(*spectrum_plot_value({100.0, 0.001}, true, false), -3.0);
  EXPECT_FALSE(spectrum_plot_value({100.0, 0.0}, true, false));
  EXPECT_DOUBLE_EQ(*spectrum_plot_value({100.0, 0.0}, false, false), 0.0);
  EXPECT_FALSE(spectrum_plot_value({100.0, -1.0}, true, false));
  EXPECT_FALSE(spectrum_plot_value({100.0, std::numeric_limits<double>::infinity()}, true, false));
}

TEST(Spectrum, ExcludingDcRescalesBothFftAndRfftByFrequency) {
  const std::array<float, 4> power{1e12f, 1, 100, 10};
  for (bool full_fft : {false, true}) {
    const auto samples    = make_spectrum(power, full_fft ? 4 : 6, 600.0, full_fft);
    const auto full_range = spectrum_power_range(samples, false, false);
    const auto without_dc = spectrum_power_range(samples, false, true);
    EXPECT_GT(full_range.maximum, 1e12);
    EXPECT_DOUBLE_EQ(without_dc.minimum, 0.0);
    EXPECT_DOUBLE_EQ(without_dc.maximum, 105.0);
    for (const auto &sample : samples) {
      EXPECT_EQ(spectrum_plot_value(sample, false, true).has_value(), sample.frequency_hz != 0.0);
    }
  }
}

TEST(Spectrum, Log10RangeFitsRemainingPositivePowerWhenDcIsExcluded) {
  const std::array<SpectrumSample, 4> samples{
      {{-100.0, 1e-4}, {0.0, 1e12}, {100.0, 1e2}, {200.0, 0.0}}};
  const auto range = spectrum_power_range(samples, true, true);
  EXPECT_DOUBLE_EQ(range.minimum, -4.3);
  EXPECT_DOUBLE_EQ(range.maximum, 2.3);
}

TEST(Spectrum, ConstantLog10PowerHasNonzeroRange) {
  const std::array<SpectrumSample, 2> samples{{{-100.0, 0.01}, {100.0, 0.01}}};
  const auto                          range = spectrum_power_range(samples, true, false);
  EXPECT_DOUBLE_EQ(range.minimum, -2.5);
  EXPECT_DOUBLE_EQ(range.maximum, -1.5);
}

TEST(Spectrum, EmptyOrExcludedDataKeepsValidLimits) {
  const std::array<SpectrumSample, 1> dc{{{0.0, 42.0}}};
  const std::array<SpectrumSample, 2> zero_or_invalid{
      {{100.0, 0.0}, {200.0, std::numeric_limits<double>::quiet_NaN()}}};
  for (bool log10_scale : {false, true}) {
    for (auto samples : {std::span<const SpectrumSample>{}, std::span<const SpectrumSample>{dc},
                         std::span<const SpectrumSample>{zero_or_invalid}}) {
      const auto range = spectrum_power_range(samples, log10_scale, true);
      EXPECT_TRUE(std::isfinite(range.minimum));
      EXPECT_TRUE(std::isfinite(range.maximum));
      EXPECT_LT(range.minimum, range.maximum);
    }
  }
}

} // namespace
} // namespace holovibes::ui
