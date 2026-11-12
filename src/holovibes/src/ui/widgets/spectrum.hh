// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace holovibes::ui {

struct SpectrumSample {
  double frequency_hz;
  double power;
};

struct SpectrumPowerRange {
  double minimum;
  double maximum;
};

inline std::optional<double> spectrum_plot_value(const SpectrumSample &sample, bool log10_scale,
                                                 bool exclude_dc) {
  if (!std::isfinite(sample.power) || sample.power < 0.0 ||
      (exclude_dc && sample.frequency_hz == 0.0) || (log10_scale && sample.power == 0.0)) {
    return std::nullopt;
  }
  return log10_scale ? std::log10(sample.power) : sample.power;
}

inline SpectrumPowerRange spectrum_power_range(std::span<const SpectrumSample> samples,
                                               bool log10_scale, bool exclude_dc) {
  std::optional<double> minimum;
  double                maximum = 0.0;
  for (const auto &sample : samples) {
    if (const auto value = spectrum_plot_value(sample, log10_scale, exclude_dc)) {
      if (!minimum) {
        minimum = *value;
        maximum = *value;
      } else {
        minimum = std::min(*minimum, *value);
        maximum = std::max(maximum, *value);
      }
    }
  }
  if (!log10_scale) {
    return {0.0, maximum > 0.0 ? maximum * 1.05 : 1.0};
  }
  if (!minimum) {
    return {-1.0, 1.0};
  }
  const double margin = maximum > *minimum ? (maximum - *minimum) * 0.05 : 0.5;
  return {*minimum - margin, maximum + margin};
}

// Input power follows the transform's native order. Shift both frequencies and power for FFT;
// RFFT retains its nonnegative bins, including the positive Nyquist bin for even windows.
inline std::vector<SpectrumSample> make_spectrum(std::span<const float> power, size_t window_size,
                                                 double sampling_frequency_hz, bool full_fft) {
  if (window_size == 0 || !std::isfinite(sampling_frequency_hz) || sampling_frequency_hz <= 0.0 ||
      power.size() != (full_fft ? window_size : window_size / 2 + 1)) {
    throw std::invalid_argument("Invalid spectrum size or sampling frequency");
  }

  std::vector<SpectrumSample> samples;
  samples.reserve(power.size());
  const double df = sampling_frequency_hz / static_cast<double>(window_size);
  for (size_t i = 0; i < power.size(); ++i) {
    const size_t source = full_fft ? (i + (window_size + 1) / 2) % window_size : i;
    const double bin    = full_fft ? static_cast<double>(i) - static_cast<double>(window_size / 2)
                                   : static_cast<double>(i);
    samples.push_back({bin * df, power[source]});
  }
  return samples;
}

} // namespace holovibes::ui
