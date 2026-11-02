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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"
#include "holotask/syncs/mesi_remi.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape) {
  return TDesc(std::move(shape), DType::F32, MemLoc::Device);
}

std::vector<std::byte> as_bytes(const std::vector<float> &values) {
  std::vector<std::byte> result(values.size() * sizeof(float));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

std::vector<float> as_floats(const std::vector<std::byte> &bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

double exposure_average(int frames, double frame_period, double gate_duration, double tau,
                        double decay_multiplier) {
  const double rate   = decay_multiplier / tau;
  const double gate_z = rate * gate_duration;
  double       same_gate;
  if (std::abs(gate_z) >= 1.0e-3) {
    same_gate = 2.0 * (std::expm1(-gate_z) + gate_z) / (gate_z * gate_z);
  } else {
    const double z2 = gate_z * gate_z;
    same_gate       = 1.0 - gate_z / 3.0 + z2 / 12.0 - z2 * gate_z / 60.0 + z2 * z2 / 360.0;
  }
  if (frames == 1) {
    return same_gate;
  }
  const double gate_average = -std::expm1(-gate_z) / gate_z;
  double       integrated   = static_cast<double>(frames) * same_gate;
  for (int lag = 1; lag < frames; ++lag) {
    const double start_gap = std::max(static_cast<double>(lag) * frame_period - gate_duration, 0.0);
    const double covariance = gate_average * gate_average * std::exp(-rate * start_gap);
    integrated += 2.0 * static_cast<double>(frames - lag) * covariance;
  }
  return integrated / (static_cast<double>(frames) * frames);
}

double model_k2(const holotask::syncs::MesiRemiSettings &s, int exposure_frames,
                double tau_frames) {
  const double tau  = tau_frames * s.frame_period_seconds;
  const double gate = s.exposure_fraction * s.frame_period_seconds;
  const double rho  = s.rho;
  return s.beta *
         (rho * rho * exposure_average(exposure_frames, s.frame_period_seconds, gate, tau, 2.0) +
          2.0 * rho * (1.0 - rho) *
              exposure_average(exposure_frames, s.frame_period_seconds, gate, tau, 1.0) +
          (1.0 - rho) * (1.0 - rho));
}

holotask::syncs::MesiRemiSettings fit_settings() {
  return {.exposure_frames      = {1, 2, 4, 8},
          .frame_period_seconds = 0.01,
          .exposure_fraction    = 0.8,
          .beta                 = 0.9,
          .rho                  = 0.8,
          .tau_min_frames       = 1.0,
          .tau_max_frames       = 16.0,
          .tau_samples          = 5};
}

} // namespace

TEST(MesiRemiInferTest, ProducesTauAndParameterMaps) {
  holotask::syncs::MesiRemiFactory factory;
  const std::vector<TDesc>         input{device_desc({4, 3, 4})};
  const auto                       result = factory.infer(input, fit_settings());

  ASSERT_EQ(result.output_descs.size(), 2);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 3, 4}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(result.output_descs[1].shape, (std::vector<size_t>{3, 3, 4}));
  EXPECT_EQ(result.output_descs[1].dtype, DType::F32);
}

TEST(MesiRemiInferTest, RejectsNonDyadicOrMismatchedExposureChannels) {
  holotask::syncs::MesiRemiFactory factory;
  const std::vector<TDesc>         input{device_desc({4, 3, 4})};
  auto                             settings = fit_settings();
  settings.exposure_frames                  = {1, 2, 3, 8};
  EXPECT_THROW(factory.infer(input, settings), std::invalid_argument);

  settings                 = fit_settings();
  settings.exposure_frames = {1, 2, 4};
  EXPECT_THROW(factory.infer(input, settings), std::invalid_argument);
}

TEST(MesiRemiInferTest, RemiRequiresFiveContinuousExposureChannels) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.estimator                        = holotask::syncs::MesiEstimator::Remi;
  EXPECT_THROW(factory.infer(std::vector<TDesc>{device_desc({4, 2, 2})}, settings),
               std::invalid_argument);
  settings.exposure_frames.push_back(16);
  EXPECT_THROW(factory.infer(std::vector<TDesc>{device_desc({5, 2, 2})}, settings),
               std::invalid_argument);
  settings.exposure_fraction = 1.0;
  EXPECT_NO_THROW(factory.infer(std::vector<TDesc>{device_desc({5, 2, 2})}, settings));
}

TEST(MesiRemiExecuteTest, RecoversAnInteriorTauFromTheGatingAwareLookupTable) {
  holotask::syncs::MesiRemiFactory factory;
  const auto                       settings = fit_settings();
  std::vector<float>               input;
  for (const int exposure : settings.exposure_frames) {
    input.push_back(static_cast<float>(model_k2(settings, exposure, 4.0)));
  }

  const auto result =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{device_desc({input.size(), 1, 1})},
                                    std::vector<std::vector<std::byte>>{as_bytes(input)}, settings);
  const auto output     = as_floats(result.output_bytes[0]);
  const auto parameters = as_floats(result.output_bytes[1]);

  ASSERT_EQ(output.size(), 1);
  ASSERT_EQ(parameters.size(), 3);
  EXPECT_NEAR(output[0], 0.04F, 1.0e-6F);
  EXPECT_FLOAT_EQ(parameters[2], output[0]);
}

TEST(MesiRemiExecuteTest, MarksFlatOrNonFiniteMeasurementsUnidentifiable) {
  holotask::syncs::MesiRemiFactory factory;
  const auto                       settings = fit_settings();
  std::vector<float>               input(settings.exposure_frames.size(), 0.0F);
  input[1] = std::numeric_limits<float>::quiet_NaN();

  const auto result =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{device_desc({input.size(), 1, 1})},
                                    std::vector<std::vector<std::byte>>{as_bytes(input)}, settings);

  EXPECT_FLOAT_EQ(as_floats(result.output_bytes[0])[0], 0.0F);
}

TEST(MesiRemiExecuteTest, RemiAndSpatialRemiProcessContinuousExposureCurve) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.exposure_frames.push_back(16);
  settings.exposure_frames.push_back(32);
  settings.exposure_fraction             = 1.0;
  settings.tau_min_frames                = 0.05;
  settings.tau_max_frames                = 100.0;
  settings.beta_smoothing_size           = 3;
  constexpr int      width               = 3;
  constexpr int      height              = 3;
  constexpr double   expected_tau_frames = 4.0;
  std::vector<float> observations;
  for (const int exposure : settings.exposure_frames) {
    const float k2 = static_cast<float>(model_k2(settings, exposure, expected_tau_frames));
    observations.insert(observations.end(), width * height, k2);
  }
  for (const auto estimator :
       {holotask::syncs::MesiEstimator::Remi, holotask::syncs::MesiEstimator::SpatialRemi}) {
    settings.estimator = estimator;
    const auto result  = holonp_test::run_sync_factory(
        factory, std::vector<TDesc>{device_desc({settings.exposure_frames.size(), height, width})},
        std::vector<std::vector<std::byte>>{as_bytes(observations)}, settings);
    const auto output     = as_floats(result.output_bytes[0]);
    const auto parameters = as_floats(result.output_bytes[1]);
    ASSERT_EQ(output.size(), width * height);
    ASSERT_EQ(parameters.size(), 3 * width * height);
    for (size_t pixel = 0; pixel < output.size(); ++pixel) {
      const float tau = output[pixel];
      EXPECT_FLOAT_EQ(parameters[2 * output.size() + pixel], tau);
      EXPECT_TRUE(std::isfinite(tau));
      EXPECT_GT(tau, 0.0F);
      if (estimator == holotask::syncs::MesiEstimator::Remi) {
        EXPECT_NEAR(tau, expected_tau_frames * settings.frame_period_seconds, 0.01F);
      }
    }
  }
}

TEST(MesiRemiExecuteTest, RemiRejectsNonFiniteAndFlatCurves) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.estimator                        = holotask::syncs::MesiEstimator::Remi;
  settings.exposure_frames.push_back(16);
  settings.exposure_fraction = 1.0;
  std::vector<float> observations(settings.exposure_frames.size() * 2, 0.5F);
  observations[1]   = std::numeric_limits<float>::quiet_NaN();
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_desc({settings.exposure_frames.size(), 1, 2})},
      std::vector<std::vector<std::byte>>{as_bytes(observations)}, settings);
  const auto output = as_floats(result.output_bytes[0]);
  EXPECT_FLOAT_EQ(output[0], 0.0F);
  EXPECT_FLOAT_EQ(output[1], 0.0F);
}

TEST(MesiRemiExecuteTest, RemiAndSpatialRemiExtrapolatePastLongestExposure) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.exposure_frames.push_back(16);
  settings.exposure_fraction   = 1.0;
  settings.tau_min_frames      = 0.05;
  settings.tau_max_frames      = 100.0;
  settings.beta_smoothing_size = 1;
  std::vector<float> observations;
  for (const int exposure : settings.exposure_frames) {
    observations.push_back(static_cast<float>(model_k2(settings, exposure, 64.0)));
  }

  for (const auto estimator :
       {holotask::syncs::MesiEstimator::Remi, holotask::syncs::MesiEstimator::SpatialRemi}) {
    settings.estimator = estimator;
    const auto result  = holonp_test::run_sync_factory(
        factory, std::vector<TDesc>{device_desc({observations.size(), 1, 1})},
        std::vector<std::vector<std::byte>>{as_bytes(observations)}, settings);
    const float tau = as_floats(result.output_bytes[0])[0];
    EXPECT_TRUE(std::isfinite(tau));
    EXPECT_GT(tau, 16.0F * settings.frame_period_seconds);
    EXPECT_LE(tau, settings.tau_max_frames * settings.frame_period_seconds);
  }
}

TEST(MesiRemiExecuteTest, RemiExtrapolatesBeforeShortestExposure) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.exposure_frames.push_back(16);
  settings.exposure_fraction = 1.0;
  settings.tau_min_frames    = 0.05;
  settings.tau_max_frames    = 100.0;
  settings.estimator         = holotask::syncs::MesiEstimator::Remi;
  const std::vector<float> observations{0.23322531F, 0.06005375F, 0.02803598F, 0.00799293F,
                                        0.00215128F};

  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_desc({observations.size(), 1, 1})},
      std::vector<std::vector<std::byte>>{as_bytes(observations)}, settings);
  const float tau = as_floats(result.output_bytes[0])[0];
  EXPECT_GT(tau, settings.tau_min_frames * settings.frame_period_seconds);
  EXPECT_LT(tau, settings.frame_period_seconds);
}

TEST(MesiRemiExecuteTest, RemiUsesLastAboveThresholdForNoisyCurve) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.exposure_frames.push_back(16);
  settings.exposure_fraction = 1.0;
  settings.tau_min_frames    = 0.05;
  settings.tau_max_frames    = 100.0;
  settings.estimator         = holotask::syncs::MesiEstimator::Remi;
  // The final point rises above Kcrit after an earlier crossing. Eq. 24 uses
  // the final pair, rather than the earlier descending pair.
  const std::vector<float> observations{0.25F, 0.20F, 0.10F, 0.05F, 0.15F};

  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_desc({observations.size(), 1, 1})},
      std::vector<std::vector<std::byte>>{as_bytes(observations)}, settings);
  const float tau = as_floats(result.output_bytes[0])[0];
  EXPECT_TRUE(std::isfinite(tau));
  EXPECT_GT(tau, 8.0F * settings.frame_period_seconds);
  EXPECT_LT(tau, 16.0F * settings.frame_period_seconds);
}

TEST(MesiRemiExecuteTest, SpatialRemiSmoothsEstimatedBetaAcrossPixels) {
  holotask::syncs::MesiRemiFactory factory;
  auto                             settings = fit_settings();
  settings.estimator                        = holotask::syncs::MesiEstimator::SpatialRemi;
  settings.exposure_frames.push_back(16);
  settings.exposure_frames.push_back(32);
  settings.exposure_fraction = 1.0;
  settings.tau_min_frames    = 0.05;
  settings.tau_max_frames    = 100.0;
  std::vector<float> observations;
  for (const int exposure : settings.exposure_frames) {
    for (int pixel = 0; pixel < 9; ++pixel) {
      settings.beta = pixel == 4 ? 0.5 : 0.9;
      observations.push_back(static_cast<float>(model_k2(settings, exposure, 4.0)));
    }
  }
  const auto input = std::vector<TDesc>{device_desc({settings.exposure_frames.size(), 3, 3})};
  const auto bytes = std::vector<std::vector<std::byte>>{as_bytes(observations)};
  settings.beta_smoothing_size = 1;
  const auto unsmoothed =
      as_floats(holonp_test::run_sync_factory(factory, input, bytes, settings).output_bytes[0]);
  settings.beta_smoothing_size = 3;
  const auto smoothed =
      as_floats(holonp_test::run_sync_factory(factory, input, bytes, settings).output_bytes[0]);
  ASSERT_EQ(smoothed.size(), 9);
  EXPECT_TRUE(std::isfinite(smoothed[4]));
  EXPECT_GT(smoothed[4], 0.0F);
  EXPECT_GT(std::abs(smoothed[4] - unsmoothed[4]), 1.0e-5F);
}

TEST(MesiRemiSettingsTest, JsonRoundTripPreservesCalibrationAndSearchRange) {
  const auto           settings = fit_settings();
  const nlohmann::json json     = settings;
  const auto           restored = json.get<holotask::syncs::MesiRemiSettings>();

  EXPECT_EQ(restored, settings);
}
