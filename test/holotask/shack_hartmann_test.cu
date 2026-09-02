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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cuComplex.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/cross_correlation2.hh"
#include "holotask/syncs/shack_hartmann_slopes.hh"
#include "holotask/syncs/zernike_from_slopes.hh"

#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

constexpr size_t kSy = 5;
constexpr size_t kSx = 5;

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T> std::vector<T> from_bytes(const std::vector<std::byte> &bytes) {
  EXPECT_EQ(bytes.size() % sizeof(T), 0);
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

TDesc desc(std::vector<size_t> shape, MemLoc location) {
  return TDesc(std::move(shape), DType::F32, location);
}

float pattern(float x, float y, float width, float height) {
  const auto gaussian = [](float px, float py, float cx, float cy, float sigma, float amplitude) {
    const float dx = px - cx;
    const float dy = py - cy;
    return amplitude * std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
  };

  return gaussian(x, y, 0.43f * width, 0.48f * height, 3.4f, 1.0f) +
         gaussian(x, y, 0.62f * width, 0.36f * height, 2.1f, 0.55f) +
         gaussian(x, y, 0.31f * width, 0.66f * height, 1.7f, 0.27f);
}

std::vector<float> translated_subapertures(size_t sy_count, size_t sx_count, size_t height,
                                           size_t                                      width,
                                           const std::vector<std::pair<float, float>> &shifts) {
  std::vector<float> images(sy_count * sx_count * height * width);
  for (size_t sample = 0; sample < shifts.size(); ++sample) {
    const auto [dx, dy] = shifts[sample];
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        images[(sample * height + y) * width + x] =
            pattern(static_cast<float>(x) - dx, static_cast<float>(y) - dy,
                    static_cast<float>(width), static_cast<float>(height));
      }
    }
  }
  return images;
}

holotask::syncs::ShackHartmannSlopeSettings
slope_settings(size_t height, size_t width, bool output_maps, bool skip_outside = false) {
  return {
      .mode               = holotask::syncs::ShackHartmannSlopeMode::SingleReference,
      .lambda             = static_cast<float>(width),
      .dx                 = 1.0f,
      .dy                 = static_cast<float>(width) / static_cast<float>(height),
      .z                  = 2.0f,
      .subaperture_height = height,
      .subaperture_width  = width,
      .stride_y           = height,
      .stride_x           = width,
      .correlation_roi    = {.cx = 0.5f, .cy = 0.5f, .rx = 10.0f, .ry = 10.0f, .angle = 0.0f},
      .skip_subapertures_outside_pupil = skip_outside,
      .output_xcorr_maps               = output_maps,
  };
}

std::vector<float> run_slopes(const std::vector<float> &images, size_t sy, size_t sx, size_t height,
                              size_t                                             width,
                              const holotask::syncs::ShackHartmannSlopeSettings &settings) {
  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto                                  run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc({1, sy, sx, height, width}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(images)}, settings);
  return from_bytes<float>(run.output_bytes[0]);
}

std::vector<float> complete_graph_recovery(const std::vector<float> &values,
                                           const std::vector<float> *edge_noise = nullptr) {
  const size_t       count = values.size();
  std::vector<float> rhs(count, 0.0f);
  size_t             edge = 0;
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = i + 1; j < count; ++j) {
      const float measurement =
          values[i] - values[j] + (edge_noise == nullptr ? 0.0f : (*edge_noise)[edge]);
      rhs[i] += measurement;
      rhs[j] -= measurement;
      ++edge;
    }
  }
  for (float &value : rhs) {
    value /= static_cast<float>(count);
  }
  const float mean = std::accumulate(rhs.begin(), rhs.end(), 0.0f) / static_cast<float>(count);
  for (float &value : rhs) {
    value -= mean;
  }
  return rhs;
}

std::vector<float> star_graph_recovery(const std::vector<float> &values, size_t center,
                                       const std::vector<float> *edge_noise = nullptr) {
  std::vector<float> recovered(values.size(), 0.0f);
  size_t             edge = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != center) {
      recovered[i] =
          values[i] - values[center] + (edge_noise == nullptr ? 0.0f : (*edge_noise)[edge++]);
    }
  }
  const float mean = std::accumulate(recovered.begin(), recovered.end(), 0.0f) /
                     static_cast<float>(recovered.size());
  for (float &value : recovered) {
    value -= mean;
  }
  return recovered;
}

float rmse(const std::vector<float> &actual, const std::vector<float> &expected) {
  float squared_error = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = actual[i] - expected[i];
    squared_error += error * error;
  }
  return std::sqrt(squared_error / static_cast<float>(actual.size()));
}

float time_slope_execution(const std::vector<float> &images, size_t sy, size_t sx, size_t height,
                           size_t                                             width,
                           const holotask::syncs::ShackHartmannSlopeSettings &settings,
                           size_t                                             iterations) {
  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto         input_desc = desc({1, sy, sx, height, width}, MemLoc::Device);
  const auto         inference  = factory.infer(std::vector<TDesc>{input_desc}, settings);
  curaii::CudaStream stream;
  auto task = factory.create(std::vector<TDesc>{input_desc}, settings, {.stream = stream.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(inference.output_descs[0]);
  input.upload(as_bytes(images));
  std::array              input_views{input.view()};
  std::array              output_views{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx context{
      .inputs       = input_views,
      .outputs      = output_views,
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  EXPECT_EQ(task->execute(context), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream.get()));
  for (size_t i = 0; i < iterations; ++i) {
    EXPECT_EQ(task->execute(context), holoflow::core::OpResult::Ok);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream.get()));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / static_cast<float>(iterations);
}

std::pair<float, float> correlation_peak(const float *map, size_t height, size_t width) {
  size_t peak_y = 0;
  size_t peak_x = 0;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      if (map[y * width + x] > map[peak_y * width + peak_x]) {
        peak_y = y;
        peak_x = x;
      }
    }
  }

  const auto parabolic = [](float minus, float zero, float plus) {
    const float denominator = minus - 2.0f * zero + plus;
    return std::abs(denominator) <= 1e-9f ? 0.0f : 0.5f * (minus - plus) / denominator;
  };
  const size_t xm = (peak_x + width - 1) % width;
  const size_t xp = (peak_x + 1) % width;
  const size_t ym = (peak_y + height - 1) % height;
  const size_t yp = (peak_y + 1) % height;
  const float  dx_sub =
      parabolic(map[peak_y * width + xm], map[peak_y * width + peak_x], map[peak_y * width + xp]);
  const float dy_sub =
      parabolic(map[ym * width + peak_x], map[peak_y * width + peak_x], map[yp * width + peak_x]);
  const auto signed_coordinate = [](size_t peak, size_t size) {
    return peak < (size + 1) / 2 ? static_cast<float>(peak)
                                 : static_cast<float>(peak) - static_cast<float>(size);
  };
  return {signed_coordinate(peak_x, width) + dx_sub, signed_coordinate(peak_y, height) + dy_sub};
}

struct Derivative {
  float x;
  float y;
};

Derivative derivative(int index, float x, float y) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);
  const float sqrt8 = std::sqrt(8.0f);
  switch (index) {
  case 2:
    return {2.0f, 0.0f};
  case 3:
    return {0.0f, 2.0f};
  case 4:
    return {4.0f * sqrt3 * x, 4.0f * sqrt3 * y};
  case 5:
    return {2.0f * sqrt6 * y, 2.0f * sqrt6 * x};
  case 6:
    return {2.0f * sqrt6 * x, -2.0f * sqrt6 * y};
  case 7:
    return {6.0f * sqrt8 * x * y, 3.0f * sqrt8 * (x * x - y * y)};
  case 8:
    return {6.0f * sqrt8 * x * y, sqrt8 * (3.0f * x * x + 9.0f * y * y - 2.0f)};
  case 9:
    return {sqrt8 * (9.0f * x * x + 3.0f * y * y - 2.0f), 6.0f * sqrt8 * x * y};
  case 10:
    return {3.0f * sqrt8 * (x * x - y * y), -6.0f * sqrt8 * x * y};
  default:
    return {0.0f, 0.0f};
  }
}

float zernike_value(int index, float x, float y) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);
  const float sqrt8 = std::sqrt(8.0f);
  switch (index) {
  case 4:
    return sqrt3 * (2.0f * (x * x + y * y) - 1.0f);
  case 5:
    return 2.0f * sqrt6 * x * y;
  case 6:
    return sqrt6 * (x * x - y * y);
  case 7:
    return sqrt8 * y * (3.0f * x * x - y * y);
  case 8:
    return sqrt8 * y * (3.0f * x * x + 3.0f * y * y - 2.0f);
  case 9:
    return sqrt8 * x * (3.0f * x * x + 3.0f * y * y - 2.0f);
  case 10:
    return sqrt8 * x * (x * x - 3.0f * y * y);
  default:
    return 0.0f;
  }
}

bool active_sample(size_t sy, size_t sx) {
  const float x = (static_cast<float>(sx) - 2.0f) / 2.5f;
  const float y = (static_cast<float>(sy) - 2.0f) / 2.5f;
  return x * x + y * y <= 1.0f;
}

std::vector<float> ideal_slopes(int injected_mode, float coefficient_m) {
  std::vector<float>      slopes(kSy * kSx * 2, 0.0f);
  std::vector<Derivative> values(kSy * kSx);
  Derivative              mean{0.0f, 0.0f};
  size_t                  count = 0;
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      if (!active_sample(sy, sx)) {
        continue;
      }
      const size_t i = sy * kSx + sx;
      values[i]      = derivative(injected_mode, (static_cast<float>(sx) - 2.0f) / 2.5f,
                                  (static_cast<float>(sy) - 2.0f) / 2.5f);
      values[i].x /= 25.0f;
      values[i].y /= 25.0f;
      mean.x += values[i].x;
      mean.y += values[i].y;
      ++count;
    }
  }
  mean.x /= static_cast<float>(count);
  mean.y /= static_cast<float>(count);
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      if (!active_sample(sy, sx)) {
        continue;
      }
      const size_t i    = sy * kSx + sx;
      slopes[2 * i]     = coefficient_m * (values[i].x - mean.x);
      slopes[2 * i + 1] = coefficient_m * (values[i].y - mean.y);
    }
  }
  return slopes;
}

holotask::syncs::ZernikeFromSlopesSettings fit_settings() {
  return {
      .indexes                         = {2, 3, 4, 5, 6, 7, 8, 9, 10},
      .lambda                          = 0.5f,
      .dx                              = 1.0f,
      .dy                              = 1.0f,
      .subaperture_height              = 10,
      .subaperture_width               = 10,
      .stride_y                        = 10,
      .stride_x                        = 10,
      .ny                              = 1,
      .nx                              = 1,
      .skip_subapertures_outside_pupil = true,
  };
}

std::vector<float>
run_fit(const std::vector<float> &slopes, MemLoc mem_loc = MemLoc::Device,
        const holotask::syncs::ZernikeFromSlopesSettings &settings = fit_settings()) {
  holotask::syncs::ZernikeFromSlopesFactory factory;
  const auto                                run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc({1, kSy, kSx, 2}, mem_loc)},
      std::vector<std::vector<std::byte>>{as_bytes(slopes)}, settings);
  return from_bytes<float>(run.output_bytes[0]);
}

std::array<float, 7> legacy_center_fit(const std::vector<float> &maps, size_t height, size_t width,
                                       float lambda) {
  std::array<std::array<double, 7>, 7> gtg{};
  std::array<double, 7>                gts{};
  const size_t                         center_index = (kSy / 2) * kSx + kSx / 2;
  const auto                           center_shift =
      correlation_peak(maps.data() + center_index * height * width, height, width);
  const float radius = 0.5f * static_cast<float>(kSx * width);

  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      if (!active_sample(sy, sx)) {
        continue;
      }

      const size_t sample_index = sy * kSx + sx;
      const auto   shift =
          correlation_peak(maps.data() + sample_index * height * width, height, width);
      const double slope_x = static_cast<double>(shift.first - center_shift.first);
      const double slope_y = static_cast<double>(shift.second - center_shift.second);
      const float  xn      = (static_cast<float>(sx) - 2.0f) / 2.5f;
      const float  yn      = (static_cast<float>(sy) - 2.0f) / 2.5f;

      std::array<double, 7> gx{};
      std::array<double, 7> gy{};
      for (int mode = 4; mode <= 10; ++mode) {
        const auto   value  = derivative(mode, xn, yn);
        const auto   center = derivative(mode, 0.0f, 0.0f);
        const size_t i      = static_cast<size_t>(mode - 4);
        gx[i]               = static_cast<double>((value.x - center.x) / radius);
        gy[i]               = static_cast<double>((value.y - center.y) / radius);
      }

      for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 7; ++j) {
          gtg[i][j] += gx[i] * gx[j] + gy[i] * gy[j];
        }
        gts[i] += gx[i] * slope_x + gy[i] * slope_y;
      }
    }
  }

  for (size_t column = 0; column < 7; ++column) {
    size_t pivot = column;
    for (size_t row = column + 1; row < 7; ++row) {
      if (std::abs(gtg[row][column]) > std::abs(gtg[pivot][column])) {
        pivot = row;
      }
    }
    std::swap(gtg[pivot], gtg[column]);
    std::swap(gts[pivot], gts[column]);
    const double divisor = gtg[column][column];
    for (size_t j = column; j < 7; ++j) {
      gtg[column][j] /= divisor;
    }
    gts[column] /= divisor;
    for (size_t row = column + 1; row < 7; ++row) {
      const double factor = gtg[row][column];
      for (size_t j = column; j < 7; ++j) {
        gtg[row][j] -= factor * gtg[column][j];
      }
      gts[row] -= factor * gts[column];
    }
  }

  std::array<float, 7> result{};
  for (int row = 6; row >= 0; --row) {
    double value = gts[static_cast<size_t>(row)];
    for (size_t column = static_cast<size_t>(row) + 1; column < 7; ++column) {
      value -= gtg[static_cast<size_t>(row)][column] * result[column];
    }
    result[static_cast<size_t>(row)] = static_cast<float>(value);
  }
  for (float &value : result) {
    value *= 2.0f * std::acos(-1.0f) / lambda;
  }
  return result;
}

} // namespace

TEST(ShackHartmannGraphRecovery, ExactRecoveryGaugeInvarianceAndStarEquality) {
  const std::vector<float> gx{-1.4f, 0.2f, 2.1f, -0.7f, 0.9f, -1.1f};
  const std::vector<float> gy{0.3f, -2.0f, 1.2f, 0.6f, -0.8f, 0.7f};
  const auto               recovered_x = complete_graph_recovery(gx);
  const auto               recovered_y = complete_graph_recovery(gy);
  const auto               star_x      = star_graph_recovery(gx, 2);
  const auto               star_y      = star_graph_recovery(gy, 2);

  for (size_t i = 0; i < gx.size(); ++i) {
    EXPECT_NEAR(recovered_x[i], gx[i], 3e-7f);
    EXPECT_NEAR(recovered_y[i], gy[i], 3e-7f);
    EXPECT_NEAR(star_x[i], recovered_x[i], 3e-7f);
    EXPECT_NEAR(star_y[i], recovered_y[i], 3e-7f);
  }

  auto shifted_x = gx;
  auto shifted_y = gy;
  for (float &value : shifted_x) {
    value += 37.25f;
  }
  for (float &value : shifted_y) {
    value -= 91.5f;
  }
  const auto invariant_x = complete_graph_recovery(shifted_x);
  const auto invariant_y = complete_graph_recovery(shifted_y);
  for (size_t i = 0; i < gx.size(); ++i) {
    EXPECT_NEAR(invariant_x[i], recovered_x[i], 6e-6f);
    EXPECT_NEAR(invariant_y[i], recovered_y[i], 6e-6f);
  }
}

TEST(ShackHartmannGraphRecovery, CompleteGraphReducesAggregateIndependentEdgeNoise) {
  constexpr size_t   count  = 13;
  constexpr size_t   trials = 600;
  const size_t       edges  = count * (count - 1) / 2;
  std::vector<float> truth(count);
  for (size_t i = 0; i < count; ++i) {
    truth[i] = std::sin(0.7f * static_cast<float>(i));
  }
  const float mean = std::accumulate(truth.begin(), truth.end(), 0.0f) / static_cast<float>(count);
  for (float &value : truth) {
    value -= mean;
  }

  std::mt19937                    generator(0x5A17u);
  std::normal_distribution<float> noise(0.0f, 0.15f);
  double                          star_squared_rmse_sum     = 0.0;
  double                          complete_squared_rmse_sum = 0.0;
  for (size_t trial = 0; trial < trials; ++trial) {
    std::vector<float> star_noise(count - 1);
    std::vector<float> complete_noise(edges);
    std::ranges::generate(star_noise, [&] { return noise(generator); });
    std::ranges::generate(complete_noise, [&] { return noise(generator); });
    const float star_error     = rmse(star_graph_recovery(truth, count / 2, &star_noise), truth);
    const float complete_error = rmse(complete_graph_recovery(truth, &complete_noise), truth);
    star_squared_rmse_sum += static_cast<double>(star_error) * star_error;
    complete_squared_rmse_sum += static_cast<double>(complete_error) * complete_error;
  }

  const double star_aggregate     = std::sqrt(star_squared_rmse_sum / trials);
  const double complete_aggregate = std::sqrt(complete_squared_rmse_sum / trials);
  EXPECT_LT(complete_aggregate, star_aggregate * 0.5);
  std::cout << "Independent-edge noise aggregate RMSE: star=" << star_aggregate
            << ", complete=" << complete_aggregate
            << ", ratio=" << complete_aggregate / star_aggregate << '\n';
}

TEST(ShackHartmannSlopesInference, OptionalOutputsAndModesAreExplicit) {
  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto                                  input = desc({1, 3, 5, 32, 40}, MemLoc::Device);

  auto settings = slope_settings(32, 40, false);
  auto result   = factory.infer(std::vector<TDesc>{input}, settings);
  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 3, 5, 2}));
  EXPECT_EQ(result.output_descs[0].mem_loc, MemLoc::Device);

  settings.output_xcorr_maps = true;
  result                     = factory.infer(std::vector<TDesc>{input}, settings);
  ASSERT_EQ(result.output_descs.size(), 2);
  EXPECT_EQ(result.output_descs[1].shape, input.shape);

  const nlohmann::json serialized = settings;
  EXPECT_EQ(serialized.get<holotask::syncs::ShackHartmannSlopeSettings>(), settings);

  settings.mode              = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
  settings.output_xcorr_maps = false;
  result                     = factory.infer(std::vector<TDesc>{input}, settings);
  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 3, 5, 2}));

  settings.output_xcorr_maps = true;
  EXPECT_THROW((void)factory.infer(std::vector<TDesc>{input}, settings), std::invalid_argument);

  settings.output_xcorr_maps = false;
  settings.pair_batch_size   = 0;
  EXPECT_THROW((void)factory.infer(std::vector<TDesc>{input}, settings), std::invalid_argument);

  settings.pair_batch_size = 1;
  EXPECT_NO_THROW(
      (void)factory.infer(std::vector<TDesc>{desc({1, 2, 2, 32, 40}, MemLoc::Device)}, settings));
  EXPECT_THROW(
      (void)factory.infer(std::vector<TDesc>{desc({1, 1, 1, 32, 40}, MemLoc::Device)}, settings),
      std::invalid_argument);
}

TEST(ShackHartmannSlopesExecution, RecoversSignedIntegerTranslationsAndRawMaps) {
  constexpr size_t                           height = 32;
  constexpr size_t                           width  = 32;
  const std::vector<std::pair<float, float>> shifts{
      {-2.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}, {-1.0f, 1.0f}, {0.0f, 0.0f},
      {1.0f, -1.0f}, {0.0f, 1.0f},  {1.0f, 0.0f},  {2.0f, -1.0f},
  };
  const auto images = translated_subapertures(3, 3, height, width, shifts);
  const auto input  = desc({1, 3, 3, height, width}, MemLoc::Device);

  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto                                  settings = slope_settings(height, width, true);
  const auto                                  run      = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input}, std::vector<std::vector<std::byte>>{as_bytes(images)},
      settings);
  const auto slopes = from_bytes<float>(run.output_bytes[0]);
  const auto maps   = from_bytes<float>(run.output_bytes[1]);

  for (size_t i = 0; i < shifts.size(); ++i) {
    EXPECT_NEAR(slopes[2 * i], shifts[i].first, 0.03f) << "sample " << i;
    EXPECT_NEAR(slopes[2 * i + 1], shifts[i].second, 0.03f) << "sample " << i;
  }

  const auto center_peak = correlation_peak(maps.data() + 4 * height * width, height, width);
  EXPECT_NEAR(center_peak.first, 0.0f, 1e-6f);
  EXPECT_NEAR(center_peak.second, 0.0f, 1e-6f);

  std::vector<float>                               reference(images.begin() + 4 * height * width,
                                                             images.begin() + 5 * height * width);
  holotask::syncs::CrossCorrelation2Factory        xcorr_factory;
  const holotask::syncs::CrossCorrelation2Settings xcorr_settings{
      .axes = {-2, -1},
      .norm = holotask::syncs::FftNorm::Backward,
      .roi  = settings.correlation_roi,
  };
  const auto direct = holonp_test::run_sync_factory(
      xcorr_factory, std::vector<TDesc>{input, desc({1, height, width}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(images), as_bytes(reference)}, xcorr_settings);
  const auto direct_maps = from_bytes<float>(direct.output_bytes[0]);
  ASSERT_EQ(maps.size(), direct_maps.size());
  for (size_t i = 0; i < maps.size(); ++i) {
    EXPECT_NEAR(maps[i], direct_maps[i], 2e-6f) << "map element " << i;
  }

  const auto update_run = holonp_test::run_sync_factory_update(
      factory, std::vector<TDesc>{input}, std::vector<std::vector<std::byte>>{as_bytes(images)},
      slope_settings(height, width, false));
  const auto updated_slopes = from_bytes<float>(update_run.output_bytes[0]);
  ASSERT_EQ(updated_slopes.size(), slopes.size());
  for (size_t i = 0; i < slopes.size(); ++i) {
    EXPECT_NEAR(updated_slopes[i], slopes[i], 1e-6f);
  }
}

TEST(ShackHartmannSlopesExecution, PreservesSubpixelRefinementAndActiveGauge) {
  constexpr size_t                     height = 48;
  constexpr size_t                     width  = 48;
  std::vector<std::pair<float, float>> shifts(kSy * kSx, {0.0f, 0.0f});
  shifts[6]  = {0.35f, -0.42f};
  shifts[18] = {-0.35f, 0.42f};
  shifts[2]  = {-0.60f, -0.25f};
  shifts[22] = {0.60f, 0.25f};

  const auto images = translated_subapertures(kSy, kSx, height, width, shifts);
  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto                                  run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc({1, kSy, kSx, height, width}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(images)},
      slope_settings(height, width, false, true));
  const auto slopes = from_bytes<float>(run.output_bytes[0]);

  float sum_x = 0.0f;
  float sum_y = 0.0f;
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      const size_t i = sy * kSx + sx;
      if (active_sample(sy, sx)) {
        sum_x += slopes[2 * i];
        sum_y += slopes[2 * i + 1];
      } else {
        EXPECT_EQ(slopes[2 * i], 0.0f);
        EXPECT_EQ(slopes[2 * i + 1], 0.0f);
      }
    }
  }
  EXPECT_NEAR(sum_x, 0.0f, 2e-6f);
  EXPECT_NEAR(sum_y, 0.0f, 2e-6f);
  EXPECT_NEAR(slopes[12], 0.35f, 0.13f);
  EXPECT_NEAR(slopes[13], -0.42f, 0.13f);
  EXPECT_NEAR(slopes[36], -0.35f, 0.13f);
  EXPECT_NEAR(slopes[37], 0.42f, 0.13f);
}

TEST(ShackHartmannSlopesFullPairwise, MatchesStarGraphSignsGaugeAndBatchSizes) {
  constexpr size_t                           height = 40;
  constexpr size_t                           width  = 40;
  const std::vector<std::pair<float, float>> shifts{
      {-2.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}, {-1.0f, 1.0f}, {0.0f, 0.0f},
      {1.0f, -1.0f}, {0.0f, 1.0f},  {1.0f, 0.0f},  {2.0f, -1.0f},
  };
  const auto images = translated_subapertures(3, 3, height, width, shifts);

  const auto single = run_slopes(images, 3, 3, height, width, slope_settings(height, width, false));
  const size_t       edge_count            = shifts.size() * (shifts.size() - 1) / 2;
  float              max_translation_error = 0.0f;
  float              max_mode_difference   = 0.0f;
  std::vector<float> batch_baseline;
  for (const size_t batch_size : {size_t{1}, size_t{5}, size_t{32}, edge_count + 11}) {
    auto settings            = slope_settings(height, width, false);
    settings.mode            = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
    settings.pair_batch_size = batch_size;
    const auto full          = run_slopes(images, 3, 3, height, width, settings);
    ASSERT_EQ(full.size(), single.size());
    if (batch_baseline.empty()) {
      batch_baseline = full;
    } else {
      for (size_t i = 0; i < full.size(); ++i) {
        EXPECT_NEAR(full[i], batch_baseline[i], 2e-6f)
            << "batch-invariant element " << i << ", batch " << batch_size;
      }
    }
    for (size_t i = 0; i < shifts.size(); ++i) {
      EXPECT_NEAR(full[2 * i], shifts[i].first, 0.035f)
          << "x, sample " << i << ", batch " << batch_size;
      EXPECT_NEAR(full[2 * i + 1], shifts[i].second, 0.035f)
          << "y, sample " << i << ", batch " << batch_size;
      EXPECT_NEAR(full[2 * i], single[2 * i], 0.025f)
          << "x star/complete, sample " << i << ", batch " << batch_size;
      EXPECT_NEAR(full[2 * i + 1], single[2 * i + 1], 0.025f)
          << "y star/complete, sample " << i << ", batch " << batch_size;
      max_translation_error =
          std::max({max_translation_error, std::abs(full[2 * i] - shifts[i].first),
                    std::abs(full[2 * i + 1] - shifts[i].second)});
      max_mode_difference = std::max({max_mode_difference, std::abs(full[2 * i] - single[2 * i]),
                                      std::abs(full[2 * i + 1] - single[2 * i + 1])});
    }
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (size_t i = 0; i < shifts.size(); ++i) {
      sum_x += full[2 * i];
      sum_y += full[2 * i + 1];
    }
    EXPECT_NEAR(sum_x, 0.0f, 4e-6f);
    EXPECT_NEAR(sum_y, 0.0f, 4e-6f);
  }
  std::cout << "Integer registration maximum error=" << max_translation_error
            << " px; maximum star/complete difference=" << max_mode_difference << " px\n";
}

TEST(ShackHartmannSlopesFullPairwise, PreservesSubpixelAndPeriodicPeakConventions) {
  constexpr size_t                     height = 48;
  constexpr size_t                     width  = 48;
  std::vector<std::pair<float, float>> shifts{
      {-9.0f, 7.0f},
      {-0.40f, 0.35f},
      {9.0f, -7.0f},
  };
  const auto images        = translated_subapertures(1, 3, height, width, shifts);
  auto       settings      = slope_settings(height, width, false);
  settings.mode            = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
  settings.pair_batch_size = 2;
  const auto slopes        = run_slopes(images, 1, 3, height, width, settings);

  // All three translations are inside the unambiguous periodic interval, including the 18-pixel
  // endpoint-to-endpoint edge. Compare against the public zero-mean gauge.
  const float mean_x = (shifts[0].first + shifts[1].first + shifts[2].first) / 3.0f;
  const float mean_y = (shifts[0].second + shifts[1].second + shifts[2].second) / 3.0f;
  EXPECT_NEAR(slopes[0], shifts[0].first - mean_x, 0.08f);
  EXPECT_NEAR(slopes[1], shifts[0].second - mean_y, 0.08f);
  EXPECT_NEAR(slopes[2], shifts[1].first - mean_x, 0.08f);
  EXPECT_NEAR(slopes[3], shifts[1].second - mean_y, 0.08f);
  EXPECT_NEAR(slopes[4], shifts[2].first - mean_x, 0.08f);
  EXPECT_NEAR(slopes[5], shifts[2].second - mean_y, 0.08f);
  EXPECT_NEAR(slopes[0] - slopes[4], -18.0f, 0.2f);
  EXPECT_NEAR(slopes[1] - slopes[5], 14.0f, 0.2f);
}

TEST(ZernikeFromSlopesExecution, RecoversEachObservableModeWithoutTipTiltOrLeakage) {
  constexpr float coefficient_m = 0.7f;
  const float expected_radians  = coefficient_m * 2.0f * std::acos(-1.0f) / fit_settings().lambda;

  for (int injected = 4; injected <= 10; ++injected) {
    const auto coefficients = run_fit(ideal_slopes(injected, coefficient_m));
    ASSERT_EQ(coefficients.size(), 9);
    EXPECT_EQ(coefficients[0], 0.0f);
    EXPECT_EQ(coefficients[1], 0.0f);
    for (int mode = 4; mode <= 10; ++mode) {
      if (mode == injected) {
        EXPECT_NEAR(coefficients[static_cast<size_t>(mode - 2)], expected_radians, 2e-5f)
            << "injected mode " << injected;
      } else {
        EXPECT_NEAR(coefficients[static_cast<size_t>(mode - 2)], 0.0f, 2e-5f)
            << "leakage from mode " << injected << " to mode " << mode;
      }
    }
  }
}

TEST(ZernikeFromSlopesRegression, ObservableCoefficientsAndPhaseMatchCenterGauge) {
  const std::array<float, 7> injected_m{0.18f, -0.11f, 0.07f, 0.04f, -0.03f, 0.05f, -0.02f};
  std::vector<float>         combined(kSy * kSx * 2, 0.0f);
  for (int mode = 4; mode <= 10; ++mode) {
    const auto component = ideal_slopes(mode, injected_m[static_cast<size_t>(mode - 4)]);
    for (size_t i = 0; i < combined.size(); ++i) {
      combined[i] += component[i];
    }
  }

  const auto coefficients = run_fit(combined);
  ASSERT_EQ(coefficients.size(), 9);
  EXPECT_EQ(coefficients[0], 0.0f);
  EXPECT_EQ(coefficients[1], 0.0f);

  std::array<float, 7> legacy_center_gauge{};
  const float          radians_per_meter = 2.0f * std::acos(-1.0f) / fit_settings().lambda;
  for (size_t mode = 0; mode < injected_m.size(); ++mode) {
    // On the same ideal derivative data, the corrected center-reference fit recovers the injected
    // observable coefficients. This is its analytic result and avoids retaining the removed task.
    legacy_center_gauge[mode] = injected_m[mode] * radians_per_meter;
    EXPECT_NEAR(coefficients[mode + 2], legacy_center_gauge[mode], 5e-5f) << "A" << mode + 4;
  }

  float max_phase_difference = 0.0f;
  for (size_t y = 0; y < 65; ++y) {
    for (size_t x = 0; x < 65; ++x) {
      const float xn        = (static_cast<float>(x) - 32.0f) / 32.0f;
      const float yn        = (static_cast<float>(y) - 32.0f) / 32.0f;
      float       old_phase = 0.0f;
      float       new_phase = 0.0f;
      for (int mode = 4; mode <= 10; ++mode) {
        const float value = zernike_value(mode, xn, yn);
        old_phase += legacy_center_gauge[static_cast<size_t>(mode - 4)] * value;
        new_phase += coefficients[static_cast<size_t>(mode - 2)] * value;
      }
      max_phase_difference = std::max(max_phase_difference, std::abs(new_phase - old_phase));
    }
  }
  EXPECT_LT(max_phase_difference, 2e-4f);
}

TEST(ZernikeFromSlopesExecution, DeviceAndHostBackendsMatchForMixedModes) {
  auto settings    = fit_settings();
  settings.indexes = {10, 2, 4, 7};
  const std::array<std::pair<int, float>, 3> injected{{{10, -0.02f}, {4, 0.18f}, {7, 0.04f}}};
  std::vector<float>                         combined(kSy * kSx * 2, 0.0f);
  for (const auto &[mode, coefficient] : injected) {
    const auto component = ideal_slopes(mode, coefficient);
    for (size_t i = 0; i < combined.size(); ++i) {
      combined[i] += component[i];
    }
  }

  const auto host_coefficients   = run_fit(combined, MemLoc::Host, settings);
  const auto device_coefficients = run_fit(combined, MemLoc::Device, settings);
  ASSERT_EQ(device_coefficients.size(), host_coefficients.size());
  for (size_t mode = 0; mode < host_coefficients.size(); ++mode) {
    EXPECT_NEAR(device_coefficients[mode], host_coefficients[mode], 5e-5f) << mode;
  }
}

TEST(ShackHartmannFullPairwisePipeline, NoiselessDefocusMatchesSingleReferenceAndZernikeFit) {
  constexpr size_t height        = 48;
  constexpr size_t width         = 48;
  const float      radius        = 0.5f * static_cast<float>(kSx * width);
  const float      coefficient_m = radius / (4.0f * std::sqrt(3.0f) * 0.4f);

  std::vector<std::pair<float, float>> shifts(kSy * kSx);
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      const float xn   = (static_cast<float>(sx) - 2.0f) / 2.5f;
      const float yn   = (static_cast<float>(sy) - 2.0f) / 2.5f;
      const auto  d    = derivative(4, xn, yn);
      shifts[sy * kSx] = {coefficient_m * d.x / radius, coefficient_m * d.y / radius};
    }
  }
  const auto images             = translated_subapertures(kSy, kSx, height, width, shifts);
  auto       single_settings    = slope_settings(height, width, false, true);
  auto       full_settings      = single_settings;
  full_settings.mode            = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
  full_settings.pair_batch_size = 17;

  const auto single               = run_slopes(images, kSy, kSx, height, width, single_settings);
  const auto full                 = run_slopes(images, kSy, kSx, height, width, full_settings);
  float      max_slope_difference = 0.0f;
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      const size_t i = sy * kSx + sx;
      if (active_sample(sy, sx)) {
        EXPECT_NEAR(full[2 * i], single[2 * i], 0.025f) << "gx sample " << i;
        EXPECT_NEAR(full[2 * i + 1], single[2 * i + 1], 0.025f) << "gy sample " << i;
        max_slope_difference =
            std::max({max_slope_difference, std::abs(full[2 * i] - single[2 * i]),
                      std::abs(full[2 * i + 1] - single[2 * i + 1])});
      } else {
        EXPECT_EQ(full[2 * i], 0.0f);
        EXPECT_EQ(full[2 * i + 1], 0.0f);
      }
    }
  }

  auto zernike_settings               = fit_settings();
  zernike_settings.lambda             = static_cast<float>(width);
  zernike_settings.subaperture_height = height;
  zernike_settings.subaperture_width  = width;
  zernike_settings.stride_y           = height;
  zernike_settings.stride_x           = width;
  holotask::syncs::ZernikeFromSlopesFactory fit_factory;
  const auto                                input_desc = desc({1, kSy, kSx, 2}, MemLoc::Device);
  const auto                                single_fit = holonp_test::run_sync_factory(
      fit_factory, std::vector<TDesc>{input_desc},
      std::vector<std::vector<std::byte>>{as_bytes(single)}, zernike_settings);
  const auto full_fit = holonp_test::run_sync_factory(
      fit_factory, std::vector<TDesc>{input_desc},
      std::vector<std::vector<std::byte>>{as_bytes(full)}, zernike_settings);
  const auto single_coefficients = from_bytes<float>(single_fit.output_bytes[0]);
  const auto full_coefficients   = from_bytes<float>(full_fit.output_bytes[0]);
  ASSERT_EQ(full_coefficients.size(), single_coefficients.size());
  EXPECT_EQ(full_coefficients[0], 0.0f);
  EXPECT_EQ(full_coefficients[1], 0.0f);
  float max_coefficient_difference = 0.0f;
  for (size_t i = 2; i < full_coefficients.size(); ++i) {
    EXPECT_NEAR(full_coefficients[i], single_coefficients[i], 3e-3f) << "Noll " << i + 2;
    max_coefficient_difference = std::max(max_coefficient_difference,
                                          std::abs(full_coefficients[i] - single_coefficients[i]));
  }
  std::cout << "Noiseless defocus maximum star/complete slope difference=" << max_slope_difference
            << "; maximum A4-A10 difference=" << max_coefficient_difference << " rad\n";
}

TEST(ShackHartmannCudaGraph, ReplaysCurrentDataAndRecapturesBuffersAndStreams) {
  constexpr size_t height = 48;
  constexpr size_t width  = 48;
  std::vector<std::pair<float, float>> shifted_positions(kSy * kSx);
  for (size_t i = 0; i < shifted_positions.size(); ++i) {
    shifted_positions[i] = {static_cast<float>(static_cast<int>(i % kSx) - 2),
                            static_cast<float>(static_cast<int>(i / kSx) - 2)};
  }
  const std::vector<std::pair<float, float>> zero_positions(kSy * kSx, {0.0f, 0.0f});
  const auto shifted_images =
      translated_subapertures(kSy, kSx, height, width, shifted_positions);
  const auto zero_images = translated_subapertures(kSy, kSx, height, width, zero_positions);
  const auto input_desc  = desc({1, kSy, kSx, height, width}, MemLoc::Device);

  for (const auto mode : {holotask::syncs::ShackHartmannSlopeMode::SingleReference,
                          holotask::syncs::ShackHartmannSlopeMode::FullPairwise}) {
    SCOPED_TRACE(mode == holotask::syncs::ShackHartmannSlopeMode::SingleReference
                     ? "SingleReference"
                     : "FullPairwise");
    auto settings = slope_settings(height, width,
                                   mode == holotask::syncs::ShackHartmannSlopeMode::SingleReference);
    settings.mode            = mode;
    settings.pair_batch_size = 64;

    holotask::syncs::ShackHartmannSlopesFactory factory;
    const auto inference = factory.infer(std::vector<TDesc>{input_desc}, settings);
    curaii::CudaStream stream_a;
    curaii::CudaStream stream_b;
    auto task = factory.create(std::vector<TDesc>{input_desc}, settings, {.stream = stream_a.get()});
    task->bind_logger(spdlog::default_logger());

    holonp_test::TensorTestBuffer input_a(input_desc);
    holonp_test::TensorTestBuffer input_b(input_desc);
    holonp_test::TensorTestBuffer slopes_a(inference.output_descs[0]);
    holonp_test::TensorTestBuffer slopes_b(inference.output_descs[0]);
    std::unique_ptr<holonp_test::TensorTestBuffer> maps_a;
    std::unique_ptr<holonp_test::TensorTestBuffer> maps_b;
    if (inference.output_descs.size() == 2) {
      maps_a = std::make_unique<holonp_test::TensorTestBuffer>(inference.output_descs[1]);
      maps_b = std::make_unique<holonp_test::TensorTestBuffer>(inference.output_descs[1]);
    }

    std::array inputs_a{input_a.view()};
    std::array inputs_b{input_b.view()};
    std::vector<holoflow::core::TView> outputs_a{slopes_a.view()};
    std::vector<holoflow::core::TView> outputs_b{slopes_b.view()};
    if (maps_a) {
      outputs_a.push_back(maps_a->view());
      outputs_b.push_back(maps_b->view());
    }
    std::atomic<bool> cancelled{false};
    holoflow::core::SyncCtx context_a{inputs_a, outputs_a, &cancelled, nullptr, nullptr};
    holoflow::core::SyncCtx context_b{inputs_b, outputs_b, &cancelled, nullptr, nullptr};

    input_a.upload(as_bytes(shifted_images));
    ASSERT_EQ(task->execute(context_a), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
    const auto shifted_slopes = from_bytes<float>(slopes_a.download());
    EXPECT_GT(*std::max_element(shifted_slopes.begin(), shifted_slopes.end()), 1.0f);

    input_a.upload(as_bytes(zero_images));
    ASSERT_EQ(task->execute(context_a), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
    for (const float slope : from_bytes<float>(slopes_a.download())) {
      EXPECT_NEAR(slope, 0.0f, 2e-3f);
    }

    input_b.upload(as_bytes(shifted_images));
    ASSERT_EQ(task->execute(context_b), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
    const auto recaptured_slopes = from_bytes<float>(slopes_b.download());
    ASSERT_EQ(recaptured_slopes.size(), shifted_slopes.size());
    for (size_t i = 0; i < shifted_slopes.size(); ++i) {
      EXPECT_NEAR(recaptured_slopes[i], shifted_slopes[i], 2e-4f) << "slope " << i;
    }

    task = factory.update(std::move(task), std::vector<TDesc>{input_desc}, settings,
                          {.stream = stream_b.get()});
    input_b.upload(as_bytes(zero_images));
    ASSERT_EQ(task->execute(context_b), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_b.get()));
    for (const float slope : from_bytes<float>(slopes_b.download())) {
      EXPECT_NEAR(slope, 0.0f, 2e-3f);
    }

    input_b.upload(as_bytes(shifted_images));
    ASSERT_EQ(task->execute(context_b), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_b.get()));
    const auto updated_stream_slopes = from_bytes<float>(slopes_b.download());
    ASSERT_EQ(updated_stream_slopes.size(), shifted_slopes.size());
    for (size_t i = 0; i < shifted_slopes.size(); ++i) {
      EXPECT_NEAR(updated_stream_slopes[i], shifted_slopes[i], 2e-4f) << "slope " << i;
    }
  }
}

TEST(ShackHartmannCudaGraph, FallsBackOnDefaultStream) {
  constexpr size_t height = 24;
  constexpr size_t width  = 24;
  const std::vector<std::pair<float, float>> positions(kSy * kSx, {0.0f, 0.0f});
  const auto images     = translated_subapertures(kSy, kSx, height, width, positions);
  const auto input_desc = desc({1, kSy, kSx, height, width}, MemLoc::Device);
  auto settings         = slope_settings(height, width, false);

  holotask::syncs::ShackHartmannSlopesFactory factory;
  const auto inference = factory.infer(std::vector<TDesc>{input_desc}, settings);
  auto task = factory.create(std::vector<TDesc>{input_desc}, settings, {.stream = nullptr});
  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(inference.output_descs[0]);
  input.upload(as_bytes(images));
  std::array        inputs{input.view()};
  std::array        outputs{output.view()};
  std::atomic<bool> cancelled{false};
  holoflow::core::SyncCtx context{inputs, outputs, &cancelled, nullptr, nullptr};

  ASSERT_EQ(task->execute(context), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (const float slope : from_bytes<float>(output.download())) {
    EXPECT_NEAR(slope, 0.0f, 2e-3f);
  }
}

TEST(ShackHartmannSlopesPerformance, ReportsRepresentativeRuntimeAndExplicitBuffers) {
  constexpr size_t                     height          = 48;
  constexpr size_t                     width           = 48;
  constexpr size_t                     active_count    = 21;
  constexpr size_t                     edge_count      = active_count * (active_count - 1) / 2;
  constexpr size_t                     pair_batch_size = 64;
  constexpr size_t                     iterations      = 12;
  std::vector<std::pair<float, float>> shifts(kSy * kSx);
  for (size_t i = 0; i < shifts.size(); ++i) {
    shifts[i] = {static_cast<float>(static_cast<int>(i % 5) - 2),
                 static_cast<float>(static_cast<int>(i / 5) - 2)};
  }
  const auto images             = translated_subapertures(kSy, kSx, height, width, shifts);
  auto       single_settings    = slope_settings(height, width, false, true);
  auto       full_settings      = single_settings;
  full_settings.mode            = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
  full_settings.pair_batch_size = pair_batch_size;

  const float single_ms =
      time_slope_execution(images, kSy, kSx, height, width, single_settings, iterations);
  const float full_ms =
      time_slope_execution(images, kSy, kSx, height, width, full_settings, iterations);
  EXPECT_GT(single_ms, 0.0f);
  EXPECT_GT(full_ms, 0.0f);

  constexpr size_t spectrum_bytes   = active_count * height * width * sizeof(cuFloatComplex);
  constexpr size_t pair_map_bytes   = pair_batch_size * height * width * sizeof(cuFloatComplex);
  constexpr size_t pair_index_bytes = 2 * pair_batch_size * sizeof(size_t);
  constexpr size_t pair_shift_bytes = pair_batch_size * sizeof(float2);
  constexpr size_t explicit_temporary_bytes = pair_map_bytes + pair_index_bytes + pair_shift_bytes;
  std::cout << "FullPairwise benchmark: N=" << active_count << ", E=" << edge_count
            << ", HxW=" << height << 'x' << width << ", batch=" << pair_batch_size
            << ", persistent spectra=" << spectrum_bytes
            << " bytes, explicit pair temporary=" << explicit_temporary_bytes
            << " bytes, SingleReference=" << single_ms << " ms, FullPairwise=" << full_ms
            << " ms, ratio=" << full_ms / single_ms << "\n";
}

TEST(ShackHartmannPipelineRegression, MeasuredDataMatchesCorrectedCenterReferenceFit) {
  constexpr size_t           height = 48;
  constexpr size_t           width  = 48;
  const std::array<float, 7> injected_m{0.4f, -0.3f, 0.2f, 0.15f, -0.1f, 0.2f, -0.15f};
  const float                radius = 0.5f * static_cast<float>(kSx * width);

  std::vector<std::pair<float, float>> shifts(kSy * kSx, {0.0f, 0.0f});
  for (size_t sy = 0; sy < kSy; ++sy) {
    for (size_t sx = 0; sx < kSx; ++sx) {
      const float xn    = (static_cast<float>(sx) - 2.0f) / 2.5f;
      const float yn    = (static_cast<float>(sy) - 2.0f) / 2.5f;
      auto       &shift = shifts[sy * kSx + sx];
      for (int mode = 4; mode <= 10; ++mode) {
        const auto  value       = derivative(mode, xn, yn);
        const float coefficient = injected_m[static_cast<size_t>(mode - 4)];
        shift.first += coefficient * value.x / radius;
        shift.second += coefficient * value.y / radius;
      }
    }
  }

  const auto images = translated_subapertures(kSy, kSx, height, width, shifts);
  holotask::syncs::ShackHartmannSlopesFactory slope_factory;
  const auto                                  slope_run = holonp_test::run_sync_factory(
      slope_factory, std::vector<TDesc>{desc({1, kSy, kSx, height, width}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(images)},
      slope_settings(height, width, true, true));
  const auto measured_slopes          = from_bytes<float>(slope_run.output_bytes[0]);
  const auto maps                     = from_bytes<float>(slope_run.output_bytes[1]);
  auto       full_slope_settings      = slope_settings(height, width, false, true);
  full_slope_settings.mode            = holotask::syncs::ShackHartmannSlopeMode::FullPairwise;
  full_slope_settings.pair_batch_size = 64;
  const auto full_slopes = run_slopes(images, kSy, kSx, height, width, full_slope_settings);

  auto settings               = fit_settings();
  settings.lambda             = static_cast<float>(width);
  settings.subaperture_height = height;
  settings.subaperture_width  = width;
  settings.stride_y           = height;
  settings.stride_x           = width;
  holotask::syncs::ZernikeFromSlopesFactory fit_factory;
  const auto                                fit_run = holonp_test::run_sync_factory(
      fit_factory, std::vector<TDesc>{desc({1, kSy, kSx, 2}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(measured_slopes)}, settings);
  const auto new_coefficients = from_bytes<float>(fit_run.output_bytes[0]);
  const auto full_fit_run     = holonp_test::run_sync_factory(
      fit_factory, std::vector<TDesc>{desc({1, kSy, kSx, 2}, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(full_slopes)}, settings);
  const auto full_coefficients = from_bytes<float>(full_fit_run.output_bytes[0]);
  const auto old_coefficients  = legacy_center_fit(maps, height, width, static_cast<float>(width));

  float                max_coefficient_difference = 0.0f;
  std::array<float, 7> coefficient_differences{};
  for (size_t mode = 0; mode < 7; ++mode) {
    const float difference        = std::abs(new_coefficients[mode + 2] - old_coefficients[mode]);
    coefficient_differences[mode] = difference;
    max_coefficient_difference    = std::max(max_coefficient_difference, difference);
    EXPECT_NEAR(new_coefficients[mode + 2], old_coefficients[mode], 3e-3f) << "A" << mode + 4;
  }
  EXPECT_EQ(new_coefficients[0], 0.0f);
  EXPECT_EQ(new_coefficients[1], 0.0f);
  EXPECT_EQ(full_coefficients[0], 0.0f);
  EXPECT_EQ(full_coefficients[1], 0.0f);

  std::array<float, 7> mode_differences{};
  for (size_t mode = 0; mode < mode_differences.size(); ++mode) {
    mode_differences[mode] = std::abs(full_coefficients[mode + 2] - new_coefficients[mode + 2]);
    EXPECT_NEAR(full_coefficients[mode + 2], new_coefficients[mode + 2], 3e-3f) << "A" << mode + 4;
  }

  float max_phase_difference = 0.0f;
  for (size_t y = 0; y < 65; ++y) {
    for (size_t x = 0; x < 65; ++x) {
      const float xn         = (static_cast<float>(x) - 32.0f) / 32.0f;
      const float yn         = (static_cast<float>(y) - 32.0f) / 32.0f;
      float       difference = 0.0f;
      for (int mode = 4; mode <= 10; ++mode) {
        difference += (new_coefficients[static_cast<size_t>(mode - 2)] -
                       old_coefficients[static_cast<size_t>(mode - 4)]) *
                      zernike_value(mode, xn, yn);
      }
      max_phase_difference = std::max(max_phase_difference, std::abs(difference));
    }
  }
  EXPECT_LT(max_phase_difference, 5e-2f);
  std::cout << "A4-A10 absolute differences:";
  for (const float difference : coefficient_differences) {
    std::cout << ' ' << difference;
  }
  std::cout << "; max difference: " << max_coefficient_difference
            << " rad; max phase difference: " << max_phase_difference << " rad\n";
  std::cout << "FullPairwise versus SingleReference A4-A10 absolute differences:";
  for (const float difference : mode_differences) {
    std::cout << ' ' << difference;
  }
  std::cout << " rad\n";
}

TEST(ZernikeFromSlopesInference, RejectsPartialRegionalGaugeAndReusesTask) {
  holotask::syncs::ZernikeFromSlopesFactory factory;
  const auto                                input    = desc({1, kSy, kSx, 2}, MemLoc::Device);
  auto                                      settings = fit_settings();
  settings.nx                                        = 2;
  EXPECT_THROW((void)factory.infer(std::vector<TDesc>{input}, settings), std::invalid_argument);

  settings.nx                     = 1;
  const nlohmann::json serialized = settings;
  EXPECT_EQ(serialized.get<holotask::syncs::ZernikeFromSlopesSettings>(), settings);
  const auto device_infer = factory.infer(std::vector<TDesc>{input}, settings);
  EXPECT_EQ(device_infer.output_descs[0].mem_loc, MemLoc::Device);
  const auto host_infer =
      factory.infer(std::vector<TDesc>{desc({1, kSy, kSx, 2}, MemLoc::Host)}, settings);
  EXPECT_EQ(host_infer.output_descs[0].mem_loc, MemLoc::Host);

  const auto run = holonp_test::run_sync_factory_update(
      factory, std::vector<TDesc>{input},
      std::vector<std::vector<std::byte>>{as_bytes(ideal_slopes(4, 0.25f))}, settings);
  const auto coefficients = from_bytes<float>(run.output_bytes[0]);
  EXPECT_EQ(coefficients[0], 0.0f);
  EXPECT_EQ(coefficients[1], 0.0f);
}
