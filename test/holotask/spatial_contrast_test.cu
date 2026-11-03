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
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/spatial_contrast.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

nlohmann::json settings(int window_size = 3) {
  return holotask::syncs::SpatialContrastSettings{.window_size = window_size};
}

template <typename T>
std::vector<float> spatial_contrast_reference(const std::vector<T> &input, int width, int height,
                                              int window_size) {
  std::vector<float> output(input.size());
  const int          radius = window_size / 2;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int x_begin = std::max(0, x - radius);
      const int x_end   = std::min(width - 1, x + radius);
      const int y_begin = std::max(0, y - radius);
      const int y_end   = std::min(height - 1, y + radius);

      double sum = 0.0;
      int    count = 0;
      for (int sample_y = y_begin; sample_y <= y_end; ++sample_y) {
        for (int sample_x = x_begin; sample_x <= x_end; ++sample_x) {
          sum += static_cast<double>(input[static_cast<size_t>(sample_y * width + sample_x)]);
          ++count;
        }
      }

      if (count == 0) {
        continue;
      }

      const double mean = sum / static_cast<double>(count);
      double       squared_deviation_sum = 0.0;
      for (int sample_y = y_begin; sample_y <= y_end; ++sample_y) {
        for (int sample_x = x_begin; sample_x <= x_end; ++sample_x) {
          const double value =
              static_cast<double>(input[static_cast<size_t>(sample_y * width + sample_x)]);
          const double deviation = value - mean;
          squared_deviation_sum += deviation * deviation;
        }
      }

      const size_t index = static_cast<size_t>(y * width + x);
      output[index] = mean > 0.0
                          ? static_cast<float>(std::sqrt(squared_deviation_sum / count) / mean)
                          : 0.0F;
    }
  }

  return output;
}

std::vector<float> as_floats(const std::vector<std::byte> &bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

void expect_f32_near(const std::vector<std::byte> &actual, const std::vector<float> &expected,
                     float tolerance = 1e-5F) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(float));
  const auto values = as_floats(actual);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(values[i], expected[i], tolerance) << "element " << i;
  }
}

} // namespace

class SpatialContrastInferTest : public ::testing::Test {
protected:
  holotask::syncs::SpatialContrastFactory factory;
};

TEST_F(SpatialContrastInferTest, PreservesShapeAndReturnsF32) {
  const std::vector<TDesc> input = {device_desc({1, 3, 4}, DType::U16)};
  const auto               result = factory.infer(input, settings());

  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 3, 4}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
  EXPECT_TRUE(result.in_place.empty());
}

TEST_F(SpatialContrastInferTest, RejectsDegenerateOrEvenWindowSizes) {
  const std::vector<TDesc> input = {device_desc({1, 3, 4}, DType::U8)};
  EXPECT_THROW(factory.infer(input, settings(1)), std::invalid_argument);
  EXPECT_THROW(factory.infer(input, settings(2)), std::invalid_argument);
}

class SpatialContrastExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::SpatialContrastFactory factory;
};

TEST_F(SpatialContrastExecuteTest, U8SlidingWindowMatchesClippedBorderReference) {
  const TDesc                desc = device_desc({1, 3, 4}, DType::U8);
  const std::vector<uint8_t> input{2, 8, 3, 9, 4, 1, 7, 6, 5, 0, 2, 8};
  const auto                 expected = spatial_contrast_reference(input, 4, 3, 3);
  const auto                 result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc},
      std::vector<std::vector<std::byte>>{as_bytes(input)}, settings());

  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 3, 4}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
  expect_f32_near(result.output_bytes[0], expected);
}

TEST_F(SpatialContrastExecuteTest, U16HighBaselineRetainsSmallSpeckleVariance) {
  const TDesc                 desc = device_desc({1, 3, 3}, DType::U16);
  const std::vector<uint16_t> input{60000, 60001, 60002, 60003, 60004,
                                    60005, 60006, 60007, 60008};
  const auto                  expected = spatial_contrast_reference(input, 3, 3, 3);
  const auto                  result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc},
      std::vector<std::vector<std::byte>>{as_bytes(input)}, settings());
  const auto actual = as_floats(result.output_bytes[0]);

  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_GT(actual[4], 0.0F);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-8F) << "element " << i;
  }
}

TEST_F(SpatialContrastExecuteTest, ConstantInputProducesZeroContrast) {
  const TDesc                desc = device_desc({1, 2, 2}, DType::U8);
  const std::vector<uint8_t> input(4, 128);
  const auto                 result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc},
      std::vector<std::vector<std::byte>>{as_bytes(input)}, settings());

  expect_f32_near(result.output_bytes[0], std::vector<float>(4, 0.0F));
}
