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

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/flatfield.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

nlohmann::json settings(float sigma_y = 1.0f, float sigma_x = 1.0f) {
  return holotask::syncs::FlatfieldSettings{.sigma_y = sigma_y, .sigma_x = sigma_x};
}

std::vector<float> make_kernel(float sigma) {
  const int          radius = static_cast<int>(std::ceil(3.0f * sigma));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
  const float        inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);
  float              sum            = 0.0f;

  for (int i = -radius; i <= radius; ++i) {
    const float value                       = std::exp(-static_cast<float>(i * i) * inv_two_sigma2);
    kernel[static_cast<size_t>(i + radius)] = value;
    sum += value;
  }

  for (auto &value : kernel) {
    value /= sum;
  }
  return kernel;
}

std::vector<float> flatfield_reference(const std::vector<float> &input, int height, int width,
                                       float sigma_y, float sigma_x) {
  const auto kernel_y = make_kernel(sigma_y);
  const auto kernel_x = make_kernel(sigma_x);
  const int  radius_y = static_cast<int>(kernel_y.size() / 2);
  const int  radius_x = static_cast<int>(kernel_x.size() / 2);

  std::vector<float> temp(input.size());
  std::vector<float> output(input.size());

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius_x; k <= radius_x; ++k) {
        const int xk = x + k;
        const int xx = xk < 0 ? 0 : (xk >= width ? width - 1 : xk);
        sum += input[static_cast<size_t>(y * width + xx)] *
               kernel_x[static_cast<size_t>(k + radius_x)];
      }
      temp[static_cast<size_t>(y * width + x)] = sum;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float background = 0.0f;
      for (int k = -radius_y; k <= radius_y; ++k) {
        const int yk = y + k;
        const int yy = yk < 0 ? 0 : (yk >= height ? height - 1 : yk);
        background +=
            temp[static_cast<size_t>(yy * width + x)] * kernel_y[static_cast<size_t>(k + radius_y)];
      }
      output[static_cast<size_t>(y * width + x)] =
          input[static_cast<size_t>(y * width + x)] - background;
    }
  }

  return output;
}

void expect_f32_near(const std::vector<std::byte> &actual, const std::vector<float> &expected,
                     float atol = 1e-4f) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(float));
  const auto *a = reinterpret_cast<const float *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(a[i], expected[i], atol);
  }
}

} // namespace

class FlatfieldInferTest : public ::testing::Test {
protected:
  holotask::syncs::FlatfieldFactory factory;
};

TEST_F(FlatfieldInferTest, KeepsF32ShapeAndDtype) {
  const std::vector<TDesc> in = {device_desc({2, 3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3, 4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(FlatfieldInferTest, RejectsNonPositiveSigma) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};
  EXPECT_THROW(factory.infer(in, settings(0.0f, 1.0f)), std::invalid_argument);
  EXPECT_THROW(factory.infer(in, settings(1.0f, 0.0f)), std::invalid_argument);
}

class FlatfieldExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::FlatfieldFactory factory;
};

TEST_F(FlatfieldExecuteTest, ConstantInputSubtractsToZero) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f});

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::F32);
  expect_f32_near(run.output_bytes[0], std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
}

TEST_F(FlatfieldExecuteTest, AnisotropicSigmaMatchesReference) {
  const TDesc              d     = device_desc({3, 4}, DType::F32);
  const std::vector<float> input = {
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
  };
  const float sigma_y = 0.75f;
  const float sigma_x = 1.25f;

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(sigma_y, sigma_x));

  ASSERT_EQ(run.output_descs.size(), 1);
  expect_f32_near(run.output_bytes[0], flatfield_reference(input, 3, 4, sigma_y, sigma_x));
}

TEST_F(FlatfieldExecuteTest, LargeSigmaConstantInputSubtractsToZero) {
  const TDesc              d = device_desc({8, 8}, DType::F32);
  const std::vector<float> input(64, 7.0f);

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(12.0f, 12.0f));

  ASSERT_EQ(run.output_descs.size(), 1);
  expect_f32_near(run.output_bytes[0], std::vector<float>(64, 0.0f), 1e-3f);
}
