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
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/zernike_phase.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

TDesc desc(std::vector<size_t> shape, DType dtype, MemLoc mem_loc) {
  return TDesc(std::move(shape), dtype, mem_loc);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

float eval_zernike_noll_value(int noll_index, float x_n, float y_n) {
  const float sqrt3 = std::sqrt(3.0f);

  switch (noll_index) {
  case 2:
    return 2.0f * x_n;
  case 4:
    return sqrt3 * (2.0f * (x_n * x_n + y_n * y_n) - 1.0f);
  default:
    return 0.0f;
  }
}

std::vector<float> expected_phase() {
  constexpr int ny = 2;
  constexpr int nx = 3;

  const std::vector<int>   indexes{2, 4};
  const std::vector<float> coeffs{1.0f, 0.5f};

  const float center_y     = (static_cast<float>(ny) - 1.0f) * 0.5f;
  const float center_x     = (static_cast<float>(nx) - 1.0f) * 0.5f;
  const float pupil_radius = 0.5f * static_cast<float>(std::min(ny, nx));

  std::vector<float> expected(static_cast<size_t>(ny * nx));
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const float x_n = (static_cast<float>(x) - center_x) / pupil_radius;
      const float y_n = (static_cast<float>(y) - center_y) / pupil_radius;

      float phase = 0.0f;
      for (size_t i = 0; i < indexes.size(); ++i) {
        phase += coeffs[i] * eval_zernike_noll_value(indexes[i], x_n, y_n);
      }

      expected[static_cast<size_t>(y * nx + x)] = phase;
    }
  }
  return expected;
}

nlohmann::json host_settings() {
  return holotask::syncs::ZernikePhaseSettings{
      .indexes = {2, 4},
      .ny      = 2,
      .nx      = 3,
  };
}

nlohmann::json device_settings() {
  return holotask::syncs::ZernikePhaseSettings{
      .indexes = {2, 4},
      .ny      = 2,
      .nx      = 3,
      .output  = MemLoc::Device,
  };
}

void expect_f32_near(const std::vector<std::byte> &actual, const std::vector<float> &expected,
                     float atol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(float));
  const auto *a = reinterpret_cast<const float *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(a[i], expected[i], atol) << "at index " << i;
  }
}

} // namespace

class ZernikePhaseInferTest : public ::testing::Test {
protected:
  holotask::syncs::ZernikePhaseFactory factory;
};

TEST_F(ZernikePhaseInferTest, DefaultsToHostOutput) {
  const std::vector<TDesc> in = {desc({2}, DType::F32, MemLoc::Host)};
  const auto               r  = factory.infer(in, host_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Host);
}

TEST_F(ZernikePhaseInferTest, CanRequestDeviceOutput) {
  const std::vector<TDesc> in = {desc({2}, DType::F32, MemLoc::Device)};
  const auto               r  = factory.infer(in, device_settings());

  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(ZernikePhaseInferTest, RejectsDeviceOutputWithHostCoefficients) {
  const std::vector<TDesc> in = {desc({2}, DType::F32, MemLoc::Host)};

  EXPECT_THROW((void)factory.infer(in, device_settings()), std::invalid_argument);
}

class ZernikePhaseExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::ZernikePhaseFactory factory;
};

TEST_F(ZernikePhaseExecuteTest, ComputesHostOutput) {
  const TDesc d    = desc({2}, DType::F32, MemLoc::Host);
  const auto  data = as_bytes(std::vector<float>{1.0f, 0.5f});

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, host_settings());

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Host);
  expect_f32_near(run.output_bytes[0], expected_phase());
}

TEST_F(ZernikePhaseExecuteTest, ComputesDeviceOutputInKernel) {
  const TDesc d    = desc({2}, DType::F32, MemLoc::Device);
  const auto  data = as_bytes(std::vector<float>{1.0f, 0.5f});

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, device_settings());

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Device);
  expect_f32_near(run.output_bytes[0], expected_phase());
}
