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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/mean.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

static TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

static void expect_near_oracle(const std::vector<std::byte> &actual,
                               const std::vector<std::byte> &expected, DType dtype,
                               float rtol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t n = actual.size() / holoflow::core::size_of(dtype);
  if (dtype == DType::F32) {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol);
    }
  } else {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n * 2; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol);
    }
  }
}

static nlohmann::json mean_settings_axis(int axis, bool keepdims = false) {
  return nlohmann::json{{"axis", axis}, {"keepdims", keepdims}};
}

static nlohmann::json mean_settings_all(bool keepdims = false) {
  return nlohmann::json{{"axis", nullptr}, {"keepdims", keepdims}};
}

class MeanInferTest : public ::testing::Test {
protected:
  holonp::MeanFactory factory;
};

TEST_F(MeanInferTest, PromotesRealInputsToF32) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::U16)};
  const auto               r  = factory.infer(in, mean_settings_axis(1));
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(MeanInferTest, KeepsComplexOutputDtype) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::CF32)};
  const auto               r  = factory.infer(in, mean_settings_axis(0, true));
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
}

TEST_F(MeanInferTest, RejectsNonContiguousInput) {
  const TDesc d({2, 3}, DType::F32, MemLoc::Device, std::vector<size_t>{16, 4});
  EXPECT_THROW(factory.infer(std::vector<TDesc>{d}, mean_settings_axis(1)), std::invalid_argument);
}

class MeanOracleTest : public ::testing::Test {
protected:
  holonp::MeanFactory factory;
};

TEST_F(MeanOracleTest, F32Axis1) {
  const TDesc d    = device_desc({2, 3}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});
  const auto  j    = mean_settings_axis(1);
  const auto  run  = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                   std::vector<std::vector<std::byte>>{data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "mean";
  oi.n_outputs      = 1;
  oi.input_descs    = {d};
  oi.input_bytes    = {data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(MeanOracleTest, U16AllAxesKeepdims) {
  const TDesc d    = device_desc({2, 2}, DType::U16);
  const auto  data = as_bytes(std::vector<std::uint16_t>{2, 4, 6, 8});
  const auto  j    = mean_settings_all(true);
  const auto  run  = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                   std::vector<std::vector<std::byte>>{data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "mean";
  oi.n_outputs      = 1;
  oi.input_descs    = {d};
  oi.input_bytes    = {data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

class MeanUpdateTest : public ::testing::Test {
protected:
  holonp::MeanFactory factory;
};

TEST_F(MeanUpdateTest, ReusesTaskWithSameConfig) {
  const TDesc d    = device_desc({2, 3}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});
  const auto  j    = mean_settings_axis(1);
  const auto  run  = holonp_test::run_sync_factory_update(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "mean";
  oi.n_outputs      = 1;
  oi.input_descs    = {d};
  oi.input_bytes    = {data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}
