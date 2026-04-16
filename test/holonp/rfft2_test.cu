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
#include "holonp/rfft2.hh"

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

static void expect_cf32_near(const std::vector<std::byte> &actual,
                             const std::vector<std::byte> &expected, float rtol = 1e-4f) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t n = actual.size() / sizeof(float);
  const auto  *a = reinterpret_cast<const float *>(actual.data());
  const auto  *e = reinterpret_cast<const float *>(expected.data());
  for (size_t i = 0; i < n; ++i) {
    const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
    EXPECT_NEAR(a[i], e[i], tol);
  }
}

class RFFT2InferTest : public ::testing::Test {
protected:
  holonp::RFFT2Factory factory;
};

TEST_F(RFFT2InferTest, UsesTrailingAxesAndShrinksLastDim) {
  const std::vector<TDesc> in = {device_desc({2, 4}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
}

TEST_F(RFFT2InferTest, RejectsUnsupportedAxes) {
  const std::vector<TDesc> in = {device_desc({2, 3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json{{"axes", std::vector<int>{0, 1}}}),
               std::invalid_argument);
}

class RFFT2OracleTest : public ::testing::Test {
protected:
  holonp::RFFT2Factory factory;
};

TEST_F(RFFT2OracleTest, F32DefaultAxes) {
  const TDesc d    = device_desc({2, 4}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 0.f, -1.f, 2.f, -3.f});
  const auto  j    = nlohmann::json::object();
  const auto  run  = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                   std::vector<std::vector<std::byte>>{data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "rfft2";
  oi.n_outputs      = 1;
  oi.input_descs    = {d};
  oi.input_bytes    = {data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_cf32_near(run.output_bytes[0], oracle.output_bytes[0]);
}

class RFFT2UpdateTest : public ::testing::Test {
protected:
  holonp::RFFT2Factory factory;
};

TEST_F(RFFT2UpdateTest, ReusesTaskWithSameConfig) {
  const TDesc d    = device_desc({2, 4}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 0.f, -1.f, 2.f, -3.f});
  const auto  j    = nlohmann::json::object();
  const auto  run  = holonp_test::run_sync_factory_update(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "rfft2";
  oi.n_outputs      = 1;
  oi.input_descs    = {d};
  oi.input_bytes    = {data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_cf32_near(run.output_bytes[0], oracle.output_bytes[0]);
}
