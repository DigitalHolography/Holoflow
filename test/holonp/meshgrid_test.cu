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
#include "holonp/meshgrid.hh"

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
  if (dtype == DType::U8) {
    const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]);
  } else if (dtype == DType::U16) {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]);
  } else if (dtype == DType::F32) {
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

class MeshgridInferTest : public ::testing::Test {
protected:
  holonp::MeshgridFactory factory;
};

TEST_F(MeshgridInferTest, XYOutputShapeForTwoInputs) {
  const std::vector<TDesc> in = {device_desc({2}, DType::F32), device_desc({3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json{{"indexing", "xy"}});
  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 2u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 2}));
  EXPECT_EQ(r.output_descs[1].shape, (std::vector<size_t>{3, 2}));
}

TEST_F(MeshgridInferTest, RejectsUnsupportedSparseTrue) {
  const std::vector<TDesc> in = {device_desc({2}, DType::F32), device_desc({3}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json{{"indexing", "xy"}, {"sparse", true}}),
               std::invalid_argument);
}

class MeshgridOracleTest : public ::testing::Test {
protected:
  holonp::MeshgridFactory factory;
};

TEST_F(MeshgridOracleTest, F32XYTwoInputs) {
  const TDesc                               x    = device_desc({2}, DType::F32);
  const TDesc                               y    = device_desc({3}, DType::F32);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f}),
      as_bytes(std::vector<float>{10.f, 20.f, 30.f}),
  };
  const auto j = nlohmann::json{{"indexing", "xy"}};

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{x, y}, data, j);

  holonp_test::OracleInput oi;
  oi.op             = "meshgrid";
  oi.n_outputs      = 2;
  oi.input_descs    = {x, y};
  oi.input_bytes    = data;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(run.output_bytes.size(), 2u);
  ASSERT_EQ(oracle.output_bytes.size(), 2u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
  expect_near_oracle(run.output_bytes[1], oracle.output_bytes[1], DType::F32);
}

TEST_F(MeshgridOracleTest, U16IJThreeInputs) {
  const TDesc                               a    = device_desc({2}, DType::U16);
  const TDesc                               b    = device_desc({3}, DType::U16);
  const TDesc                               c    = device_desc({2}, DType::U16);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint16_t>{1, 2}),
      as_bytes(std::vector<std::uint16_t>{3, 4, 5}),
      as_bytes(std::vector<std::uint16_t>{6, 7}),
  };
  const auto j = nlohmann::json{{"indexing", "ij"}};

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{a, b, c}, data, j);

  holonp_test::OracleInput oi;
  oi.op             = "meshgrid";
  oi.n_outputs      = 3;
  oi.input_descs    = {a, b, c};
  oi.input_bytes    = data;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(run.output_bytes.size(), 3u);
  ASSERT_EQ(oracle.output_bytes.size(), 3u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::U16);
  expect_near_oracle(run.output_bytes[1], oracle.output_bytes[1], DType::U16);
  expect_near_oracle(run.output_bytes[2], oracle.output_bytes[2], DType::U16);
}

class MeshgridUpdateTest : public ::testing::Test {
protected:
  holonp::MeshgridFactory factory;
};

TEST_F(MeshgridUpdateTest, ReusesTaskWithSameConfig) {
  const TDesc                               x    = device_desc({2}, DType::F32);
  const TDesc                               y    = device_desc({3}, DType::F32);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f}),
      as_bytes(std::vector<float>{10.f, 20.f, 30.f}),
  };
  const auto j = nlohmann::json{{"indexing", "xy"}};

  const auto run = holonp_test::run_sync_factory_update(factory, std::vector<TDesc>{x, y}, data, j);

  holonp_test::OracleInput oi;
  oi.op             = "meshgrid";
  oi.n_outputs      = 2;
  oi.input_descs    = {x, y};
  oi.input_bytes    = data;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(run.output_bytes.size(), 2u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
  expect_near_oracle(run.output_bytes[1], oracle.output_bytes[1], DType::F32);
}
