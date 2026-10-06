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
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/matmul.hh"

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

static void expect_f32_near(const std::vector<std::byte> &actual,
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

class MatmulInferTest : public ::testing::Test {
protected:
  holonp::MatmulFactory factory;
};

TEST_F(MatmulInferTest, InfersMatrixOutput) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::F32), device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(MatmulInferTest, InfersBatchedOutput) {
  const std::vector<TDesc> in = {device_desc({5, 2, 3}, DType::CF32),
                                 device_desc({5, 3, 4}, DType::CF32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{5, 2, 4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
}

TEST_F(MatmulInferTest, RejectsBroadcastBatchDimensions) {
  const std::vector<TDesc> in = {device_desc({5, 2, 3}, DType::F32),
                                 device_desc({1, 3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(MatmulInferTest, RejectsMismatchedInnerDimensions) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::F32), device_desc({2, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

class MatmulOracleTest : public ::testing::Test {
protected:
  holonp::MatmulFactory factory;
};

TEST_F(MatmulOracleTest, F32MatchesNumpy) {
  const TDesc a_desc = device_desc({2, 3}, DType::F32);
  const TDesc b_desc = device_desc({3, 2}, DType::F32);
  const auto  a_data = as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  const auto  b_data = as_bytes(std::vector<float>{2.f, -1.f, 0.f, 3.f, 4.f, 2.f});
  const auto  j      = nlohmann::json::object();
  const auto  run =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{a_desc, b_desc},
                                    std::vector<std::vector<std::byte>>{a_data, b_data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "matmul";
  oi.n_outputs      = 1;
  oi.input_descs    = {a_desc, b_desc};
  oi.input_bytes    = {a_data, b_data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_f32_near(run.output_bytes[0], oracle.output_bytes[0]);
}

TEST_F(MatmulOracleTest, BatchedComplexMatchesNumpy) {
  struct CF32 {
    float re, im;
  };
  const TDesc a_desc = device_desc({2, 2, 2}, DType::CF32);
  const TDesc b_desc = device_desc({2, 2, 2}, DType::CF32);
  const auto  a_data = as_bytes(std::vector<CF32>{{1.f, 1.f},
                                                  {2.f, 0.f},
                                                  {0.f, -1.f},
                                                  {3.f, 2.f},
                                                  {2.f, 0.f},
                                                  {-1.f, 1.f},
                                                  {4.f, 1.f},
                                                  {0.f, 2.f}});
  const auto  b_data = as_bytes(std::vector<CF32>{{1.f, 0.f},
                                                  {0.f, 1.f},
                                                  {2.f, -1.f},
                                                  {1.f, 0.f},
                                                  {0.f, 2.f},
                                                  {1.f, 1.f},
                                                  {-1.f, 0.f},
                                                  {2.f, 0.f}});
  const auto  j      = nlohmann::json::object();
  const auto  run =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{a_desc, b_desc},
                                    std::vector<std::vector<std::byte>>{a_data, b_data}, j);

  holonp_test::OracleInput oi;
  oi.op             = "matmul";
  oi.n_outputs      = 1;
  oi.input_descs    = {a_desc, b_desc};
  oi.input_bytes    = {a_data, b_data};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(run.output_bytes[0].size(), oracle.output_bytes[0].size());
  const size_t n        = run.output_bytes[0].size() / sizeof(float);
  const auto  *actual   = reinterpret_cast<const float *>(run.output_bytes[0].data());
  const auto  *expected = reinterpret_cast<const float *>(oracle.output_bytes[0].data());
  for (size_t i = 0; i < n; ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-4f * std::max(std::abs(expected[i]), 1.0f));
}
