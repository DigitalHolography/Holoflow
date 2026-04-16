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

#include <atomic>
#include <cmath>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/zeros.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

static nlohmann::json make_jsettings(std::vector<size_t> shape) {
  return nlohmann::json{{"shape", shape}};
}

static nlohmann::json make_jsettings(std::vector<size_t> shape, const std::string &dtype) {
  auto j     = make_jsettings(std::move(shape));
  j["dtype"] = dtype;
  return j;
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
  } else {
    const auto  *a = reinterpret_cast<const float *>(actual.data());
    const auto  *e = reinterpret_cast<const float *>(expected.data());
    const size_t m = actual.size() / sizeof(float);
    for (size_t i = 0; i < m; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol);
    }
  }
}

class ZerosInferTest : public ::testing::Test {
protected:
  holonp::ZerosFactory factory;
};

TEST_F(ZerosInferTest, DefaultsF32Device) {
  const auto r = factory.infer({}, make_jsettings({2, 3}));
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(ZerosInferTest, ExplicitU16) {
  const auto r = factory.infer({}, make_jsettings({4}, "U16"));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ZerosInferTest, RejectsBadInputOrHostOrder) {
  const std::vector<TDesc> in = {TDesc({1}, DType::F32, MemLoc::Device)};
  EXPECT_THROW(factory.infer(in, make_jsettings({1})), std::invalid_argument);
  auto j     = make_jsettings({1});
  j["order"] = "F";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
  j           = make_jsettings({1});
  j["device"] = "Host";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
}

class ZerosOracleTest : public ::testing::Test {
protected:
  holonp::ZerosFactory                      factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(ZerosOracleTest, F32) {
  const auto               j   = make_jsettings({2, 3});
  const auto               run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);
  holonp_test::OracleInput oi;
  oi.op             = "zeros";
  oi.n_outputs      = 1;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(ZerosOracleTest, CF32) {
  const auto               j   = make_jsettings({3}, "CF32");
  const auto               run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);
  holonp_test::OracleInput oi;
  oi.op             = "zeros";
  oi.n_outputs      = 1;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::CF32);
}

class ZerosUpdateTest : public ::testing::Test {
protected:
  holonp::ZerosFactory                      factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(ZerosUpdateTest, ReusesZerosTask) {
  const auto j   = make_jsettings({4}, "U8");
  const auto run = holonp_test::run_sync_factory_update(factory, no_inputs, no_data, j);
  holonp_test::OracleInput oi;
  oi.op             = "zeros";
  oi.n_outputs      = 1;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::U8);
}

TEST_F(ZerosUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const auto                          j = make_jsettings({2}, "F32");
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), {}, j, ctx);

  const auto                    infer = factory.infer({}, j);
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);
  auto                          ov = out_buf.view();
  std::atomic<bool>             cancelled{false};
  holoflow::core::SyncCtx       exec_ctx{
            .inputs       = {},
            .outputs      = {&ov, 1},
            .cancelled    = &cancelled,
            .event_writer = nullptr,
            .event_reader = nullptr,
  };
  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(exec_ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  holonp_test::OracleInput oi;
  oi.op             = "zeros";
  oi.n_outputs      = 1;
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(out_buf.download(), oracle.output_bytes[0], DType::F32);
}
