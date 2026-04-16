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
#include "holonp/asarray.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

// Absolute path to oracle.py, baked in at compile time.
static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

static nlohmann::json make_jsettings(double value) { return nlohmann::json{{"value", value}}; }

static void expect_near_oracle(const std::vector<std::byte> &actual,
                               const std::vector<std::byte> &expected, float rtol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(actual.size(), sizeof(float));

  const auto *a   = reinterpret_cast<const float *>(actual.data());
  const auto *e   = reinterpret_cast<const float *>(expected.data());
  const float tol = rtol * std::max(std::abs(e[0]), 1.0f);
  EXPECT_NEAR(a[0], e[0], tol);
}

static void expect_matches_oracle(const std::vector<std::byte> &actual,
                                  const nlohmann::json         &settings) {
  holonp_test::OracleInput oi;
  oi.op        = "asarray";
  oi.n_outputs = 1;
  oi.settings  = settings;

  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_near_oracle(actual, oracle.output_bytes[0]);
}

// -------------------------------------------------------------------------------------------------
// AsArrayFactory: inference tests
// -------------------------------------------------------------------------------------------------

class AsArrayInferTest : public ::testing::Test {
protected:
  holonp::AsArrayFactory factory;
};

TEST_F(AsArrayInferTest, DefaultsF32Device) {
  const auto r = factory.infer({}, make_jsettings(2.5));

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_TRUE(r.input_descs.empty());
  EXPECT_TRUE(r.owned_inputs.empty());
  ASSERT_EQ(r.owned_outputs.size(), 1u);
  EXPECT_EQ(r.owned_outputs[0], false);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AsArrayInferTest, ExplicitDevice) {
  auto j       = make_jsettings(3.0);
  j["device"]  = "Device";
  const auto r = factory.infer({}, j);

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(AsArrayInferTest, RejectsNonEmptyInputs) {
  const std::vector<TDesc> in = {TDesc({1}, DType::F32, MemLoc::Device)};
  EXPECT_THROW(factory.infer(in, make_jsettings(1.0)), std::invalid_argument);
}

TEST_F(AsArrayInferTest, RejectsHostDevice) {
  auto j      = make_jsettings(1.0);
  j["device"] = "Host";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// AsArrayFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class AsArrayExecuteTest : public ::testing::Test {
protected:
  holonp::AsArrayFactory                    factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(AsArrayExecuteTest, ZeroValue) {
  const auto j   = make_jsettings(0.0);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], j);
}

TEST_F(AsArrayExecuteTest, PositiveValue) {
  const auto j   = make_jsettings(123.5);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], j);
}

TEST_F(AsArrayExecuteTest, NegativeFractionalValue) {
  const auto j   = make_jsettings(-0.125);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], j);
}

TEST_F(AsArrayExecuteTest, OutputDescMatchesInfer) {
  const auto j   = make_jsettings(7.0);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{1}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Device);
}

// -------------------------------------------------------------------------------------------------
// AsArrayFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class AsArrayUpdateTest : public ::testing::Test {
protected:
  holonp::AsArrayFactory                    factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(AsArrayUpdateTest, ReusesAsArrayTask) {
  const auto j   = make_jsettings(42.0);
  const auto run = holonp_test::run_sync_factory_update(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], j);
}

TEST_F(AsArrayUpdateTest, RecreatesOnChangedSettings) {
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};

  const auto j_old = make_jsettings(1.0);
  const auto j_new = make_jsettings(-2.5);

  auto task = factory.create(no_inputs, j_old, ctx);
  task      = factory.update(std::move(task), no_inputs, j_new, ctx);

  const auto infer = factory.infer(no_inputs, j_new);

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

  const auto actual = out_buf.download();
  expect_matches_oracle(actual, j_new);
}

TEST_F(AsArrayUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  const auto                          j = make_jsettings(3.25);

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), no_inputs, j, ctx);

  const auto infer = factory.infer(no_inputs, j);

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

  const auto actual = out_buf.download();
  expect_matches_oracle(actual, j);
}
