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
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/arange.hh"

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

// Build JSON settings for ArangeFactory.  dtype and device are left absent when not provided so
// that the factory uses its defaults (F32, Device).
static nlohmann::json make_jsettings(double start, double stop, double step) {
  return nlohmann::json{{"start", start}, {"stop", stop}, {"step", step}};
}

static nlohmann::json make_jsettings(double start, double stop, double step,
                                     const std::string &dtype) {
  auto j     = make_jsettings(start, stop, step);
  j["dtype"] = dtype;
  return j;
}

// Element-wise comparison: exact for integer types, toleranced for F32/CF32.
static void expect_near_oracle(const std::vector<std::byte> &actual,
                               const std::vector<std::byte> &expected, DType dtype,
                               float rtol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t n = actual.size() / holoflow::core::size_of(dtype);

  switch (dtype) {
  case DType::U8: {
    const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    break;
  }
  case DType::U16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    break;
  }
  case DType::F32: {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
    }
    break;
  }
  case DType::CF32: {
    // CF32 stores interleaved (real, imag) float pairs; compare each component.
    const size_t n_floats = actual.size() / sizeof(float);
    const auto  *a        = reinterpret_cast<const float *>(actual.data());
    const auto  *e        = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n_floats; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol) << "at float component " << i;
    }
    break;
  }
  }
}

static void expect_matches_oracle(const std::vector<std::byte> &actual, DType dtype,
                                  const nlohmann::json &settings) {
  holonp_test::OracleInput oi;
  oi.op        = "arange";
  oi.n_outputs = 1;
  oi.settings  = settings;

  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_near_oracle(actual, oracle.output_bytes[0], dtype);
}

// -------------------------------------------------------------------------------------------------
// ArangeFactory: inference tests
// -------------------------------------------------------------------------------------------------

class ArangeInferTest : public ::testing::Test {
protected:
  holonp::ArangeFactory factory;
};

TEST_F(ArangeInferTest, DefaultsF32Device) {
  const auto r = factory.infer({}, make_jsettings(0.0, 4.0, 1.0));

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_TRUE(r.input_descs.empty());
  EXPECT_TRUE(r.owned_inputs.empty());
  ASSERT_EQ(r.owned_outputs.size(), 1u);
  EXPECT_EQ(r.owned_outputs[0], false);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(ArangeInferTest, ExplicitU8) {
  const auto r = factory.infer({}, make_jsettings(0.0, 3.0, 1.0, "U8"));

  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3}));
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(ArangeInferTest, ExplicitU16) {
  const auto r = factory.infer({}, make_jsettings(0.0, 5.0, 1.0, "U16"));

  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{5}));
}

TEST_F(ArangeInferTest, ExplicitCF32) {
  const auto r = factory.infer({}, make_jsettings(0.0, 4.0, 1.0, "CF32"));

  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
}

TEST_F(ArangeInferTest, FractionalStep) {
  // 0.0, 0.25, 0.5, 0.75 → 4 elements
  const auto r = factory.infer({}, make_jsettings(0.0, 1.0, 0.25));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(ArangeInferTest, NegativeStep) {
  // 4.0, 3.0, 2.0, 1.0 → 4 elements
  const auto r = factory.infer({}, make_jsettings(4.0, 0.0, -1.0));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(ArangeInferTest, EmptyRange) {
  // stop == start with positive step → 0 elements
  const auto r = factory.infer({}, make_jsettings(2.0, 2.0, 1.0));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{0}));
}

TEST_F(ArangeInferTest, LargeRange) {
  // 1000 elements
  const auto r = factory.infer({}, make_jsettings(0.0, 1000.0, 1.0));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1000}));
}

TEST_F(ArangeInferTest, RejectsNonEmptyInputs) {
  const std::vector<TDesc> in = {TDesc({4}, DType::F32, MemLoc::Device)};
  EXPECT_THROW(factory.infer(in, make_jsettings(0.0, 4.0, 1.0)), std::invalid_argument);
}

TEST_F(ArangeInferTest, RejectsZeroStep) {
  EXPECT_THROW(factory.infer({}, make_jsettings(0.0, 4.0, 0.0)), std::invalid_argument);
}

TEST_F(ArangeInferTest, RejectsHostDevice) {
  auto j      = make_jsettings(0.0, 4.0, 1.0);
  j["device"] = "Host";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ArangeFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class ArangeExecuteTest : public ::testing::Test {
protected:
  holonp::ArangeFactory                     factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(ArangeExecuteTest, F32Simple) {
  // 0.0, 1.0, 2.0, 3.0
  const auto j   = make_jsettings(0.0, 4.0, 1.0);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::F32, j);
}

TEST_F(ArangeExecuteTest, F32FractionalStep) {
  // 0.0, 0.25, 0.5, 0.75
  const auto j   = make_jsettings(0.0, 1.0, 0.25);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::F32, j);
}

TEST_F(ArangeExecuteTest, F32NegativeStep) {
  // 4.0, 3.0, 2.0, 1.0
  const auto j   = make_jsettings(4.0, 0.0, -1.0);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::F32, j);
}

TEST_F(ArangeExecuteTest, F32NonZeroStart) {
  // 2.0, 4.0, 6.0
  const auto j   = make_jsettings(2.0, 8.0, 2.0);
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::F32, j);
}

TEST_F(ArangeExecuteTest, U8Simple) {
  // 0, 1, 2, 3, 4
  const auto j   = make_jsettings(0.0, 5.0, 1.0, "U8");
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::U8, j);
}

TEST_F(ArangeExecuteTest, U16Simple) {
  // 100, 200, 300
  const auto j   = make_jsettings(100.0, 400.0, 100.0, "U16");
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::U16, j);
}

TEST_F(ArangeExecuteTest, CF32Simple) {
  // (0,0), (1,0), (2,0), (3,0)
  const auto j   = make_jsettings(0.0, 4.0, 1.0, "CF32");
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::CF32, j);
}

TEST_F(ArangeExecuteTest, CF32FractionalStep) {
  // (0.0,0), (0.5,0), (1.0,0), (1.5,0)
  const auto j   = make_jsettings(0.0, 2.0, 0.5, "CF32");
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::CF32, j);
}

TEST_F(ArangeExecuteTest, OutputDescMatchesInfer) {
  const auto j   = make_jsettings(0.0, 6.0, 1.0, "U16");
  const auto run = holonp_test::run_sync_factory(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{6}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Device);
}

// -------------------------------------------------------------------------------------------------
// ArangeFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class ArangeUpdateTest : public ::testing::Test {
protected:
  holonp::ArangeFactory                     factory;
  const std::vector<TDesc>                  no_inputs = {};
  const std::vector<std::vector<std::byte>> no_data   = {};
};

TEST_F(ArangeUpdateTest, ReusesArangeTask) {
  // update() with the same settings should reuse the task and still produce the correct result.
  const auto j   = make_jsettings(0.0, 4.0, 1.0);
  const auto run = holonp_test::run_sync_factory_update(factory, no_inputs, no_data, j);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], DType::F32, j);
}

TEST_F(ArangeUpdateTest, RecreatesOnChangedSettings) {
  // Create with one range, update with a different range; result must match the new settings.
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};

  const auto j_old = make_jsettings(0.0, 4.0, 1.0);
  const auto j_new = make_jsettings(10.0, 13.0, 1.0);

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
  expect_matches_oracle(actual, DType::F32, j_new);
}

TEST_F(ArangeUpdateTest, RecreatesOnWrongTaskType) {
  // Passing a non-Arange ISyncTask to update() must trigger the create path and still work.
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  const auto                          j = make_jsettings(0.0, 3.0, 1.0);

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
  expect_matches_oracle(actual, DType::F32, j);
}
