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
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/abs.hh"

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

static TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// Element-wise comparison: exact for integer types, toleranced for F32.
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
  case DType::CF32:
    ADD_FAILURE() << "expect_near_oracle: CF32 should not appear as Abs output dtype";
    break;
  }
}

// -------------------------------------------------------------------------------------------------
// AbsFactory: inference tests
// -------------------------------------------------------------------------------------------------

class AbsInferTest : public ::testing::Test {
protected:
  holonp::AbsFactory factory;
};

TEST_F(AbsInferTest, U8DeviceContiguous) {
  const std::vector<TDesc> in = {device_desc({4}, DType::U8)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.input_descs.size(), 1u);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  ASSERT_EQ(r.owned_inputs.size(), 1u);
  EXPECT_EQ(r.owned_inputs[0], false);
  ASSERT_EQ(r.owned_outputs.size(), 1u);
  EXPECT_EQ(r.owned_outputs[0], false);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AbsInferTest, U16DeviceContiguous) {
  const std::vector<TDesc> in = {device_desc({8}, DType::U16)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{8}));
}

TEST_F(AbsInferTest, F32DeviceContiguous) {
  const std::vector<TDesc> in = {device_desc({3, 3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 3}));
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(AbsInferTest, CF32MapsToF32) {
  const std::vector<TDesc> in = {device_desc({6}, DType::CF32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{6}));
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(AbsInferTest, RejectsZeroInputs) {
  const std::vector<TDesc> in;
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AbsInferTest, RejectsTwoInputs) {
  const TDesc              d  = device_desc({4}, DType::F32);
  const std::vector<TDesc> in = {d, d};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AbsInferTest, RejectsHostMemLoc) {
  const std::vector<TDesc> in = {TDesc({4}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AbsInferTest, RejectsNonContiguousInput) {
  // stride 8 instead of 4 for F32 → non-contiguous
  const std::vector<TDesc> in = {TDesc({4}, DType::F32, MemLoc::Device, std::vector<size_t>{8})};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AbsInferTest, RejectsZeroElementInput) {
  const std::vector<TDesc> in = {device_desc({0}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// AbsFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class AbsOracleTest : public ::testing::Test {
protected:
  holonp::AbsFactory factory;

  // Helper: run factory + oracle and compare, for a single contiguous device tensor.
  template <typename T>
  void check(DType dtype, const std::vector<size_t> &shape, const std::vector<T> &host_input,
             DType out_dtype) {
    const TDesc                               idesc       = device_desc(shape, dtype);
    const auto                                ibytes      = as_bytes(host_input);
    const std::vector<TDesc>                  input_descs = {idesc};
    const std::vector<std::vector<std::byte>> input_data  = {ibytes};

    const auto run =
        holonp_test::run_sync_factory(factory, input_descs, input_data, nlohmann::json::object());

    holonp_test::OracleInput oi;
    oi.op             = "abs";
    oi.n_outputs      = 1;
    oi.input_descs    = {idesc};
    oi.input_bytes    = {ibytes};
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

    ASSERT_EQ(run.output_bytes.size(), 1u);
    ASSERT_EQ(oracle.output_bytes.size(), 1u);
    expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], out_dtype);
  }
};

TEST_F(AbsOracleTest, U8) {
  check(DType::U8, {4}, std::vector<std::uint8_t>{0, 1, 128, 255}, DType::U8);
}

TEST_F(AbsOracleTest, U16) {
  check(DType::U16, {4}, std::vector<std::uint16_t>{0, 1, 1000, 65535}, DType::U16);
}

TEST_F(AbsOracleTest, F32Positive) {
  check(DType::F32, {4}, std::vector<float>{0.0f, 1.5f, 3.14f, 100.0f}, DType::F32);
}

TEST_F(AbsOracleTest, F32Mixed) {
  check(DType::F32, {4}, std::vector<float>{-3.14f, 0.0f, 2.71f, -1.0f}, DType::F32);
}

TEST_F(AbsOracleTest, F32TwoDim) {
  check(DType::F32, {2, 3}, std::vector<float>{-1.f, 2.f, -3.f, 4.f, -5.f, 6.f}, DType::F32);
}

TEST_F(AbsOracleTest, CF32) {
  // cuFloatComplex and numpy complex64 both store (real, imag) pairs.
  // (1+2j), (-3+4j), (0+0j), (3-4j)  →  abs: sqrt(5), 5, 0, 5
  struct CF32 {
    float re, im;
  };
  const std::vector<CF32>  data        = {{1.f, 2.f}, {-3.f, 4.f}, {0.f, 0.f}, {3.f, -4.f}};
  const TDesc              idesc       = device_desc({4}, DType::CF32);
  const auto               ibytes      = as_bytes(data);
  const std::vector<TDesc> input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data = {ibytes};

  const auto run =
      holonp_test::run_sync_factory(factory, input_descs, input_data, nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "abs";
  oi.n_outputs      = 1;
  oi.input_descs    = {idesc};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  // CF32 input → F32 output
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32, 1e-5f);
}

// -------------------------------------------------------------------------------------------------
// AbsFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class AbsUpdateTest : public ::testing::Test {
protected:
  holonp::AbsFactory factory;
};

TEST_F(AbsUpdateTest, ReusesAbsTask) {
  // update() with an existing Abs task should reuse it and still produce correct results.
  const std::vector<float>                  host_data   = {-1.f, 2.f, -3.f, 4.f};
  const TDesc                               idesc       = device_desc({4}, DType::F32);
  const auto                                bytes       = as_bytes(host_data);
  const std::vector<TDesc>                  input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data  = {bytes};
  const nlohmann::json                      settings    = nlohmann::json::object();

  const auto run = holonp_test::run_sync_factory_update(factory, input_descs, input_data, settings);

  holonp_test::OracleInput oi;
  oi.op             = "abs";
  oi.n_outputs      = 1;
  oi.input_descs    = {idesc};
  oi.input_bytes    = {bytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(AbsUpdateTest, RecreatesOnWrongTaskType) {
  // Passing a non-Abs ISyncTask to update() must trigger the create path and still work.
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const std::vector<float> host_data = {-5.f, 0.f, 3.f};
  const TDesc              idesc     = device_desc({3}, DType::F32);
  const auto               bytes     = as_bytes(host_data);
  const std::vector<TDesc> descs     = {idesc};
  const nlohmann::json     settings  = nlohmann::json::object();

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), descs, settings, create_ctx);

  // Execute the recreated task and verify correctness.
  const auto infer = factory.infer(descs, settings);

  holonp_test::TensorTestBuffer in_buf(idesc);
  in_buf.upload(bytes);
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);

  auto                    iv = in_buf.view();
  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = {&iv, 1},
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };
  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  const auto actual = out_buf.download();

  holonp_test::OracleInput oi;
  oi.op             = "abs";
  oi.n_outputs      = 1;
  oi.input_descs    = {idesc};
  oi.input_bytes    = {bytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  expect_near_oracle(actual, oracle.output_bytes[0], DType::F32);
}
