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
#include "holonp/conj.hh"

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

// -------------------------------------------------------------------------------------------------
// ConjFactory: inference tests
// -------------------------------------------------------------------------------------------------

class ConjInferTest : public ::testing::Test {
protected:
  holonp::ConjFactory factory;
};

TEST_F(ConjInferTest, U8DeviceContiguous) {
  const TDesc in = device_desc({8}, DType::U8);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.input_descs.size(), 1u);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{8}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(ConjInferTest, U16DeviceContiguous) {
  const TDesc in = device_desc({6}, DType::U16);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{6}));
}

TEST_F(ConjInferTest, F32DeviceContiguous) {
  const TDesc in = device_desc({5}, DType::F32);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{5}));
}

TEST_F(ConjInferTest, CF32DeviceContiguous) {
  const TDesc in = device_desc({4}, DType::CF32);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
}

TEST_F(ConjInferTest, RejectsZeroInputs) {
  const std::vector<TDesc> in;
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConjInferTest, RejectsTwoInputs) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32), device_desc({4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConjInferTest, RejectsHostMemLoc) {
  const TDesc in({4}, DType::F32, MemLoc::Host);
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConjInferTest, RejectsNonContiguousInput) {
  const TDesc in({2, 2}, DType::F32, MemLoc::Device, std::vector<size_t>{16, 4});
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ConjFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class ConjOracleTest : public ::testing::Test {
protected:
  holonp::ConjFactory factory;

  template <typename T>
  void check(DType dtype, const std::vector<size_t> &shape, const std::vector<T> &host_data) {
    const TDesc idesc  = device_desc(shape, dtype);
    const auto  ibytes = as_bytes(host_data);

    const auto run =
        holonp_test::run_sync_factory(factory, {&idesc, 1}, {&ibytes, 1}, nlohmann::json::object());

    holonp_test::OracleInput oi;
    oi.op             = "conj";
    oi.n_outputs      = 1;
    oi.input_descs    = {idesc};
    oi.input_bytes    = {ibytes};
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

    ASSERT_EQ(run.output_bytes.size(), 1u);
    ASSERT_EQ(oracle.output_bytes.size(), 1u);
    expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], dtype);
  }
};

TEST_F(ConjOracleTest, U8) {
  check(DType::U8, {6}, std::vector<std::uint8_t>{0, 1, 2, 3, 200, 255});
}

TEST_F(ConjOracleTest, U16) {
  check(DType::U16, {5}, std::vector<std::uint16_t>{0, 1, 256, 1024, 65535});
}

TEST_F(ConjOracleTest, F32TwoDim) {
  check(DType::F32, {2, 3}, std::vector<float>{1.f, -2.f, 3.5f, -4.f, 0.f, 6.f});
}

TEST_F(ConjOracleTest, CF32) {
  struct CF32 {
    float re, im;
  };
  check(DType::CF32, {4}, std::vector<CF32>{{1.f, 2.f}, {-3.f, 4.f}, {0.f, -5.f}, {2.5f, 0.25f}});
}

// -------------------------------------------------------------------------------------------------
// ConjFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class ConjUpdateTest : public ::testing::Test {
protected:
  holonp::ConjFactory factory;
};

TEST_F(ConjUpdateTest, ReusesConjTask) {
  const TDesc in     = device_desc({4}, DType::F32);
  const auto  ibytes = as_bytes(std::vector<float>{1.f, -2.f, 3.f, -4.f});

  const auto run = holonp_test::run_sync_factory_update(factory, {&in, 1}, {&ibytes, 1},
                                                        nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "conj";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(ConjUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc in     = device_desc({3}, DType::F32);
  const auto  ibytes = as_bytes(std::vector<float>{1.5f, -1.f, 2.f});

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), {&in, 1}, nlohmann::json::object(), create_ctx);

  const auto                    infer = factory.infer({&in, 1}, nlohmann::json::object());
  holonp_test::TensorTestBuffer in_buf(in);
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);
  in_buf.upload(ibytes);

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

  holonp_test::OracleInput oi;
  oi.op             = "conj";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  const auto actual = out_buf.download();
  expect_near_oracle(actual, oracle.output_bytes[0], DType::F32);
}
