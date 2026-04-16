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
#include "holonp/ascontiguousarray.hh"

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

static TDesc device_desc(std::vector<size_t> shape, DType dtype, std::vector<size_t> strides) {
  return TDesc(std::move(shape), dtype, MemLoc::Device, std::move(strides));
}

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

template <typename T>
static std::vector<std::byte> make_strided_2d_bytes(const std::vector<T> &logical, size_t rows,
                                                    size_t cols, size_t row_stride_bytes) {
  const size_t elem = sizeof(T);
  const size_t need = rows * cols;
  if (logical.size() != need) {
    throw std::invalid_argument("make_strided_2d_bytes: logical size mismatch");
  }

  std::vector<std::byte> out(rows * row_stride_bytes, std::byte{0});
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      const size_t src_idx = r * cols + c;
      const size_t dst_off = r * row_stride_bytes + c * elem;
      std::memcpy(out.data() + dst_off, &logical[src_idx], elem);
    }
  }
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

static void expect_matches_oracle(const std::vector<std::byte> &actual, const TDesc &idesc,
                                  const std::vector<std::byte> &input_bytes) {
  holonp_test::OracleInput oi;
  oi.op          = "ascontiguousarray";
  oi.n_outputs   = 1;
  oi.input_descs = {idesc};
  oi.input_bytes = {input_bytes};

  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_near_oracle(actual, oracle.output_bytes[0], idesc.dtype);
}

// -------------------------------------------------------------------------------------------------
// AsContiguousArrayFactory: inference tests
// -------------------------------------------------------------------------------------------------

class AsContiguousArrayInferTest : public ::testing::Test {
protected:
  holonp::AsContiguousArrayFactory factory;
};

TEST_F(AsContiguousArrayInferTest, ContiguousInputKeepsDescAndInPlace) {
  const TDesc in = device_desc({2, 3}, DType::U16);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.input_descs.size(), 1u);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, in.shape);
  EXPECT_EQ(r.output_descs[0].dtype, in.dtype);
  EXPECT_EQ(r.output_descs[0].mem_loc, in.mem_loc);
  EXPECT_EQ(r.output_descs[0].strides, in.strides);
  EXPECT_EQ(r.output_descs[0].offset, in.offset);
  ASSERT_EQ(r.in_place.size(), 1u);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
}

TEST_F(AsContiguousArrayInferTest, NonContiguousInputProducesContiguousOutput) {
  const TDesc in = device_desc({2, 3}, DType::U16, {8, 2});
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());

  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, in.shape);
  EXPECT_EQ(r.output_descs[0].dtype, in.dtype);
  EXPECT_EQ(r.output_descs[0].mem_loc, in.mem_loc);
  EXPECT_EQ(r.output_descs[0].strides, (std::vector<size_t>{6, 2}));
  EXPECT_EQ(r.output_descs[0].offset, 0u);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AsContiguousArrayInferTest, RejectsZeroInputs) {
  const std::vector<TDesc> in;
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AsContiguousArrayInferTest, RejectsTwoInputs) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 2}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AsContiguousArrayInferTest, RejectsHostInput) {
  const std::vector<TDesc> in = {TDesc({2, 2}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// AsContiguousArrayFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class AsContiguousArrayExecuteTest : public ::testing::Test {
protected:
  holonp::AsContiguousArrayFactory factory;
};

TEST_F(AsContiguousArrayExecuteTest, U16Strided2DCopy) {
  const TDesc in =
      device_desc({2, 3}, DType::U16, {8, 2}); // row stride 8 bytes, 2 bytes trailing pad per row
  const auto input = make_strided_2d_bytes<std::uint16_t>({10, 20, 30, 40, 50, 60}, 2, 3, 8);
  const auto run =
      holonp_test::run_sync_factory(factory, {&in, 1}, {&input, 1}, nlohmann::json::object());

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], in, input);
}

TEST_F(AsContiguousArrayExecuteTest, F32Strided2DCopy) {
  const TDesc in =
      device_desc({2, 2}, DType::F32, {16, 4}); // row stride 16 bytes, 8 bytes trailing pad per row
  const auto input = make_strided_2d_bytes<float>({1.25f, -2.5f, 3.75f, 4.5f}, 2, 2, 16);
  const auto run =
      holonp_test::run_sync_factory(factory, {&in, 1}, {&input, 1}, nlohmann::json::object());

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], in, input);
}

TEST_F(AsContiguousArrayExecuteTest, OutputDescMatchesInferForNonContiguousInput) {
  const TDesc in  = device_desc({2, 3}, DType::U8, {4, 1});
  const auto  raw = make_strided_2d_bytes<std::uint8_t>({1, 2, 3, 4, 5, 6}, 2, 3, 4);
  const auto  run =
      holonp_test::run_sync_factory(factory, {&in, 1}, {&raw, 1}, nlohmann::json::object());

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::U8);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_EQ(run.output_descs[0].strides, (std::vector<size_t>{3, 1}));
  EXPECT_EQ(run.output_descs[0].offset, 0u);
}

// -------------------------------------------------------------------------------------------------
// AsContiguousArrayFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class AsContiguousArrayUpdateTest : public ::testing::Test {
protected:
  holonp::AsContiguousArrayFactory factory;
};

TEST_F(AsContiguousArrayUpdateTest, ReusesTaskOnSameNonContiguousDesc) {
  const TDesc in    = device_desc({2, 3}, DType::U16, {8, 2});
  const auto  input = make_strided_2d_bytes<std::uint16_t>({7, 8, 9, 10, 11, 12}, 2, 3, 8);
  const auto  run   = holonp_test::run_sync_factory_update(factory, {&in, 1}, {&input, 1},
                                                           nlohmann::json::object());

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_matches_oracle(run.output_bytes[0], in, input);
}

TEST_F(AsContiguousArrayUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc in    = device_desc({2, 2}, DType::F32, {16, 4});
  const auto  input = make_strided_2d_bytes<float>({5.0f, -6.0f, 7.0f, 8.5f}, 2, 2, 16);

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), {&in, 1}, nlohmann::json::object(), create_ctx);

  const auto infer = factory.infer({&in, 1}, nlohmann::json::object());

  holonp_test::TensorTestBuffer in_buf(in);
  in_buf.upload(input);
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
  expect_matches_oracle(actual, in, input);
}
