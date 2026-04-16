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
#include "holonp/copy.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

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
  std::vector<std::byte> out(rows * row_stride_bytes, std::byte{0});
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      const size_t src_idx = r * cols + c;
      const size_t dst_off = r * row_stride_bytes + c * sizeof(T);
      std::memcpy(out.data() + dst_off, &logical[src_idx], sizeof(T));
    }
  }
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

class CopyInferTest : public ::testing::Test {
protected:
  holonp::CopyFactory factory;
};

TEST_F(CopyInferTest, DeviceInputProducesContiguousOutput) {
  const TDesc in = device_desc({2, 3}, DType::U16, {8, 2});
  const auto  r  = factory.infer({&in, 1}, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(r.output_descs[0].strides, (std::vector<size_t>{6, 2}));
}

TEST_F(CopyInferTest, RejectsWrongInputCount) {
  const std::vector<TDesc> none;
  EXPECT_THROW(factory.infer(none, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> two = {device_desc({1}, DType::U8), device_desc({1}, DType::U8)};
  EXPECT_THROW(factory.infer(two, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(CopyInferTest, RejectsHostInput) {
  const TDesc in({2}, DType::F32, MemLoc::Host);
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json::object()), std::invalid_argument);
}

class CopyOracleTest : public ::testing::Test {
protected:
  holonp::CopyFactory factory;
};

TEST_F(CopyOracleTest, U8Contiguous) {
  const TDesc in     = device_desc({6}, DType::U8);
  const auto  ibytes = as_bytes(std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6});
  const auto  run =
      holonp_test::run_sync_factory(factory, {&in, 1}, {&ibytes, 1}, nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "copy";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::U8);
}

TEST_F(CopyOracleTest, F32Strided) {
  const TDesc in     = device_desc({2, 2}, DType::F32, {16, 4});
  const auto  ibytes = make_strided_2d_bytes<float>({1.5f, -2.0f, 3.25f, 4.5f}, 2, 2, 16);
  const auto  run =
      holonp_test::run_sync_factory(factory, {&in, 1}, {&ibytes, 1}, nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "copy";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

class CopyUpdateTest : public ::testing::Test {
protected:
  holonp::CopyFactory factory;
};

TEST_F(CopyUpdateTest, ReusesCopyTask) {
  const TDesc in     = device_desc({4}, DType::U16);
  const auto  ibytes = as_bytes(std::vector<std::uint16_t>{10, 20, 30, 40});
  const auto  run    = holonp_test::run_sync_factory_update(factory, {&in, 1}, {&ibytes, 1},
                                                            nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "copy";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::U16);
}

TEST_F(CopyUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc                         in     = device_desc({3}, DType::F32);
  const auto                          ibytes = as_bytes(std::vector<float>{1.f, 2.f, 3.f});
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};

  auto       fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto       task = factory.update(std::move(fake), {&in, 1}, nlohmann::json::object(), create_ctx);
  const auto infer = factory.infer({&in, 1}, nlohmann::json::object());

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
  oi.op             = "copy";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  const auto actual = out_buf.download();
  expect_near_oracle(actual, oracle.output_bytes[0], DType::F32);
}
