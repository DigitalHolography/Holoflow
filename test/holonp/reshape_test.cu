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
#include "holonp/reshape.hh"

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

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

static nlohmann::json reshape_settings(std::vector<int64_t> shape) {
  return nlohmann::json{{"shape", shape}};
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

class ReshapeInferTest : public ::testing::Test {
protected:
  holonp::ReshapeFactory factory;
};

TEST_F(ReshapeInferTest, ViewPathForContiguousInput) {
  const TDesc in = device_desc({2, 3}, DType::F32);
  const auto  r  = factory.infer({&in, 1}, reshape_settings({3, 2}));
  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.in_place.size(), 1u);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 2}));
}

TEST_F(ReshapeInferTest, CopyPathWhenForced) {
  const TDesc in = device_desc({2, 3}, DType::F32);
  auto        j  = reshape_settings({3, 2});
  j["copy"]      = true;
  const auto r   = factory.infer({&in, 1}, j);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(ReshapeInferTest, RejectsInvalidShapeSpec) {
  const TDesc in = device_desc({2, 3}, DType::F32);
  EXPECT_THROW(factory.infer({&in, 1}, reshape_settings({-1, -1})), std::invalid_argument);
}

class ReshapeOracleTest : public ::testing::Test {
protected:
  holonp::ReshapeFactory factory;
};

TEST_F(ReshapeOracleTest, CopyModeMatchesOracle) {
  const TDesc in    = device_desc({2, 3}, DType::F32);
  auto        j     = reshape_settings({3, 2});
  j["copy"]         = true;
  const auto ibytes = as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6});
  const auto run    = holonp_test::run_sync_factory(factory, {&in, 1}, {&ibytes, 1}, j);

  holonp_test::OracleInput oi;
  oi.op             = "reshape";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

class ReshapeUpdateTest : public ::testing::Test {
protected:
  holonp::ReshapeFactory factory;
};

TEST_F(ReshapeUpdateTest, ReusesReshapeTaskOnSameConfig) {
  const TDesc in    = device_desc({2, 3}, DType::F32);
  auto        j     = reshape_settings({3, 2});
  j["copy"]         = true;
  const auto ibytes = as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6});
  const auto run    = holonp_test::run_sync_factory_update(factory, {&in, 1}, {&ibytes, 1}, j);

  holonp_test::OracleInput oi;
  oi.op             = "reshape";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(ReshapeUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };
  const TDesc in    = device_desc({2, 3}, DType::F32);
  auto        j     = reshape_settings({6});
  j["copy"]         = true;
  const auto ibytes = as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6});

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto       fake  = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto       task  = factory.update(std::move(fake), {&in, 1}, j, create_ctx);
  const auto infer = factory.infer({&in, 1}, j);

  holonp_test::TensorTestBuffer in_buf(in), out_buf(infer.output_descs[0]);
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
  oi.op             = "reshape";
  oi.n_outputs      = 1;
  oi.input_descs    = {in};
  oi.input_bytes    = {ibytes};
  oi.settings       = j;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(out_buf.download(), oracle.output_bytes[0], DType::F32);
}
