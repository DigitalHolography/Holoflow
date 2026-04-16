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
#include "holonp/where.hh"

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

class WhereInferTest : public ::testing::Test {
protected:
  holonp::WhereFactory factory;
};

TEST_F(WhereInferTest, BroadcastShapeAndOutputType) {
  const std::vector<TDesc> in = {device_desc({2, 1}, DType::U8), device_desc({2, 3}, DType::F32),
                                 device_desc({3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(WhereInferTest, RejectsInvalidInputTypesOrCount) {
  const std::vector<TDesc> bad_cond = {device_desc({2}, DType::F32), device_desc({2}, DType::F32),
                                       device_desc({2}, DType::F32)};
  EXPECT_THROW(factory.infer(bad_cond, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> mismatch = {device_desc({2}, DType::U8), device_desc({2}, DType::U16),
                                       device_desc({2}, DType::F32)};
  EXPECT_THROW(factory.infer(mismatch, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> none;
  EXPECT_THROW(factory.infer(none, nlohmann::json::object()), std::invalid_argument);
}

class WhereOracleTest : public ::testing::Test {
protected:
  holonp::WhereFactory factory;
};

TEST_F(WhereOracleTest, F32Broadcast) {
  const TDesc                               cond = device_desc({2, 1}, DType::U8);
  const TDesc                               x    = device_desc({2, 3}, DType::F32);
  const TDesc                               y    = device_desc({3}, DType::F32);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 0}),
      as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6}),
      as_bytes(std::vector<float>{10, 20, 30}),
  };
  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{cond, x, y}, data,
                                                 nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "where";
  oi.n_outputs      = 1;
  oi.input_descs    = {cond, x, y};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(WhereOracleTest, U16SameShape) {
  const TDesc                               cond = device_desc({4}, DType::U8);
  const TDesc                               x    = device_desc({4}, DType::U16);
  const TDesc                               y    = device_desc({4}, DType::U16);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 0, 1, 0}),
      as_bytes(std::vector<std::uint16_t>{1, 2, 3, 4}),
      as_bytes(std::vector<std::uint16_t>{10, 20, 30, 40}),
  };
  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{cond, x, y}, data,
                                                 nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "where";
  oi.n_outputs      = 1;
  oi.input_descs    = {cond, x, y};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::U16);
}

class WhereUpdateTest : public ::testing::Test {
protected:
  holonp::WhereFactory factory;
};

TEST_F(WhereUpdateTest, ReusesWhereTask) {
  const TDesc                               cond = device_desc({3}, DType::U8);
  const TDesc                               x    = device_desc({3}, DType::F32);
  const TDesc                               y    = device_desc({3}, DType::F32);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 0, 1}),
      as_bytes(std::vector<float>{1, 2, 3}),
      as_bytes(std::vector<float>{10, 20, 30}),
  };
  const auto run = holonp_test::run_sync_factory_update(factory, std::vector<TDesc>{cond, x, y},
                                                        data, nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "where";
  oi.n_outputs      = 1;
  oi.input_descs    = {cond, x, y};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(WhereUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc cond = device_desc({2}, DType::U8);
  const TDesc x    = device_desc({2}, DType::F32);
  const TDesc y    = device_desc({2}, DType::F32);
  const auto  bc   = as_bytes(std::vector<std::uint8_t>{1, 0});
  const auto  bx   = as_bytes(std::vector<float>{1, 2});
  const auto  by   = as_bytes(std::vector<float>{10, 20});

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto       fake  = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto       task  = factory.update(std::move(fake), std::vector<TDesc>{cond, x, y},
                                    nlohmann::json::object(), create_ctx);
  const auto infer = factory.infer(std::vector<TDesc>{cond, x, y}, nlohmann::json::object());

  holonp_test::TensorTestBuffer c_buf(cond), x_buf(x), y_buf(y), out_buf(infer.output_descs[0]);
  c_buf.upload(bc);
  x_buf.upload(bx);
  y_buf.upload(by);
  auto                                 vc       = c_buf.view();
  auto                                 vx       = x_buf.view();
  auto                                 vy       = y_buf.view();
  auto                                 ov       = out_buf.view();
  std::array<holoflow::core::TView, 3> in_views = {vc, vx, vy};
  std::atomic<bool>                    cancelled{false};
  holoflow::core::SyncCtx              ctx{
                   .inputs       = {in_views.data(), in_views.size()},
                   .outputs      = {&ov, 1},
                   .cancelled    = &cancelled,
                   .event_writer = nullptr,
                   .event_reader = nullptr,
  };
  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  holonp_test::OracleInput oi;
  oi.op             = "where";
  oi.n_outputs      = 1;
  oi.input_descs    = {cond, x, y};
  oi.input_bytes    = {bc, bx, by};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(out_buf.download(), oracle.output_bytes[0], DType::F32);
}
