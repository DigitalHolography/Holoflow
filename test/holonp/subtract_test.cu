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
#include "holonp/subtract.hh"

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

class SubtractInferTest : public ::testing::Test {
protected:
  holonp::SubtractFactory factory;
};

TEST_F(SubtractInferTest, F32Broadcast) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::F32), device_desc({3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(SubtractInferTest, RejectsBadInputs) {
  const std::vector<TDesc> none;
  EXPECT_THROW(factory.infer(none, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> one = {device_desc({2}, DType::F32)};
  EXPECT_THROW(factory.infer(one, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> mismatch = {device_desc({2}, DType::F32), device_desc({2}, DType::U16)};
  EXPECT_THROW(factory.infer(mismatch, nlohmann::json::object()), std::invalid_argument);
}

class SubtractOracleTest : public ::testing::Test {
protected:
  holonp::SubtractFactory factory;

  template <typename T>
  void check(DType dtype, const std::vector<size_t> &shape_a, const std::vector<T> &a,
             const std::vector<size_t> &shape_b, const std::vector<T> &b) {
    const TDesc                               da          = device_desc(shape_a, dtype);
    const TDesc                               db          = device_desc(shape_b, dtype);
    const auto                                ba          = as_bytes(a);
    const auto                                bb          = as_bytes(b);
    const std::vector<TDesc>                  input_descs = {da, db};
    const std::vector<std::vector<std::byte>> input_data  = {ba, bb};
    const auto                                run =
        holonp_test::run_sync_factory(factory, input_descs, input_data, nlohmann::json::object());

    holonp_test::OracleInput oi;
    oi.op             = "subtract";
    oi.n_outputs      = 1;
    oi.input_descs    = input_descs;
    oi.input_bytes    = input_data;
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
    expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], dtype);
  }
};

TEST_F(SubtractOracleTest, F32SameShape) {
  check(DType::F32, {4}, std::vector<float>{4.f, 3.f, 2.f, 1.f}, {4},
        std::vector<float>{1.f, 1.f, 1.f, 1.f});
}

TEST_F(SubtractOracleTest, F32Broadcast) {
  check(DType::F32, {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, {3}, std::vector<float>{1, 2, 3});
}

TEST_F(SubtractOracleTest, U8SameShape) {
  check(DType::U8, {4}, std::vector<std::uint8_t>{10, 20, 30, 40}, {4},
        std::vector<std::uint8_t>{1, 2, 3, 4});
}

TEST_F(SubtractOracleTest, CF32SameShape) {
  struct CF32 {
    float re, im;
  };
  check(DType::CF32, {3}, std::vector<CF32>{{1, 2}, {3, 4}, {5, 6}}, {3},
        std::vector<CF32>{{1, 1}, {1, 1}, {1, 1}});
}

class SubtractUpdateTest : public ::testing::Test {
protected:
  holonp::SubtractFactory factory;
};

TEST_F(SubtractUpdateTest, ReusesSubtractTask) {
  const TDesc                               da    = device_desc({3}, DType::F32);
  const TDesc                               db    = device_desc({3}, DType::F32);
  const std::vector<TDesc>                  descs = {da, db};
  const std::vector<std::vector<std::byte>> data  = {
      as_bytes(std::vector<float>{2.f, 4.f, 6.f}),
      as_bytes(std::vector<float>{1.f, 2.f, 3.f}),
  };
  const auto run =
      holonp_test::run_sync_factory_update(factory, descs, data, nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "subtract";
  oi.n_outputs      = 1;
  oi.input_descs    = descs;
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(SubtractUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };
  const TDesc                  da = device_desc({2}, DType::F32);
  const TDesc                  db = device_desc({2}, DType::F32);
  const std::vector<std::byte> ba = as_bytes(std::vector<float>{5.f, 6.f});
  const std::vector<std::byte> bb = as_bytes(std::vector<float>{1.f, 2.f});

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), std::vector<TDesc>{da, db}, nlohmann::json::object(),
                             create_ctx);
  const auto infer = factory.infer(std::vector<TDesc>{da, db}, nlohmann::json::object());

  holonp_test::TensorTestBuffer a_buf(da), b_buf(db), out_buf(infer.output_descs[0]);
  a_buf.upload(ba);
  b_buf.upload(bb);
  auto                                 va       = a_buf.view();
  auto                                 vb       = b_buf.view();
  auto                                 ov       = out_buf.view();
  std::array<holoflow::core::TView, 2> in_views = {va, vb};
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
  oi.op             = "subtract";
  oi.n_outputs      = 1;
  oi.input_descs    = {da, db};
  oi.input_bytes    = {ba, bb};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_near_oracle(out_buf.download(), oracle.output_bytes[0], DType::F32);
}
