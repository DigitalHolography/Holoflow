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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/equal.hh"

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

static void expect_eq_u8(const std::vector<std::byte> &actual,
                         const std::vector<std::byte> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
  const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(a[i], e[i]);
}

class EqualInferTest : public ::testing::Test {
protected:
  holonp::EqualFactory factory;
};

TEST_F(EqualInferTest, BroadcastOutputIsU8) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::F32), device_desc({3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
}

TEST_F(EqualInferTest, RejectsBadInputs) {
  const std::vector<TDesc> none;
  EXPECT_THROW(factory.infer(none, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> mismatch = {device_desc({2}, DType::F32), device_desc({2}, DType::U16)};
  EXPECT_THROW(factory.infer(mismatch, nlohmann::json::object()), std::invalid_argument);
  const std::vector<TDesc> host = {TDesc({2}, DType::F32, MemLoc::Host),
                                   device_desc({2}, DType::F32)};
  EXPECT_THROW(factory.infer(host, nlohmann::json::object()), std::invalid_argument);
}

class EqualOracleTest : public ::testing::Test {
protected:
  holonp::EqualFactory factory;
};

TEST_F(EqualOracleTest, F32Broadcast) {
  const TDesc                               a    = device_desc({2, 3}, DType::F32);
  const TDesc                               b    = device_desc({3}, DType::F32);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6}),
      as_bytes(std::vector<float>{1, 0, 3}),
  };
  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{a, b}, data,
                                                 nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "equal";
  oi.n_outputs      = 1;
  oi.input_descs    = {a, b};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_eq_u8(run.output_bytes[0], oracle.output_bytes[0]);
}

TEST_F(EqualOracleTest, U16SameShape) {
  const TDesc                               a    = device_desc({4}, DType::U16);
  const TDesc                               b    = device_desc({4}, DType::U16);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint16_t>{10, 20, 30, 40}),
      as_bytes(std::vector<std::uint16_t>{10, 0, 30, 1}),
  };
  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{a, b}, data,
                                                 nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "equal";
  oi.n_outputs      = 1;
  oi.input_descs    = {a, b};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_eq_u8(run.output_bytes[0], oracle.output_bytes[0]);
}

class EqualUpdateTest : public ::testing::Test {
protected:
  holonp::EqualFactory factory;
};

TEST_F(EqualUpdateTest, ReusesEqualTask) {
  const TDesc                               a    = device_desc({3}, DType::U8);
  const TDesc                               b    = device_desc({3}, DType::U8);
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 2, 3}),
      as_bytes(std::vector<std::uint8_t>{1, 0, 3}),
  };
  const auto run = holonp_test::run_sync_factory_update(factory, std::vector<TDesc>{a, b}, data,
                                                        nlohmann::json::object());
  holonp_test::OracleInput oi;
  oi.op             = "equal";
  oi.n_outputs      = 1;
  oi.input_descs    = {a, b};
  oi.input_bytes    = data;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_eq_u8(run.output_bytes[0], oracle.output_bytes[0]);
}

TEST_F(EqualUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };
  const TDesc a  = device_desc({2}, DType::F32);
  const TDesc b  = device_desc({2}, DType::F32);
  const auto  ba = as_bytes(std::vector<float>{1.f, 2.f});
  const auto  bb = as_bytes(std::vector<float>{1.f, 0.f});

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), std::vector<TDesc>{a, b}, nlohmann::json::object(),
                             create_ctx);
  const auto infer = factory.infer(std::vector<TDesc>{a, b}, nlohmann::json::object());

  holonp_test::TensorTestBuffer a_buf(a), b_buf(b), out_buf(infer.output_descs[0]);
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
  oi.op             = "equal";
  oi.n_outputs      = 1;
  oi.input_descs    = {a, b};
  oi.input_bytes    = {ba, bb};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);
  expect_eq_u8(out_buf.download(), oracle.output_bytes[0]);
}
