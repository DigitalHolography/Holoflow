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
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/empty.hh"

#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static nlohmann::json make_jsettings(std::vector<size_t> shape) {
  return nlohmann::json{{"shape", shape}};
}

class EmptyInferTest : public ::testing::Test {
protected:
  holonp::EmptyFactory factory;
};

TEST_F(EmptyInferTest, DefaultsF32Device) {
  const auto r = factory.infer({}, make_jsettings({2, 3}));
  EXPECT_EQ(r.kind, TaskKind::Sync);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(EmptyInferTest, RejectsBadInputOrOrderOrHost) {
  const std::vector<TDesc> in = {TDesc({1}, DType::F32, MemLoc::Device)};
  EXPECT_THROW(factory.infer(in, make_jsettings({1})), std::invalid_argument);
  auto j     = make_jsettings({1});
  j["order"] = "F";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
  j           = make_jsettings({1});
  j["device"] = "Host";
  EXPECT_THROW(factory.infer({}, j), std::invalid_argument);
}

class EmptyExecuteTest : public ::testing::Test {
protected:
  holonp::EmptyFactory factory;
};

TEST_F(EmptyExecuteTest, ExecuteIsNoOpOnOutputBytes) {
  const auto                          j = make_jsettings({4});
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  auto                                task  = factory.create({}, j, ctx);
  const auto                          infer = factory.infer({}, j);

  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);
  const std::vector<float>      sentinel = {9.0f, 8.0f, 7.0f, 6.0f};
  std::vector<std::byte>        bytes(sentinel.size() * sizeof(float));
  std::memcpy(bytes.data(), sentinel.data(), bytes.size());
  out_buf.upload(bytes);

  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx exec_ctx{
      .inputs       = {},
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };
  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(exec_ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  const auto after = out_buf.download();
  ASSERT_EQ(after.size(), bytes.size());
  EXPECT_EQ(std::memcmp(after.data(), bytes.data(), bytes.size()), 0);
}

class EmptyUpdateTest : public ::testing::Test {
protected:
  holonp::EmptyFactory factory;
};

TEST_F(EmptyUpdateTest, ReusesEmptyTaskOnSameSettings) {
  const auto                          j = make_jsettings({2});
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  auto                                task = factory.create({}, j, ctx);
  auto                                same = factory.update(std::move(task), {}, j, ctx);
  EXPECT_NE(same.get(), nullptr);
}

TEST_F(EmptyUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), {}, make_jsettings({1}), ctx);
  EXPECT_NE(task.get(), nullptr);
}
