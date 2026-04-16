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
#include "holonp/transpose.hh"

#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

class TransposeInferTest : public ::testing::Test {
protected:
  holonp::TransposeFactory factory;
};

TEST_F(TransposeInferTest, PermutesShapeAndStridesWithInPlace) {
  const TDesc in = device_desc({2, 3}, DType::F32);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json{{"axes", std::vector<int>{1, 0}}});
  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 2}));
  EXPECT_EQ(r.output_descs[0].strides, (std::vector<size_t>{4, 12}));
  ASSERT_EQ(r.in_place.size(), 1u);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
}

TEST_F(TransposeInferTest, SupportsNegativeAxes) {
  const TDesc in = device_desc({2, 3, 4}, DType::U8);
  const auto  r  = factory.infer({&in, 1}, nlohmann::json{{"axes", std::vector<int>{0, -1, -2}}});
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 4, 3}));
}

TEST_F(TransposeInferTest, RejectsInvalidAxes) {
  const TDesc in = device_desc({2, 3}, DType::F32);
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json{{"axes", std::vector<int>{0}}}),
               std::invalid_argument);
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json{{"axes", std::vector<int>{0, 2}}}),
               std::invalid_argument);
  EXPECT_THROW(factory.infer({&in, 1}, nlohmann::json{{"axes", std::vector<int>{0, 0}}}),
               std::invalid_argument);
}

class TransposeUpdateTest : public ::testing::Test {
protected:
  holonp::TransposeFactory factory;
};

TEST_F(TransposeUpdateTest, ReusesOnValidTask) {
  const TDesc                         in = device_desc({2, 3}, DType::F32);
  const auto                          j  = nlohmann::json{{"axes", std::vector<int>{1, 0}}};
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};
  auto                                task    = factory.create({&in, 1}, j, ctx);
  auto                                updated = factory.update(std::move(task), {&in, 1}, j, ctx);
  EXPECT_NE(updated.get(), nullptr);
}

TEST_F(TransposeUpdateTest, RecreatesOnWrongTaskTypeAndExecuteNoOp) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc                         in = device_desc({2, 2}, DType::F32);
  const auto                          j  = nlohmann::json{{"axes", std::vector<int>{1, 0}}};
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), {&in, 1}, j, create_ctx);

  const auto                    infer = factory.infer({&in, 1}, j);
  holonp_test::TensorTestBuffer in_buf(in), out_buf(infer.output_descs[0]);

  auto                    iv = in_buf.view();
  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx exec_ctx{
      .inputs       = {&iv, 1},
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };
  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(exec_ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
}
