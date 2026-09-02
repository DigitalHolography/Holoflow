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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/asyncs/dual_reader_batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/syncs/causal_sliding_average.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::Storage;
using holoflow::core::TDesc;
using holoflow::core::TView;

class TestStorageAccess final : public holoflow::core::IOStorageAccess {
public:
  TestStorageAccess(std::span<const TDesc> inputs, std::span<const TDesc> outputs) {
    for (const auto &desc : inputs) {
      inputs_.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
    }
    for (const auto &desc : outputs) {
      outputs_.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
    }
  }

  Storage &owned_input_storage(size_t index) override { return inputs_.at(index); }
  Storage &owned_output_storage(size_t index) override { return outputs_.at(index); }

private:
  std::vector<Storage> inputs_;
  std::vector<Storage> outputs_;
};

TEST(CausalSlidingAverageTest, EmitsPartialThenFullAveragesForArbitraryRank) {
  const TDesc input_desc({1, 1, 1, 1, 1}, DType::F32, MemLoc::Device);
  const holotask::syncs::CausalSlidingAverageSettings settings{3};
  holotask::syncs::CausalSlidingAverageFactory        factory;
  const std::array                                    input_descs{input_desc};
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));

  curaii::CudaStream stream;
  auto               task = factory.create(input_descs, nlohmann::json(settings), {stream.get()});
  task->bind_logger(spdlog::default_logger());

  auto    input  = curaii::make_unique_device_ptr<float>(1);
  auto    output = curaii::make_unique_device_ptr<float>(1);
  Storage input_storage{MemLoc::Device, sizeof(float), reinterpret_cast<std::byte *>(input.get())};
  Storage output_storage{MemLoc::Device, sizeof(float),
                         reinterpret_cast<std::byte *>(output.get())};
  std::array              input_views{TView{input_desc, &input_storage}};
  std::array              output_views{TView{infer.output_descs[0], &output_storage}};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = input_views,
      .outputs      = output_views,
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  const std::array values{2.0f, 4.0f, 8.0f, 10.0f};
  const std::array expected{2.0f, 3.0f, 14.0f / 3.0f, 22.0f / 3.0f};
  for (size_t i = 0; i < values.size(); ++i) {
    CUDA_CHECK(cudaMemcpyAsync(input.get(), &values[i], sizeof(float), cudaMemcpyHostToDevice,
                               stream.get()));
    ASSERT_EQ(task->execute(ctx), OpResult::Ok);
    float actual = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&actual, output.get(), sizeof(float), cudaMemcpyDeviceToHost,
                               stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    EXPECT_NEAR(actual, expected[i], 1e-6f);
  }
}

TEST(DualReaderBatchQueueTest, EmitsCurrentDelayedAndValidityAtOneFrameCadence) {
  const TDesc                                          input_desc({2}, DType::F32, MemLoc::Host);
  const holotask::asyncs::DualReaderBatchQueueSettings settings{
      .target_capacity = 4,
      .window_size     = 4,
  };
  holotask::asyncs::DualReaderBatchQueueFactory factory;
  const std::array                              input_descs{input_desc};
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));
  auto       task  = factory.create(input_descs, nlohmann::json(settings), {});
  task->bind_logger(spdlog::default_logger());
  TestStorageAccess storage_access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&storage_access);

  std::uint8_t valid_value = 0;
  Storage      valid_storage{MemLoc::Host, sizeof(valid_value),
                             reinterpret_cast<std::byte *>(&valid_value)};
  std::array   output_views{
      TView{infer.output_descs[0], &storage_access.owned_output_storage(0)},
      TView{infer.output_descs[1], &storage_access.owned_output_storage(1)},
      TView{infer.output_descs[2], &valid_storage},
  };
  std::atomic<bool>           cancelled{false};
  holoflow::core::AsyncPopCtx pop_ctx{output_views, &cancelled};

  const std::array values{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                          16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f};
  for (size_t batch = 0; batch < values.size() / 2; ++batch) {
    auto acquired = task->acquire_input(0);
    ASSERT_TRUE(acquired.has_value());
    std::memcpy(acquired->data(), values.data() + 2 * batch, input_desc.num_bytes());
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);

    for (size_t offset = 0; offset < 2; ++offset) {
      const size_t n = 2 * batch + offset;
      ASSERT_EQ(task->try_pop(pop_ctx), OpResult::Ok);
      EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(output_views[0].data()), values[n]);
      if (n == 0) {
        EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(output_views[1].data()), 0.0f);
      } else {
        EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(output_views[1].data()), values[n - 1]);
      }
      EXPECT_EQ(valid_value, n >= 3 ? std::uint8_t{1} : std::uint8_t{0});
      task->release_output(1);
      task->release_output(0);
    }
  }
}

TEST(DualReaderBatchQueueTest, UnitWindowAliasesCurrentFrameAndIsImmediatelyValid) {
  const TDesc                                          input_desc({1}, DType::F32, MemLoc::Host);
  const holotask::asyncs::DualReaderBatchQueueSettings settings{
      .target_capacity = 2,
      .window_size     = 1,
  };
  holotask::asyncs::DualReaderBatchQueueFactory factory;
  const std::array                              input_descs{input_desc};
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));
  auto       task  = factory.create(input_descs, nlohmann::json(settings), {});
  task->bind_logger(spdlog::default_logger());
  TestStorageAccess storage_access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&storage_access);

  auto acquired = task->acquire_input(0);
  ASSERT_TRUE(acquired.has_value());
  const float expected = 42.0f;
  std::memcpy(acquired->data(), &expected, sizeof(expected));
  std::atomic<bool>            cancelled{false};
  std::array                   input_views{*acquired};
  holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
  ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);

  std::uint8_t valid_value = 0;
  Storage      valid_storage{MemLoc::Host, sizeof(valid_value),
                             reinterpret_cast<std::byte *>(&valid_value)};
  std::array   output_views{
      TView{infer.output_descs[0], &storage_access.owned_output_storage(0)},
      TView{infer.output_descs[1], &storage_access.owned_output_storage(1)},
      TView{infer.output_descs[2], &valid_storage},
  };
  holoflow::core::AsyncPopCtx pop_ctx{output_views, &cancelled};
  ASSERT_EQ(task->try_pop(pop_ctx), OpResult::Ok);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(output_views[0].data()), expected);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(output_views[1].data()), expected);
  EXPECT_EQ(valid_value, std::uint8_t{1});
  task->release_output(0);
  task->release_output(1);
}

TEST(SlidingAverageTest, DiscardsInvalidInputsBeforeFullWindowWarmup) {
  const TDesc                                    image_desc({1, 1, 1}, DType::F32, MemLoc::Device);
  const TDesc                                    valid_desc({1}, DType::U8, MemLoc::Host);
  const std::array                               input_descs{image_desc, valid_desc};
  const holotask::asyncs::SlidingAverageSettings settings{
      .target_capacity = 4,
      .window_size     = 3,
  };
  holotask::asyncs::SlidingAverageFactory factory;
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));
  EXPECT_TRUE(infer.synchronizes_producer_stream);

  curaii::CudaStream producer_stream;
  curaii::CudaStream consumer_stream;
  auto               task = factory.create(input_descs, nlohmann::json(settings),
                                           {producer_stream.get(), consumer_stream.get()});
  task->bind_logger(spdlog::default_logger());
  TestStorageAccess storage_access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&storage_access);

  std::uint8_t valid_value = 0;
  Storage      valid_storage{MemLoc::Host, sizeof(valid_value),
                             reinterpret_cast<std::byte *>(&valid_value)};
  TView        valid_view{valid_desc, &valid_storage};
  std::array   output_views{TView{infer.output_descs[0], &storage_access.owned_output_storage(0)}};
  std::atomic<bool>           cancelled{false};
  holoflow::core::AsyncPopCtx pop_ctx{output_views, &cancelled};

  const std::array values{1.0f, 100.0f, 3.0f, 5.0f, 7.0f};
  const std::array valid{true, false, true, true, true};
  size_t           output_count = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    auto acquired = task->acquire_input(0);
    ASSERT_TRUE(acquired.has_value());
    CUDA_CHECK(cudaMemcpy(acquired->data(), &values[i], sizeof(float), cudaMemcpyHostToDevice));
    valid_value = valid[i] ? std::uint8_t{1} : std::uint8_t{0};
    std::array                   input_views{*acquired, valid_view};
    holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
    cudaEvent_t                  pending_work = nullptr;
    if (!valid[i]) {
      CUDA_CHECK(cudaMemsetAsync(acquired->data(), 0, sizeof(float), producer_stream.get()));
      CUDA_CHECK(cudaEventCreate(&pending_work));
      CUDA_CHECK(cudaEventRecord(pending_work, producer_stream.get()));
    }
    ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);
    if (pending_work != nullptr) {
      EXPECT_EQ(cudaEventQuery(pending_work), cudaSuccess);
      CUDA_CHECK(cudaEventDestroy(pending_work));
    }

    const auto pop_result = task->try_pop(pop_ctx);
    if (i < 3) {
      EXPECT_EQ(pop_result, OpResult::NotReady);
      continue;
    }

    ASSERT_EQ(pop_result, OpResult::Ok);
    float actual = 0.0f;
    CUDA_CHECK(cudaMemcpy(&actual, output_views[0].data(), sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(actual, output_count == 0 ? 3.0f : 5.0f);
    ++output_count;
    task->release_output(0);
  }
  EXPECT_EQ(output_count, 2);
}

TEST(SlidingAverageTest, DiscardsConfiguredInitialFramesWithoutValidityInput) {
  const TDesc                                    image_desc({1, 1, 1}, DType::F32, MemLoc::Device);
  const std::array                               input_descs{image_desc};
  const holotask::asyncs::SlidingAverageSettings settings{
      .target_capacity = 4,
      .window_size     = 3,
      .discard_first   = 2,
  };
  holotask::asyncs::SlidingAverageFactory factory;
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));
  EXPECT_TRUE(infer.synchronizes_producer_stream);

  curaii::CudaStream producer_stream;
  curaii::CudaStream consumer_stream;
  auto               task = factory.create(input_descs, nlohmann::json(settings),
                                           {producer_stream.get(), consumer_stream.get()});
  task->bind_logger(spdlog::default_logger());
  TestStorageAccess storage_access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&storage_access);

  std::array output_views{TView{infer.output_descs[0], &storage_access.owned_output_storage(0)}};
  std::atomic<bool>           cancelled{false};
  holoflow::core::AsyncPopCtx pop_ctx{output_views, &cancelled};
  const std::array            values{100.0f, 200.0f, 3.0f, 5.0f, 7.0f};

  for (size_t i = 0; i < values.size(); ++i) {
    auto acquired = task->acquire_input(0);
    ASSERT_TRUE(acquired.has_value());
    CUDA_CHECK(cudaMemcpy(acquired->data(), &values[i], sizeof(float), cudaMemcpyHostToDevice));
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);

    const auto pop_result = task->try_pop(pop_ctx);
    if (i + 1 < values.size()) {
      EXPECT_EQ(pop_result, OpResult::NotReady);
      continue;
    }

    ASSERT_EQ(pop_result, OpResult::Ok);
    float actual = 0.0f;
    CUDA_CHECK(cudaMemcpy(&actual, output_views[0].data(), sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(actual, 5.0f);
    task->release_output(0);
  }
}

TEST(SlidingAverageTest, ProducesMultiElementAveragesAcrossRingWraparound) {
  constexpr size_t width         = 3;
  constexpr size_t height        = 2;
  constexpr size_t element_count = width * height;
  constexpr size_t window_size   = 3;
  constexpr size_t frame_count   = 8;

  const TDesc      image_desc({1, height, width}, DType::F32, MemLoc::Device);
  const std::array input_descs{image_desc};
  const holotask::asyncs::SlidingAverageSettings settings{
      .target_capacity = 2,
      .window_size     = window_size,
  };
  holotask::asyncs::SlidingAverageFactory factory;
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));

  curaii::CudaStream producer_stream;
  curaii::CudaStream consumer_stream;
  auto               task = factory.create(input_descs, nlohmann::json(settings),
                                           {producer_stream.get(), consumer_stream.get()});
  task->bind_logger(spdlog::default_logger());
  TestStorageAccess storage_access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&storage_access);

  std::array output_views{TView{infer.output_descs[0], &storage_access.owned_output_storage(0)}};
  std::atomic<bool>                                         cancelled{false};
  holoflow::core::AsyncPopCtx                               pop_ctx{output_views, &cancelled};
  std::array<float, element_count>                          running_average{};
  std::array<std::array<float, element_count>, window_size> history{};

  for (size_t frame = 0; frame < frame_count; ++frame) {
    std::array<float, element_count> input{};
    for (size_t pixel = 0; pixel < element_count; ++pixel) {
      input[pixel] = 3.0f * static_cast<float>(frame * element_count + pixel + 1);
      running_average[pixel] += input[pixel] / static_cast<float>(window_size);
      running_average[pixel] -=
          history[frame % window_size][pixel] / static_cast<float>(window_size);
    }
    history[frame % window_size] = input;

    auto acquired = task->acquire_input(0);
    ASSERT_TRUE(acquired.has_value());
    CUDA_CHECK(
        cudaMemcpy(acquired->data(), input.data(), image_desc.num_bytes(), cudaMemcpyHostToDevice));
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);

    const auto pop_result = task->try_pop(pop_ctx);
    if (frame + 1 < window_size) {
      EXPECT_EQ(pop_result, OpResult::NotReady);
      continue;
    }

    ASSERT_EQ(pop_result, OpResult::Ok);
    std::array<float, element_count> actual{};
    CUDA_CHECK(cudaMemcpy(actual.data(), output_views[0].data(), image_desc.num_bytes(),
                          cudaMemcpyDeviceToHost));
    for (size_t pixel = 0; pixel < element_count; ++pixel) {
      EXPECT_FLOAT_EQ(actual[pixel], running_average[pixel]);
    }
    task->release_output(0);
  }
}

} // namespace
