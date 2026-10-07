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

#include <algorithm>
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
#include "holotask/asyncs/batch_queue.hh"
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

size_t sequence_at(const holoflow::core::PointerSequence &sequence, size_t step) {
  return step < sequence.prefix.size()
             ? sequence.prefix[step]
             : sequence.cycle[(step - sequence.prefix.size()) % sequence.cycle.size()];
}

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

TEST(CausalSlidingAverageCudaGraph, ReplayAdvancesDeviceSampleCount) {
  const TDesc                                         input_desc({1}, DType::F32, MemLoc::Device);
  const holotask::syncs::CausalSlidingAverageSettings settings{3};
  holotask::syncs::CausalSlidingAverageFactory        factory;
  const std::array                                    input_descs{input_desc};
  const auto infer = factory.infer(input_descs, nlohmann::json(settings));

  curaii::CudaStream stream;
  auto               task = factory.create(input_descs, nlohmann::json(settings), {stream.get()});
  ASSERT_TRUE(task->supports_cuda_graph());

  auto    input  = curaii::make_unique_device_ptr<float>(1);
  auto    output = curaii::make_unique_device_ptr<float>(1);
  Storage input_storage{MemLoc::Device, sizeof(float), reinterpret_cast<std::byte *>(input.get())};
  Storage output_storage{MemLoc::Device, sizeof(float),
                         reinterpret_cast<std::byte *>(output.get())};
  std::array input_views{TView{input_desc, &input_storage}};
  std::array output_views{TView{infer.output_descs[0], &output_storage}};

  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  holoflow::core::CudaGraphCtx recording{input_views, output_views, stream.get(), nullptr};
  task->record_cuda_graph(recording);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));

  const std::array values{2.0f, 4.0f, 8.0f, 10.0f};
  const std::array expected{2.0f, 3.0f, 14.0f / 3.0f, 22.0f / 3.0f};
  for (size_t i = 0; i < values.size(); ++i) {
    CUDA_CHECK(cudaMemcpyAsync(input.get(), &values[i], sizeof(float), cudaMemcpyHostToDevice,
                               stream.get()));
    CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
    float actual = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&actual, output.get(), sizeof(float), cudaMemcpyDeviceToHost,
                               stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    EXPECT_NEAR(actual, expected[i], 1e-6f);
  }

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

TEST(BatchQueueUpdateTest, TransfersAllocationResetsCursorsAndInvalidatesBeforeReallocation) {
  using namespace holoflow::core;
  const std::array                    input_descs{TDesc({1}, DType::F32, MemLoc::Host)};
  holotask::asyncs::BatchQueueFactory factory;
  nlohmann::json                      settings = holotask::asyncs::BatchQueueSettings{4, 1, 1};
  const auto                          infer    = factory.infer(input_descs, settings);
  auto                                task     = factory.create(input_descs, settings, {});
  TestStorageAccess                   access(infer.input_descs, infer.output_descs);
  task->bind_storage_access(&access);
  const auto pointers = *task->owned_input_pointers(0);
  auto       acquired = task->acquire_input(0);
  ASSERT_TRUE(acquired);
  std::array        inputs{*acquired};
  std::atomic<bool> cancelled{false};
  AsyncPushCtx      push{inputs, &cancelled};
  ASSERT_EQ(task->try_push(push), OpResult::Ok);
  EXPECT_EQ(sequence_at(*task->owned_input_pointer_sequence(0), 0), 1);

  int                   invalidations = 0;
  ExecutionInvalidation invalidation{[&]() noexcept { ++invalidations; }};
  const AsyncCreateCtx  ctx{.execution_invalidation = &invalidation};
  task = factory.update(std::move(task), input_descs, settings, ctx);
  task->bind_storage_access(&access);
  EXPECT_EQ(invalidations, 0);
  EXPECT_EQ(*task->owned_input_pointers(0), pointers);
  EXPECT_EQ(sequence_at(*task->owned_input_pointer_sequence(0), 0), 0);
  std::array  outputs{TView{infer.output_descs[0], &access.owned_output_storage(0)}};
  AsyncPopCtx pop{outputs, &cancelled};
  EXPECT_EQ(task->try_pop(pop), OpResult::NotReady);

  settings["target_capacity"] = 8;
  task                        = factory.update(std::move(task), input_descs, settings, ctx);
  EXPECT_EQ(invalidations, 1);
  EXPECT_TRUE(invalidation.invalidated);
  EXPECT_NE(task->owned_input_pointers(0)->front(), pointers.front());
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

  const auto inputs           = *task->owned_input_pointers(0);
  const auto current          = *task->owned_output_pointers(0);
  const auto delayed          = *task->owned_output_pointers(1);
  const auto input_sequence   = *task->owned_input_pointer_sequence(0);
  const auto current_sequence = *task->owned_output_pointer_sequence(0);
  const auto delayed_sequence = *task->owned_output_pointer_sequence(1);
  EXPECT_EQ(inputs.size(), *infer.owned_input_pointer_counts[0]);
  EXPECT_EQ(current.size(), *infer.owned_output_pointer_counts[0]);
  EXPECT_EQ(delayed.size(), current.size() + 1);

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
    EXPECT_NE(std::find(inputs.begin(), inputs.end(), acquired->storage->ptr), inputs.end());
    EXPECT_EQ(acquired->storage->ptr, inputs[sequence_at(input_sequence, batch)]);
    std::memcpy(acquired->data(), values.data() + 2 * batch, input_desc.num_bytes());
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push_ctx{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push_ctx), OpResult::Ok);

    for (size_t offset = 0; offset < 2; ++offset) {
      const size_t n = 2 * batch + offset;
      ASSERT_EQ(task->try_pop(pop_ctx), OpResult::Ok);
      EXPECT_EQ(output_views[0].storage->ptr, current[sequence_at(current_sequence, n)]);
      EXPECT_EQ(output_views[1].storage->ptr, delayed[sequence_at(delayed_sequence, n)]);
      EXPECT_EQ(sequence_at(*task->owned_output_pointer_sequence(1), 0),
                sequence_at(delayed_sequence, n));
      EXPECT_NE(std::find(current.begin(), current.end(), output_views[0].storage->ptr),
                current.end());
      EXPECT_NE(std::find(delayed.begin(), delayed.end(), output_views[1].storage->ptr),
                delayed.end());
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
  EXPECT_EQ(task->owned_output_pointers(0), task->owned_output_pointers(1));
  EXPECT_TRUE(task->owned_output_pointer_sequence(1)->prefix.empty());

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

TEST(DualReaderBatchQueueUpdateTest, RetainsBuffersAndResetsHeldReadersOnHostAndDevice) {
  for (const auto location : {MemLoc::Host, MemLoc::Device}) {
    SCOPED_TRACE(static_cast<int>(location));
    curaii::CudaStream                    stream;
    holoflow::core::ExecutionInvalidation invalidation;
    holoflow::core::AsyncCreateCtx        create_ctx{stream.get(), stream.get(), &invalidation};
    holotask::asyncs::DualReaderBatchQueueFactory  factory;
    const std::array                               descs{TDesc({2}, DType::F32, location)};
    holotask::asyncs::DualReaderBatchQueueSettings settings{4, 3};
    auto                                           infer = factory.infer(descs, settings);
    auto              task = factory.create(descs, settings, create_ctx);
    TestStorageAccess access(infer.input_descs, infer.output_descs);
    task->bind_storage_access(&access);
    const auto        inputs  = task->owned_input_pointers(0);
    const auto        current = task->owned_output_pointers(0);
    const auto        delayed = task->owned_output_pointers(1);
    std::uint8_t      valid   = 0;
    Storage           valid_storage{MemLoc::Host, 1, reinterpret_cast<std::byte *>(&valid)};
    std::array        outputs{TView{infer.output_descs[0], &access.owned_output_storage(0)},
                              TView{infer.output_descs[1], &access.owned_output_storage(1)},
                              TView{infer.output_descs[2], &valid_storage}};
    std::atomic<bool> cancelled{false};
    holoflow::core::AsyncPopCtx pop{outputs, &cancelled};
    auto                        push = [&] {
      auto acquired = task->acquire_input(0);
      ASSERT_TRUE(acquired);
      const std::array values{10.F, 11.F};
      if (location == MemLoc::Host)
        std::memcpy(acquired->data(), values.data(), sizeof(values));
      else {
        CUDA_CHECK(cudaMemcpyAsync(acquired->data(), values.data(), sizeof(values),
                                   cudaMemcpyHostToDevice, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
      }
      std::array                   views{*acquired};
      holoflow::core::AsyncPushCtx ctx{views, &cancelled};
      ASSERT_EQ(task->try_push(ctx), OpResult::Ok);
    };
    push();
    ASSERT_EQ(task->try_pop(pop), OpResult::Ok);
    task->release_output(0); // Simulate a stopped section still holding its delayed output.
    auto *scratch = delayed->back();
    if (location == MemLoc::Host)
      std::memset(scratch, 1, sizeof(float));
    else {
      CUDA_CHECK(cudaMemsetAsync(scratch, 1, sizeof(float), stream.get()));
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    }

    // All these settings have eight slots and the same scratch size.
    for (const auto next : {settings, holotask::asyncs::DualReaderBatchQueueSettings{4, 4},
                            holotask::asyncs::DualReaderBatchQueueSettings{5, 1}, settings}) {
      task = factory.update(std::move(task), descs, next, create_ctx);
      EXPECT_FALSE(invalidation.invalidated);
      EXPECT_EQ(task->owned_input_pointers(0), inputs);
      EXPECT_EQ(task->owned_output_pointers(0), current);
      const auto new_delayed = *task->owned_output_pointers(1);
      EXPECT_EQ(new_delayed.size(), current->size() + (next.window_size > 2 ? 1 : 0));
      if (next.window_size > 2)
        EXPECT_EQ(new_delayed.back(), scratch);
      float zero = -1;
      if (location == MemLoc::Host)
        std::memcpy(&zero, scratch, sizeof(zero));
      else
        CUDA_CHECK(cudaMemcpy(&zero, scratch, sizeof(zero), cudaMemcpyDeviceToHost));
      EXPECT_FLOAT_EQ(zero, 0);
      task->bind_storage_access(&access);
      EXPECT_EQ(task->try_pop(pop), OpResult::NotReady);
      EXPECT_EQ(task->owned_input_pointer_sequence(0)->cycle.front(), 0);
      EXPECT_EQ(task->owned_output_pointer_sequence(0)->cycle.front(), 0);
      EXPECT_EQ(task->owned_output_pointer_sequence(1)->prefix.size(), (next.window_size - 1) / 2);
      for (size_t frame = 0; frame < 4; ++frame) {
        if (frame % 2 == 0)
          push();
        ASSERT_EQ(task->try_pop(pop), OpResult::Ok);
        EXPECT_EQ(valid, frame >= next.window_size - 1 ? 1 : 0);
        const size_t delay = (next.window_size - 1) / 2;
        for (size_t port = 0; port < 2; ++port) {
          float value = -1;
          if (location == MemLoc::Host)
            std::memcpy(&value, outputs[port].data(), sizeof(value));
          else
            CUDA_CHECK(
                cudaMemcpy(&value, outputs[port].data(), sizeof(value), cudaMemcpyDeviceToHost));
          const float expected = port == 1 && frame < delay
                                     ? 0.F
                                     : 10.F + float((frame - (port == 1 ? delay : 0)) % 2);
          EXPECT_FLOAT_EQ(value, expected);
        }
        task->release_output(1);
        task->release_output(0);
      }
    }
  }
}

TEST(DualReaderBatchQueueUpdateTest, InvalidatesBeforeReallocationAndOnInvalidSettings) {
  holotask::asyncs::DualReaderBatchQueueFactory factory;
  const std::array                              initial_descs{TDesc({2}, DType::F32, MemLoc::Host)};
  const holotask::asyncs::DualReaderBatchQueueSettings initial{4, 3};
  for (int scenario = 0; scenario < 4; ++scenario) {
    SCOPED_TRACE(scenario);
    auto                                  task    = factory.create(initial_descs, initial, {});
    auto                                 *ring    = task->owned_input_pointers(0)->front();
    auto                                 *scratch = task->owned_output_pointers(1)->back();
    holoflow::core::ExecutionInvalidation invalidation{[&] {
      // Captured allocations are still live when invalidation runs.
      *ring    = std::byte{42};
      *scratch = std::byte{42};
    }};
    auto                                  descs    = initial_descs;
    auto                                  settings = initial;
    if (scenario == 0)
      settings.target_capacity = 20;
    else if (scenario == 1) {
      descs[0] = TDesc({2}, DType::CF32, MemLoc::Host);
      settings = {1, 1}; // Same ring byte count, different scratch byte count.
    } else if (scenario == 2)
      descs[0] = TDesc({2}, DType::F32, MemLoc::Device);
    else
      settings.window_size = 0;
    if (scenario == 3)
      EXPECT_THROW(factory.update(std::move(task), descs, settings,
                                  {.execution_invalidation = &invalidation}),
                   std::invalid_argument);
    else {
      task = factory.update(std::move(task), descs, settings,
                            {.execution_invalidation = &invalidation});
      EXPECT_NE(task, nullptr);
    }
    EXPECT_TRUE(invalidation.invalidated);
  }
}

TEST(BatchQueuePointersTest, EnumerationMatchesAlignedWraparound) {
  const TDesc                         desc({2}, DType::F32, MemLoc::Host);
  const std::array                    descs{desc};
  holotask::asyncs::BatchQueueFactory factory;
  const nlohmann::json                settings = holotask::asyncs::BatchQueueSettings{5, 1, 1};
  const auto                          inferred = factory.infer(descs, settings);
  auto                                task     = factory.create(descs, settings, {});
  TestStorageAccess                   access(inferred.input_descs, inferred.output_descs);
  task->bind_storage_access(&access);
  const auto inputs  = *task->owned_input_pointers(0);
  const auto outputs = *task->owned_output_pointers(0);
  EXPECT_EQ(inputs.size(), 4U);
  EXPECT_EQ(outputs.size(), 8U);
  EXPECT_EQ(inputs.size(), *inferred.owned_input_pointer_counts[0]);
  EXPECT_EQ(outputs.size(), *inferred.owned_output_pointer_counts[0]);
  std::atomic<bool> cancelled{false};
  std::array        output_views{TView{inferred.output_descs[0], &access.owned_output_storage(0)}};
  holoflow::core::AsyncPopCtx pop{output_views, &cancelled};
  for (size_t batch = 0; batch < 12; ++batch) {
    const auto next_input = sequence_at(*task->owned_input_pointer_sequence(0), 0);
    auto       acquired   = task->acquire_input(0);
    ASSERT_TRUE(acquired);
    EXPECT_EQ(acquired->storage->ptr, inputs[batch % inputs.size()]);
    EXPECT_EQ(acquired->storage->ptr, inputs[next_input]);
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push), OpResult::Ok);
    for (size_t j = 0; j < 2; ++j) {
      const auto next_output = sequence_at(*task->owned_output_pointer_sequence(0), 0);
      ASSERT_EQ(task->try_pop(pop), OpResult::Ok);
      EXPECT_EQ(output_views[0].storage->ptr, outputs[next_output]);
      EXPECT_EQ(output_views[0].storage->ptr, outputs[(batch * 2 + j) % outputs.size()]);
      task->release_output(0);
    }
  }
}

TEST(SlidingAverageTest, PointerDomainsRemainStableThroughWarmupAndWraparound) {
  const TDesc          desc({1, 1, 1}, DType::F32, MemLoc::Device);
  const std::array     descs{desc};
  const nlohmann::json settings = holotask::asyncs::SlidingAverageSettings{2, 3, 0};
  holotask::asyncs::SlidingAverageFactory factory;
  const auto                              inferred = factory.infer(descs, settings);
  curaii::CudaStream                      stream;
  auto              task = factory.create(descs, settings, {stream.get(), stream.get()});
  TestStorageAccess access(inferred.input_descs, inferred.output_descs);
  task->bind_storage_access(&access);
  task->bind_logger(spdlog::default_logger());
  const auto inputs  = *task->owned_input_pointers(0);
  const auto outputs = *task->owned_output_pointers(0);
  EXPECT_EQ(inputs, outputs);
  EXPECT_EQ(inputs.size(), 5U);
  EXPECT_EQ(inputs.size(), *inferred.owned_input_pointer_counts[0]);
  std::atomic<bool> cancelled{false};
  std::array        output_views{TView{inferred.output_descs[0], &access.owned_output_storage(0)}};
  holoflow::core::AsyncPopCtx pop{output_views, &cancelled};
  for (size_t frame = 0; frame < 17; ++frame) {
    const auto next_input  = sequence_at(*task->owned_input_pointer_sequence(0), 0);
    const auto next_output = sequence_at(*task->owned_output_pointer_sequence(0), 0);
    auto       acquired    = task->acquire_input(0);
    ASSERT_TRUE(acquired);
    EXPECT_EQ(acquired->storage->ptr, inputs[frame % inputs.size()]);
    EXPECT_EQ(acquired->storage->ptr, inputs[next_input]);
    CUDA_CHECK(cudaMemsetAsync(acquired->data(), 0, sizeof(float), stream.get()));
    std::array                   input_views{*acquired};
    holoflow::core::AsyncPushCtx push{input_views, &cancelled};
    ASSERT_EQ(task->try_push(push), OpResult::Ok);
    if (task->try_pop(pop) == OpResult::Ok) {
      EXPECT_EQ(output_views[0].storage->ptr, outputs[next_output]);
      EXPECT_NE(std::find(outputs.begin(), outputs.end(), output_views[0].storage->ptr),
                outputs.end());
      task->release_output(0);
    }
  }
  EXPECT_EQ(task->owned_output_pointers(0), outputs);
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
  EXPECT_FALSE(task->owned_input_pointer_sequence(0));
  EXPECT_TRUE(task->owned_output_pointer_sequence(0));

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
  const auto sequence = *task->owned_input_pointer_sequence(0);
  const auto pointers = *task->owned_input_pointers(0);
  EXPECT_EQ(sequence.prefix.size(), 2U);

  std::array output_views{TView{infer.output_descs[0], &storage_access.owned_output_storage(0)}};
  std::atomic<bool>           cancelled{false};
  holoflow::core::AsyncPopCtx pop_ctx{output_views, &cancelled};
  const std::array            values{100.0f, 200.0f, 3.0f, 5.0f, 7.0f};

  for (size_t i = 0; i < values.size(); ++i) {
    auto acquired = task->acquire_input(0);
    ASSERT_TRUE(acquired.has_value());
    EXPECT_EQ(acquired->storage->ptr, pointers[sequence_at(sequence, i)]);
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
