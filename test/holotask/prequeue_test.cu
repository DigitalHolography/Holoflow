// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <spdlog/spdlog.h>

#include "../../src/holoflow/src/runtime/diagnostics_file_writer.hh"
#include <thread>

#include "holoflow/runtime/compiler.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/dual_reader_batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/syncs/pca.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

class StorageAccess final : public IOStorageAccess {
public:
  explicit StorageAccess(const InferResult &infer) {
    for (const auto &desc : infer.input_descs)
      inputs.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
    for (const auto &desc : infer.output_descs)
      outputs.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
  }
  Storage             &owned_input_storage(size_t port) override { return inputs.at(port); }
  Storage             &owned_output_storage(size_t port) override { return outputs.at(port); }
  std::vector<Storage> inputs, outputs;
};

TEST(PrequeueOwnership, BatchReservationsArePrivateAndRetireInOrder) {
  holotask::asyncs::BatchQueueFactory factory;
  const std::array                    inputs{TDesc({2}, DType::F32, MemLoc::Host)};
  const auto                          settings = holotask::asyncs::BatchQueueSettings{4, 1, 2};
  const auto                          infer    = factory.infer(inputs, settings);
  EXPECT_TRUE(infer.supports_prequeued_push);
  EXPECT_TRUE(infer.supports_prequeued_pop);
  auto          task = factory.create(inputs, settings, {});
  StorageAccess access(infer);
  task->bind_storage_access(&access);
  std::array        in{TView{inputs[0], &access.inputs[0]}};
  std::array        out{TView{infer.output_descs[0], &access.outputs[0]}};
  std::atomic<bool> cancelled{false};
  AsyncPushCtx      push{in, &cancelled};
  AsyncPopCtx       pop{out, &cancelled};
  AsyncTicket       first, second, third, read_first, read_second;
  ASSERT_EQ(task->reserve_push(push, first), OpResult::Ok);
  auto *first_address                       = in[0].data();
  *reinterpret_cast<float *>(first_address) = 11.F;
  ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
  auto *second_address = in[0].data();
  EXPECT_NE(first_address, second_address);
  *reinterpret_cast<float *>(second_address) = 22.F;
  EXPECT_EQ(task->reserve_push(push, third), OpResult::NotReady);
  task->cancel_push(second);
  ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
  EXPECT_EQ(in[0].data(), second_address);
  task->enqueue_push(first);
  task->enqueue_push(second);
  EXPECT_THROW(task->commit_push(second), std::logic_error);
  EXPECT_THROW(task->cancel_push(second), std::logic_error);
  EXPECT_EQ(task->reserve_pop(pop, read_first), OpResult::NotReady);
  task->commit_push(first);
  ASSERT_EQ(task->reserve_pop(pop, read_first), OpResult::Ok);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[0].data()), 11.F);
  task->commit_push(second);
  ASSERT_EQ(task->reserve_pop(pop, read_second), OpResult::Ok);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[0].data()), 22.F);
  EXPECT_EQ(task->reserve_push(push, third), OpResult::NotReady);
  EXPECT_THROW(task->release_pop(read_second), std::logic_error);
  task->release_pop(read_first);
  ASSERT_EQ(task->reserve_push(push, third), OpResult::Ok);
  task->cancel_push(third);
  task->cancel_pop(read_second);
  ASSERT_EQ(task->reserve_pop(pop, read_second), OpResult::Ok);
  EXPECT_EQ(out[0].data(), second_address);
  task->release_pop(read_second);
  cancelled.store(true);
  EXPECT_EQ(task->reserve_push(push, third), OpResult::Cancelled);
}

TEST(PrequeueOwnership, SlidingAveragePreservesDiscardsAndWraparound) {
  for (size_t window : {size_t{1}, size_t{3}}) {
    holotask::asyncs::SlidingAverageFactory factory;
    const std::array                        inputs{TDesc({1, 1, 1}, DType::F32, MemLoc::Device)};
    const auto         settings = holotask::asyncs::SlidingAverageSettings{4, window, 2};
    const auto         infer    = factory.infer(inputs, settings);
    curaii::CudaStream producer, consumer;
    auto               task = factory.create(inputs, settings, {producer.get(), consumer.get()});
    task->bind_logger(spdlog::default_logger());
    StorageAccess access(infer);
    task->bind_storage_access(&access);
    std::array        in{TView{inputs[0], &access.inputs[0]}};
    std::array        out{TView{infer.output_descs[0], &access.outputs[0]}};
    std::atomic<bool> cancelled{false};
    AsyncPushCtx      push{in, &cancelled};
    AsyncPopCtx       pop{out, &cancelled};
    auto              values = curaii::make_unique_host_ptr<float>(32);
    for (size_t i = 0; i < 32; ++i)
      values.get()[i] = static_cast<float>(i + 1);
    std::vector<float> actual;
    for (size_t i = 0; i < 32; i += 2) {
      std::array<AsyncTicket, 2> tickets;
      for (size_t k = 0; k < 2; ++k) {
        ASSERT_EQ(task->reserve_push(push, tickets[k]), OpResult::Ok);
        CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get() + i + k, sizeof(float),
                                   cudaMemcpyHostToDevice, producer.get()));
        task->enqueue_push(tickets[k]);
      }
      AsyncTicket read;
      EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
      CUDA_CHECK(cudaStreamSynchronize(producer.get()));
      task->commit_push(tickets[0]);
      task->commit_push(tickets[1]);
      std::vector<AsyncTicket> reads;
      while (task->reserve_pop(pop, read) == OpResult::Ok) {
        float value = 0;
        CUDA_CHECK(cudaMemcpy(&value, out[0].data(), sizeof(float), cudaMemcpyDeviceToHost));
        actual.push_back(value);
        reads.push_back(read);
      }
      for (auto ticket : reads)
        task->release_pop(ticket);
    }
    ASSERT_EQ(actual.size(), 32 - settings.discard_first - window + 1);
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_FLOAT_EQ(actual[i], static_cast<float>(settings.discard_first + i + 1) +
                                     static_cast<float>(window - 1) / 2.F);
    const std::array with_validity{inputs[0], TDesc({1}, DType::U8, MemLoc::Host)};
    const auto       validity_infer = factory.infer(with_validity, settings);
    EXPECT_FALSE(validity_infer.supports_prequeued_push);
    EXPECT_TRUE(validity_infer.supports_prequeued_pop);
  }
}

TEST(PrequeueOwnership, DualReaderRetainsPairsAcrossStartupCancellationAndWraparound) {
  for (size_t window : {size_t{1}, size_t{4}, size_t{5}}) {
    holotask::asyncs::DualReaderBatchQueueFactory factory;
    const std::array                              inputs{TDesc({2}, DType::F32, MemLoc::Host)};
    const auto settings = holotask::asyncs::DualReaderBatchQueueSettings{4, window};
    const auto infer    = factory.infer(inputs, settings);
    EXPECT_TRUE(infer.supports_prequeued_pop);
    EXPECT_TRUE(infer.supports_prequeued_push);
    auto          task = factory.create(inputs, settings, {});
    StorageAccess access(infer);
    task->bind_storage_access(&access);
    std::uint8_t valid    = 0;
    access.outputs[2].ptr = reinterpret_cast<std::byte *>(&valid);
    std::array        out{TView{infer.output_descs[0], &access.outputs[0]},
                          TView{infer.output_descs[1], &access.outputs[1]},
                          TView{infer.output_descs[2], &access.outputs[2]}};
    std::atomic<bool> cancelled{false};
    AsyncPopCtx       pop{out, &cancelled};
    const auto        current      = *task->owned_output_pointers(0);
    const auto        delayed      = *task->owned_output_pointers(1);
    const auto        next_pointer = [](const auto &pointers, const PointerSequence &sequence) {
      return pointers.at(sequence.prefix.empty() ? sequence.cycle.front()
                                                 : sequence.prefix.front());
    };
    for (size_t frame = 0; frame < 40; frame += 2) {
      auto input = task->acquire_input(0);
      ASSERT_TRUE(input);
      auto *data = reinterpret_cast<float *>(input->data());
      data[0]    = static_cast<float>(frame + 1);
      data[1]    = static_cast<float>(frame + 2);
      std::array   in{*input};
      AsyncPushCtx push{in, &cancelled};
      ASSERT_EQ(task->try_push(push), OpResult::Ok);
      std::array<AsyncTicket, 2> tickets;
      for (size_t k = 0; k < 2; ++k) {
        auto *expected_current = next_pointer(current, *task->owned_output_pointer_sequence(0));
        auto *expected_delayed = next_pointer(delayed, *task->owned_output_pointer_sequence(1));
        ASSERT_EQ(task->reserve_pop(pop, tickets[k]), OpResult::Ok);
        EXPECT_EQ(out[0].data(), expected_current);
        EXPECT_EQ(out[1].data(), expected_delayed);
        const size_t n = frame + k;
        EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[0].data()), static_cast<float>(n + 1));
        const size_t delay = (window - 1) / 2;
        EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[1].data()),
                        n < delay ? 0.F : static_cast<float>(n + 1 - delay));
        EXPECT_EQ(valid, n >= window - 1 ? 1 : 0);
      }
      AsyncTicket extra;
      EXPECT_EQ(task->reserve_pop(pop, extra), OpResult::NotReady);
      EXPECT_THROW(task->release_pop(tickets[1]), std::logic_error);
      auto *second_current = out[0].data();
      auto *second_delayed = out[1].data();
      task->cancel_pop(tickets[1]);
      ASSERT_EQ(task->reserve_pop(pop, tickets[1]), OpResult::Ok);
      EXPECT_EQ(out[0].data(), second_current);
      EXPECT_EQ(out[1].data(), second_delayed);
      task->release_pop(tickets[0]);
      EXPECT_EQ(out[0].data(), second_current);
      task->release_pop(tickets[1]);
    }
    cancelled.store(true);
    AsyncTicket ticket;
    EXPECT_EQ(task->reserve_pop(pop, ticket), OpResult::Cancelled);
  }
}

TEST(PrequeueOwnership, DualReaderWritesPublishInOrderAcrossCancellationAndWraparound) {
  for (size_t window : {size_t{1}, size_t{4}, size_t{5}}) {
    holotask::asyncs::DualReaderBatchQueueFactory factory;
    const std::array                              inputs{TDesc({2}, DType::F32, MemLoc::Host)};
    const auto    settings = holotask::asyncs::DualReaderBatchQueueSettings{8, window};
    const auto    infer    = factory.infer(inputs, settings);
    auto          task     = factory.create(inputs, settings, {});
    StorageAccess access(infer);
    task->bind_storage_access(&access);
    std::uint8_t valid    = 0;
    access.outputs[2].ptr = reinterpret_cast<std::byte *>(&valid);
    std::array        in{TView{inputs[0], &access.inputs[0]}};
    std::array        out{TView{infer.output_descs[0], &access.outputs[0]},
                          TView{infer.output_descs[1], &access.outputs[1]},
                          TView{infer.output_descs[2], &access.outputs[2]}};
    std::atomic<bool> cancelled{false};
    AsyncPushCtx      push{in, &cancelled};
    AsyncPopCtx       pop{out, &cancelled};
    for (size_t frame = 0; frame < 80; frame += 4) {
      AsyncTicket first, second, extra, read;
      ASSERT_EQ(task->reserve_push(push, first), OpResult::Ok);
      auto *first_ptr = reinterpret_cast<float *>(in[0].data());
      first_ptr[0]    = static_cast<float>(frame + 1);
      first_ptr[1]    = static_cast<float>(frame + 2);
      ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
      auto *second_ptr = reinterpret_cast<float *>(in[0].data());
      EXPECT_NE(first_ptr, second_ptr);
      EXPECT_EQ(task->reserve_push(push, extra), OpResult::NotReady);
      EXPECT_THROW((void)task->acquire_input(0), std::logic_error);
      EXPECT_THROW((void)task->try_push(push), std::logic_error);
      EXPECT_THROW(task->commit_push(first), std::logic_error);
      EXPECT_THROW(task->cancel_push(first), std::logic_error);
      task->cancel_push(second);
      ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
      EXPECT_EQ(in[0].data(), reinterpret_cast<std::byte *>(second_ptr));
      second_ptr[0] = static_cast<float>(frame + 3);
      second_ptr[1] = static_cast<float>(frame + 4);
      task->enqueue_push(first);
      task->enqueue_push(second);
      EXPECT_THROW(task->enqueue_push(first), std::logic_error);
      EXPECT_THROW(task->commit_push(second), std::logic_error);
      EXPECT_THROW(task->cancel_push(second), std::logic_error);
      EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
      for (size_t group = 0; group < 2; ++group) {
        task->commit_push(group == 0 ? first : second);
        EXPECT_EQ(in[0].data(), reinterpret_cast<std::byte *>(second_ptr));
        for (size_t k = 0; k < 2; ++k) {
          ASSERT_EQ(task->reserve_pop(pop, read), OpResult::Ok);
          const size_t n = frame + 2 * group + k;
          EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[0].data()), static_cast<float>(n + 1));
          const size_t delay = (window - 1) / 2;
          EXPECT_FLOAT_EQ(*reinterpret_cast<float *>(out[1].data()),
                          n < delay ? 0.F : static_cast<float>(n + 1 - delay));
          EXPECT_EQ(valid, n >= window - 1 ? 1 : 0);
          task->release_pop(read);
        }
        EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
      }
    }
    cancelled.store(true);
    AsyncTicket ticket;
    EXPECT_EQ(task->reserve_push(push, ticket), OpResult::Cancelled);
  }
}

TEST(PrequeueOwnership, DualReaderReservedWritesRespectDelayedReaderBackpressure) {
  holotask::asyncs::DualReaderBatchQueueFactory factory;
  const std::array                              inputs{TDesc({2}, DType::F32, MemLoc::Host)};
  const auto    settings = holotask::asyncs::DualReaderBatchQueueSettings{4, 5};
  const auto    infer    = factory.infer(inputs, settings);
  auto          task     = factory.create(inputs, settings, {});
  StorageAccess access(infer);
  task->bind_storage_access(&access);
  std::uint8_t valid    = 0;
  access.outputs[2].ptr = reinterpret_cast<std::byte *>(&valid);
  std::array   in{TView{inputs[0], &access.inputs[0]}};
  std::array   out{TView{infer.output_descs[0], &access.outputs[0]},
                   TView{infer.output_descs[1], &access.outputs[1]},
                   TView{infer.output_descs[2], &access.outputs[2]}};
  AsyncPushCtx push{in, nullptr};
  AsyncPopCtx  pop{out, nullptr};
  AsyncTicket  write, extra, first, second;
  // Ten ring elements: eight can be occupied before the two-element writer must wait.
  ASSERT_EQ(task->owned_input_pointers(0)->size(), 5U);
  for (size_t group = 0; group < 4; ++group) {
    ASSERT_EQ(task->reserve_push(push, write), OpResult::Ok);
    if (group == 3)
      EXPECT_EQ(task->reserve_push(push, extra), OpResult::NotReady);
    task->enqueue_push(write);
    task->commit_push(write);
  }
  for (size_t k = 0; k < 2; ++k) {
    ASSERT_EQ(task->reserve_pop(pop, first), OpResult::Ok);
    task->release_pop(first);
  }
  // Advancing through startup scratch does not release the delayed history.
  EXPECT_EQ(task->reserve_push(push, write), OpResult::NotReady);
  ASSERT_EQ(task->reserve_pop(pop, first), OpResult::Ok);
  ASSERT_EQ(task->reserve_pop(pop, second), OpResult::Ok);
  auto *held_current = out[0].data();
  auto *held_delayed = out[1].data();
  EXPECT_EQ(task->reserve_push(push, write), OpResult::NotReady);
  task->release_pop(first);
  ASSERT_EQ(task->reserve_push(push, write), OpResult::Ok);
  EXPECT_NE(in[0].data(), held_current);
  EXPECT_NE(in[0].data(), held_delayed);
  // A second group would wrap over the still-held delayed frame.
  EXPECT_EQ(task->reserve_push(push, extra), OpResult::NotReady);
  task->cancel_push(write);
  task->release_pop(second);
  ASSERT_EQ(task->reserve_push(push, write), OpResult::Ok);
  task->cancel_push(write);
}

TEST(PrequeueOwnership, DualReaderDoesNotReturnReservedStorageToWriter) {
  holotask::asyncs::DualReaderBatchQueueFactory factory;
  const std::array                              inputs{TDesc({2}, DType::F32, MemLoc::Host)};
  const auto    settings = holotask::asyncs::DualReaderBatchQueueSettings{4, 1};
  const auto    infer    = factory.infer(inputs, settings);
  auto          task     = factory.create(inputs, settings, {});
  StorageAccess access(infer);
  task->bind_storage_access(&access);
  std::uint8_t valid    = 0;
  access.outputs[2].ptr = reinterpret_cast<std::byte *>(&valid);
  std::array  out{TView{infer.output_descs[0], &access.outputs[0]},
                  TView{infer.output_descs[1], &access.outputs[1]},
                  TView{infer.output_descs[2], &access.outputs[2]}};
  AsyncPopCtx pop{out, nullptr};
  while (auto input = task->acquire_input(0)) {
    std::array   in{*input};
    AsyncPushCtx push{in, nullptr};
    ASSERT_EQ(task->try_push(push), OpResult::Ok);
  }
  AsyncTicket first, second;
  ASSERT_EQ(task->reserve_pop(pop, first), OpResult::Ok);
  ASSERT_EQ(task->reserve_pop(pop, second), OpResult::Ok);
  EXPECT_FALSE(task->acquire_input(0));
  task->release_pop(first);
  auto input = task->acquire_input(0);
  ASSERT_TRUE(input);
  EXPECT_NE(input->data(), out[0].data());
  std::array   in{*input};
  AsyncPushCtx push{in, nullptr};
  ASSERT_EQ(task->try_push(push), OpResult::Ok);
  EXPECT_FALSE(task->acquire_input(0));
  task->release_pop(second);
}

__global__ void fill_frames(float *output, size_t count, size_t first) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = static_cast<float>(first + i);
}
__global__ void transform_frame(const float *input, float *output,
                                unsigned long long delay_cycles) {
  const auto started = clock64();
  while (clock64() - started < delay_cycles) {
  }
  *output = 2.F * *input + 1.F;
}

struct PipelineState {
  size_t                      target       = 128;
  unsigned long long          delay_cycles = 0;
  std::atomic<bool>           hold_source{false};
  std::vector<float>          results;
  std::function<void(size_t)> hook;
};

class PipelineTask final : public ISyncTask {
public:
  PipelineTask(int kind, cudaStream_t stream, std::shared_ptr<PipelineState> state)
      : kind_(kind), stream_(stream), state_(std::move(state)) {}
  bool supports_cuda_graph() const noexcept override { return kind_ == 1; }
  void record_cuda_graph(CudaGraphCtx &ctx) override {
    transform_frame<<<1, 1, 0, ctx.stream>>>(reinterpret_cast<float *>(ctx.inputs[0].data()),
                                             reinterpret_cast<float *>(ctx.outputs[0].data()),
                                             state_->delay_cycles);
  }
  OpResult execute(SyncCtx &ctx) override {
    if (kind_ == 0) {
      while (state_->hold_source.load() && !ctx.cancelled->load())
        std::this_thread::yield();
      if (ctx.cancelled->load())
        return OpResult::Cancelled;
      const auto count = ctx.outputs[0].desc.num_elements();
      fill_frames<<<1, 16, 0, stream_>>>(reinterpret_cast<float *>(ctx.outputs[0].data()), count,
                                         next_);
      next_ += count;
    } else if (kind_ == 1) {
      transform_frame<<<1, 1, 0, stream_>>>(reinterpret_cast<float *>(ctx.inputs[0].data()),
                                            reinterpret_cast<float *>(ctx.outputs[0].data()),
                                            state_->delay_cycles);
    } else {
      float value;
      CUDA_CHECK(cudaMemcpyAsync(&value, ctx.inputs[0].data(), sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      state_->results.push_back(value);
      if (state_->hook)
        state_->hook(state_->results.size());
      if (state_->results.size() == state_->target)
        return OpResult::Eof;
    }
    return OpResult::Ok;
  }

private:
  int                            kind_;
  cudaStream_t                   stream_;
  std::shared_ptr<PipelineState> state_;
  size_t                         next_ = 0;
};

class PipelineFactory final : public ISyncTaskFactory {
public:
  PipelineFactory(int kind, std::shared_ptr<PipelineState> state) : kind_(kind), state_(state) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    if (kind_ == 0 && settings.contains("shape"))
      return {{},
              {TDesc(settings.at("shape").get<std::vector<size_t>>(), DType::F32, MemLoc::Device)},
              {},
              {},
              {false},
              TaskKind::Sync};
    if (kind_ == 0)
      return {
          {},
          {TDesc({settings.is_null() ? size_t{16} : settings.value("batch_size", size_t{16}), 1, 1},
                 DType::F32, MemLoc::Device)},
          {},
          {},
          {false},
          TaskKind::Sync};
    if (kind_ == 1)
      return {{inputs.begin(), inputs.end()},          {inputs[0]}, {},
              std::vector<bool>(inputs.size(), false), {false},     TaskKind::Sync};
    return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<PipelineTask>(kind_, ctx.stream, state_);
  }

private:
  int                            kind_;
  std::shared_ptr<PipelineState> state_;
};

void run_pipeline(size_t graph_limit, bool prequeue, size_t output_capacity, bool pause,
                  bool sliding = false, bool performance = false, bool miss = false,
                  bool stop_paused = false, bool dual_reader = false, bool validity = false) {
  auto state          = std::make_shared<PipelineState>();
  state->target       = performance ? 4096 : 128;
  state->delay_cycles = performance ? 100'000 : 0;
  Registry registry;
  registry.register_sync("source", std::make_unique<PipelineFactory>(0, state));
  registry.register_sync("compute", std::make_unique<PipelineFactory>(1, state));
  registry.register_sync("sink", std::make_unique<PipelineFactory>(2, state));
  registry.register_async("batch", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_async("average", std::make_unique<holotask::asyncs::SlidingAverageFactory>());
  registry.register_async("dual",
                          std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
  GraphSpec spec;
  auto      source = add_vertex(NodeSpec{"source", "source", {}}, spec);
  auto      input =
      add_vertex(NodeSpec{"input", "batch", holotask::asyncs::BatchQueueSettings{32, 1, 1}}, spec);
  auto compute = add_vertex(NodeSpec{"compute", "compute", {}}, spec);
  auto output  = add_vertex(
      NodeSpec{"output", "batch",
               holotask::asyncs::BatchQueueSettings{static_cast<int>(output_capacity), 1, 1}},
      spec);
  auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, spec);
  add_edge(source, input, EdgeSpec{0, 0}, spec);
  add_edge(input, compute, EdgeSpec{0, 0}, spec);
  add_edge(compute, output, EdgeSpec{0, 0}, spec);
  add_edge(output, sink, EdgeSpec{0, 0}, spec);
  if (dual_reader) {
    spec[source].settings = {{"batch_size", 4}};
    spec[input].kind      = "dual";
    spec[input].settings  = holotask::asyncs::DualReaderBatchQueueSettings{8, 64};
    if (validity)
      add_edge(input, compute, EdgeSpec{2, 1}, spec);
  }
  if (sliding) {
    spec[output].kind     = "average";
    spec[output].settings = dual_reader
                                ? holotask::asyncs::SlidingAverageSettings{output_capacity, 64, 0}
                                : holotask::asyncs::SlidingAverageSettings{output_capacity, 3, 2};
  }
  Compiler::Config config;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.max_section_cuda_graphs = graph_limit;
  if (dual_reader && !performance) {
    config.log_dir = std::filesystem::temp_directory_path() / "holoflow-dual-prequeue-diagnostics";
    state->hold_source.store(true);
  }
  auto out = Compiler(registry, config).compile(spec);
  if (!prequeue)
    for (auto &section : out->sections)
      section.supports_prequeue = false; // Test-only serial-graph control.
  Scheduler scheduler(out->graph, out->sections, out->resources);
  if (pause)
    state->hook = [&](size_t frames) {
      if (frames == 5 || frames == 17)
        scheduler.request_pause();
    };
  const auto started = std::chrono::steady_clock::now();
  for (size_t pass = 0; pass < (pause ? 3U : 1U); ++pass) {
    scheduler.start();
    if (pass == 0 && !config.log_dir.empty()) {
      section_diagnostics_file_writer().flush();
      std::ifstream  file(config.log_dir / "section_cuda_graphs.json");
      nlohmann::json reports;
      file >> reports;
      for (const auto &section : out->sections)
        if (out->graph[section.sync_topo.front()].spec.name == "compute") {
          const auto &report = reports.at(section.id);
          EXPECT_EQ(report["prequeue_depth"], prequeue && !validity ? 2 : 1);
          EXPECT_TRUE(report.contains("prequeue_fallback_reason"));
          EXPECT_EQ(report["prequeue_submissions"], 0);
        }
      state->hold_source.store(false);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool stopped = scheduler.stop_requested();
    if (!stopped)
      scheduler.request_stop();
    scheduler.wait();
    ASSERT_TRUE(stopped) << "Pipeline deadlocked";
    if (pause && pass < 2)
      ASSERT_EQ(state->results.size(), pass == 0 ? 5U : 17U);
    if (pause && pass == 0 && miss)
      for (const auto &section : out->sections)
        if (out->graph[section.sync_topo.front()].spec.name == "compute")
          out->resources.section_cuda_graphs.at(section.id)->tuple_indices.clear();
    if (stop_paused) {
      scheduler.request_stop();
      scheduler.wait();
      state->target = 5;
      break;
    }
  }
  ASSERT_EQ(state->results.size(), state->target);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i],
                    2.F * static_cast<float>(i) + (sliding ? dual_reader ? 64.F : 7.F : 1.F));
  for (const auto &section : out->sections) {
    if (out->graph[section.sync_topo.front()].spec.name != "compute")
      continue;
    const auto &graphs = *out->resources.section_cuda_graphs.at(section.id);
    const auto  report = graphs.snapshot();
    if (prequeue && !validity) {
      EXPECT_EQ(report["prequeue_depth"], 2);
      EXPECT_GT(graphs.prequeue_submissions.load(), 0U);
      EXPECT_EQ(graphs.prequeue_submissions.load(), graphs.prequeue_retirements.load());
      EXPECT_EQ(graphs.prequeue_peak.load(), output_capacity == 1 ? 1U : 2U);
    } else {
      EXPECT_EQ(graphs.prequeue_submissions.load(), 0U);
      if (validity)
        EXPECT_NE(report["prequeue_fallback_reason"].get<std::string>().find("non-owned output"),
                  std::string::npos);
    }
    if (dual_reader && sliding && graph_limit && !validity)
      EXPECT_EQ(graphs.variant_count, 792U);
    EXPECT_EQ(graphs.pointer_misses.load(), 0U);
    EXPECT_EQ(graphs.tuple_misses.load(), miss ? 1U : 0U);
  }
  if (performance)
    std::cout
        << "prequeue=" << prequeue << " frames_per_second="
        << static_cast<double>(state->target) /
               std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        << '\n';
}

std::vector<float> run_pca_pipeline(size_t limit, bool prequeue, bool pause = false,
                                    bool miss = false, bool stop_paused = false) {
  auto state    = std::make_shared<PipelineState>();
  state->target = 64;
  Registry registry;
  registry.register_sync("source", std::make_unique<PipelineFactory>(0, state));
  registry.register_sync("sink", std::make_unique<PipelineFactory>(2, state));
  registry.register_sync("pca", std::make_unique<holotask::syncs::PcaFactory>());
  registry.register_async("batch", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_async("dual",
                          std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
  GraphSpec spec;
  // One producer commit makes two complete PCA inputs available, so depth two does not depend
  // on an upstream worker winning a scheduling race. Each PCA projects a different 2x2 matrix.
  auto source = add_vertex(NodeSpec{"source", "source", {{"shape", {2, 2, 1, 2}}}}, spec);
  auto input =
      add_vertex(NodeSpec{"input", "batch", holotask::asyncs::BatchQueueSettings{8, 1, 1}}, spec);
  auto pca    = add_vertex(NodeSpec{"pca", "pca", holotask::syncs::PcaSettings{1, 2}}, spec);
  auto output = add_vertex(
      NodeSpec{"output", "dual", holotask::asyncs::DualReaderBatchQueueSettings{8, 5}}, spec);
  auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, spec);
  add_edge(source, input, EdgeSpec{0, 0}, spec);
  add_edge(input, pca, EdgeSpec{0, 0}, spec);
  add_edge(pca, output, EdgeSpec{0, 0}, spec);
  add_edge(output, sink, EdgeSpec{0, 0}, spec);
  Compiler::Config config;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.max_section_cuda_graphs = limit;
  auto out                       = Compiler(registry, config).compile(spec);
  if (!prequeue)
    for (auto &section : out->sections)
      section.supports_prequeue = false;
  Scheduler scheduler(out->graph, out->sections, out->resources);
  if (pause)
    state->hook = [&](size_t frames) {
      if (frames == 5 || frames == 17)
        scheduler.request_pause();
    };
  for (size_t pass = 0; pass < (pause ? 3U : 1U); ++pass) {
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool stopped = scheduler.stop_requested();
    if (!stopped)
      scheduler.request_stop();
    scheduler.wait();
    EXPECT_TRUE(stopped) << "PCA pipeline deadlocked";
    if (!stopped)
      break;
    if (pause && pass < 2)
      EXPECT_EQ(state->results.size(), pass == 0 ? 5U : 17U);
    if (pause && pass == 0 && miss)
      for (const auto &section : out->sections)
        if (out->graph[section.sync_topo.front()].spec.name == "pca")
          out->resources.section_cuda_graphs.at(section.id)->tuple_indices.clear();
    if (stop_paused) {
      state->target = 5;
      break;
    }
  }
  scheduler.request_stop();
  scheduler.wait();
  EXPECT_EQ(state->results.size(), state->target);
  for (const auto &section : out->sections)
    if (out->graph[section.sync_topo.front()].spec.name == "pca") {
      const auto &graphs = *out->resources.section_cuda_graphs.at(section.id);
      const auto  report = graphs.snapshot();
      EXPECT_EQ(report["prequeue_depth"], prequeue ? 2 : 1);
      EXPECT_EQ(graphs.prequeue_peak.load(), prequeue ? 2U : 0U);
      EXPECT_EQ(graphs.prequeue_submissions.load(), graphs.prequeue_retirements.load());
      EXPECT_EQ(graphs.pointer_misses.load(), 0U);
      EXPECT_EQ(graphs.tuple_misses.load(), miss ? 1U : 0U);
    }
  return state->results;
}

TEST(PrequeueScheduler, PcaPublishesTwoInputsAcrossPauseFallbackAndStop) {
  const auto expected = run_pca_pipeline(0, false);
  const auto compare  = [&](const auto &actual) {
    ASSERT_LE(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1e-3F * std::max(1.F, std::abs(expected[i])))
          << "frame " << i;
  };
  compare(run_pca_pipeline(128, false));
  compare(run_pca_pipeline(128, true, true));
  compare(run_pca_pipeline(128, true, true, true));
  compare(run_pca_pipeline(128, true, true, false, true));
}

TEST(PrequeueScheduler, MatchesSerialGraphsAndOrdinaryExecution) {
  run_pipeline(0, false, 4, false);
  run_pipeline(512, false, 4, false);
  run_pipeline(512, true, 4, false);
}
TEST(PrequeueScheduler, OrdinaryFallbackRetainsTwoSlotsWithoutPreparedGraphs) {
  run_pipeline(0, true, 4, true);
  run_pipeline(1, true, 4, true);
}
TEST(PrequeueScheduler, RetiresWhenOnlyOneOutputSlotIsAvailable) {
  run_pipeline(512, true, 1, false);
}
TEST(PrequeueScheduler, RetainsBothIterationsAcrossPauseAndResume) {
  run_pipeline(512, true, 4, true);
}
TEST(PrequeueScheduler, SlidingAverageAcrossPauseAndResume) {
  run_pipeline(512, true, 4, true, true);
}
TEST(PrequeueScheduler, GraphMissDrainsPendingWorkBeforeFallback) {
  run_pipeline(512, true, 4, true, false, false, true);
}
TEST(PrequeueScheduler, StopWhilePausedReturnsPendingReservations) {
  run_pipeline(512, true, 4, true, false, false, false, true);
}
TEST(PrequeueScheduler, PerformanceSaturatedPipeline) {
  run_pipeline(512, false, 31, false, false, true);
  run_pipeline(512, true, 31, false, false, true);
}
TEST(PrequeueScheduler, DualReaderSection3Retains792VariantsAcrossPauseAndResume) {
  run_pipeline(0, false, 8, false, true, false, false, false, true);
  run_pipeline(2048, false, 8, false, true, false, false, false, true);
  run_pipeline(2048, true, 8, true, true, false, false, false, true);
}
TEST(PrequeueScheduler, DualReaderSharedValidityForcesSerialConsumption) {
  run_pipeline(2048, true, 8, false, true, false, false, false, true, true);
}
TEST(PrequeueScheduler, DualReaderGraphMissAndStopWhilePausedReturnReservations) {
  run_pipeline(2048, true, 8, true, true, false, true, false, true);
  run_pipeline(2048, true, 8, true, true, false, false, true, true);
}
TEST(PrequeueScheduler, PerformanceDualReaderSection3) {
  run_pipeline(2048, false, 8, false, true, true, false, false, true);
  run_pipeline(2048, true, 8, false, true, true, false, false, true);
}

} // namespace
