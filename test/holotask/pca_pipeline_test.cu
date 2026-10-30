// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <functional>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <thread>

#include "curaii/cuda.hh"
#include "holoflow/runtime/compiler.hh"
#include "holonp/copy.hh"
#include "holonp/fftshift.hh"
#include "holonp/mean.hh"
#include "holonp/reshape.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/dual_reader_batch_queue.hh"
#include "holotask/asyncs/pca_decomposition_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;
using QueueSettings = holotask::asyncs::PcaDecompositionQueueSettings;

class QueueStorage final : public IOStorageAccess {
public:
  explicit QueueStorage(const InferResult &infer) {
    for (const auto &desc : infer.input_descs)
      inputs.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
    for (const auto &desc : infer.output_descs)
      outputs.push_back({desc.mem_loc, desc.num_bytes(), nullptr});
  }
  Storage             &owned_input_storage(size_t index) override { return inputs.at(index); }
  Storage             &owned_output_storage(size_t index) override { return outputs.at(index); }
  std::vector<Storage> inputs, outputs;
};

TEST(PcaPipeline, ValidatesDescriptorsAndPreservesGrouping) {
  holotask::asyncs::PcaDecompositionQueueFactory queue;
  const std::array input{TDesc({1, 32, 320, 512}, DType::F32, MemLoc::Device)};
  const auto       infer = queue.infer(input, QueueSettings{});
  EXPECT_EQ(infer.output_descs[0].shape, (std::vector<size_t>{4, 32, 320, 512}));
  EXPECT_EQ(infer.output_descs[1].shape, (std::vector<size_t>{4, 32, 32}));
  EXPECT_EQ(infer.owned_input_pointer_counts[0], 12U);
  EXPECT_EQ(infer.owned_output_pointer_counts[0], 3U);
  EXPECT_TRUE(infer.supports_prequeued_push);
  EXPECT_TRUE(infer.supports_prequeued_pop);
  EXPECT_THROW(queue.infer(input, QueueSettings{0, 3}), std::invalid_argument);
  EXPECT_THROW(queue.infer(input, QueueSettings{4, 0}), std::invalid_argument);
  for (auto bad : {TDesc({2, 32, 2, 2}, DType::F32, MemLoc::Device),
                   TDesc({1, 32, 2, 2}, DType::U8, MemLoc::Device),
                   TDesc({1, 32, 2, 2}, DType::F32, MemLoc::Host),
                   TDesc({1, 0, 2, 2}, DType::F32, MemLoc::Device),
                   TDesc({1, 32, 2, 2}, DType::F32, MemLoc::Device, size_t{4})}) {
    const std::array inputs{bad};
    EXPECT_THROW(queue.infer(inputs, QueueSettings{}), std::invalid_argument);
  }
  holotask::syncs::PcaProjectionFactory projection;
  EXPECT_EQ(projection.infer(infer.output_descs, holotask::syncs::PcaSettings{1, 8})
                .output_descs[0]
                .shape,
            (std::vector<size_t>{4, 7, 320, 512}));
  auto bad = infer.output_descs;
  bad[1]   = TDesc({3, 32, 32}, DType::F32, MemLoc::Device);
  EXPECT_THROW(projection.infer(bad, holotask::syncs::PcaSettings{1, 8}), std::invalid_argument);
}

// A host callback gates covariance's stream without spending GPU resources. The queue's
// private solver must wait for it, and an unready group must never escape through pop.
void CUDART_CB wait_for_gate(void *data) {
  auto &gate = *static_cast<std::atomic<bool> *>(data);
  while (!gate.load())
    std::this_thread::yield();
}

TEST(PcaPipeline, ReservationsRetainBothOutputsAndRejectUnfinishedSolver) {
  holotask::asyncs::PcaDecompositionQueueFactory factory;
  const std::array    inputs{TDesc({1, 2, 1, 2}, DType::F32, MemLoc::Device)};
  const QueueSettings settings{1, 3};
  const auto          infer = factory.infer(inputs, settings);
  curaii::CudaStream  producer(cudaStreamNonBlocking), consumer(cudaStreamNonBlocking);
  auto                task = factory.create(inputs, settings, {producer.get(), consumer.get()});
  QueueStorage        storage(infer);
  task->bind_storage_access(&storage);
  std::array        in{TView{inputs[0], &storage.inputs[0]}};
  std::array        out{TView{infer.output_descs[0], &storage.outputs[0]},
                        TView{infer.output_descs[1], &storage.outputs[1]}};
  std::atomic<bool> cancelled{false};
  AsyncPushCtx      push{in, &cancelled};
  AsyncPopCtx       pop{out, &cancelled};
  auto              values = curaii::make_unique_host_ptr<float>(4);
  values.get()[0]          = 1;
  values.get()[1]          = 0;
  values.get()[2]          = 0;
  values.get()[3]          = 2;
  AsyncTicket first, second, third, read;
  ASSERT_EQ(task->reserve_push(push, first), OpResult::Ok);
  const auto first_pointer = in[0].data();
  CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get(), 4 * sizeof(float), cudaMemcpyHostToDevice,
                             producer.get()));
  CUDA_CHECK(cudaStreamSynchronize(producer.get()));
  std::atomic<bool> gate{false};
  CUDA_CHECK(cudaLaunchHostFunc(producer.get(), wait_for_gate, &gate));
  task->enqueue_push(first);
  // Simulate a delayed solver while making its group visible to the reader. Pop must still
  // query the completion event rather than rely only on the committed input cursor.
  task->commit_push(first);
  const auto unfinished = task->reserve_pop(pop, read);
  gate.store(true); // Release before any assertion or teardown can leave the callback blocked.
  EXPECT_EQ(unfinished, OpResult::NotReady);
  ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
  const auto second_pointer = in[0].data();
  task->cancel_push(second);
  ASSERT_EQ(task->reserve_push(push, second), OpResult::Ok);
  EXPECT_EQ(in[0].data(), second_pointer);
  // Reverse the eigenvalue ordering in the next slot, so sharing/overwriting an eigenvector
  // buffer cannot accidentally pass because the two decompositions have the same basis.
  values.get()[0] = 3;
  values.get()[3] = 1;
  CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get(), 4 * sizeof(float), cudaMemcpyHostToDevice,
                             producer.get()));
  task->enqueue_push(second);
  EXPECT_THROW(task->cancel_push(second), std::logic_error);
  CUDA_CHECK(cudaStreamSynchronize(producer.get()));
  task->commit_push(second);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  auto       result   = OpResult::NotReady;
  while (result == OpResult::NotReady && std::chrono::steady_clock::now() < deadline)
    result = task->reserve_pop(pop, read);
  ASSERT_EQ(result, OpResult::Ok);
  EXPECT_EQ(out[0].data(), first_pointer);
  std::array<float, 4> preserved_x, vectors;
  CUDA_CHECK(
      cudaMemcpy(preserved_x.data(), out[0].data(), sizeof(preserved_x), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vectors.data(), out[1].data(), sizeof(vectors), cudaMemcpyDeviceToHost));
  EXPECT_FLOAT_EQ(preserved_x[0], 1);
  EXPECT_FLOAT_EQ(preserved_x[3], 2);
  EXPECT_NEAR(std::abs(vectors[0]), 1.F, 1e-4F);
  EXPECT_NEAR(std::abs(vectors[3]), 1.F, 1e-4F);
  task->cancel_pop(read);
  ASSERT_EQ(task->try_pop(pop), OpResult::Ok);
  ASSERT_EQ(task->reserve_push(push, third), OpResult::Ok);
  task->cancel_push(third);
  // Fill the final slot. Releasing X alone must not return the first slot to the writer.
  ASSERT_TRUE(task->acquire_input(0));
  CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get(), 4 * sizeof(float), cudaMemcpyHostToDevice,
                             producer.get()));
  ASSERT_EQ(task->try_push(push), OpResult::Ok);
  EXPECT_FALSE(task->acquire_input(0));
  task->release_output(0);
  EXPECT_FALSE(task->acquire_input(0));
  task->release_output(1);
  ASSERT_TRUE(task->acquire_input(0));
  EXPECT_EQ(in[0].data(), first_pointer);
  const auto second_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  result                     = OpResult::NotReady;
  while (result == OpResult::NotReady && std::chrono::steady_clock::now() < second_deadline)
    result = task->reserve_pop(pop, read);
  ASSERT_EQ(result, OpResult::Ok);
  EXPECT_EQ(out[0].data(), second_pointer);
  CUDA_CHECK(cudaMemcpy(vectors.data(), out[1].data(), sizeof(vectors), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(std::abs(vectors[0]), 0.F, 1e-4F);
  EXPECT_NEAR(std::abs(vectors[1]), 1.F, 1e-4F);
  task->release_pop(read);
  cancelled.store(true);
  EXPECT_EQ(task->reserve_push(push, third), OpResult::Cancelled);
  EXPECT_EQ(task->reserve_pop(pop, read), OpResult::Cancelled);
  // Task destruction drains any remaining HEEV work, including unpopped groups.
}

TEST(PcaPipeline, ReplacementDrainsUnpoppedGroupsAndDiscardsPartialInput) {
  holotask::asyncs::PcaDecompositionQueueFactory factory;
  const std::array    inputs{TDesc({1, 2, 1, 2}, DType::F32, MemLoc::Device)};
  const QueueSettings settings{4, 3};
  const auto          infer = factory.infer(inputs, settings);
  curaii::CudaStream  producer(cudaStreamNonBlocking), consumer(cudaStreamNonBlocking);
  auto                task = factory.create(inputs, settings, {producer.get(), consumer.get()});
  QueueStorage        storage(infer);
  task->bind_storage_access(&storage);
  std::array        in{TView{inputs[0], &storage.inputs[0]}};
  std::array        out{TView{infer.output_descs[0], &storage.outputs[0]},
                        TView{infer.output_descs[1], &storage.outputs[1]}};
  std::atomic<bool> cancelled{false};
  AsyncPushCtx      push{in, &cancelled};
  AsyncPopCtx       pop{out, &cancelled};
  auto              values = curaii::make_unique_host_ptr<float>(4);
  values.get()[0]          = 1;
  values.get()[1]          = 0;
  values.get()[2]          = 0;
  values.get()[3]          = 2;
  for (size_t i = 0; i < 7; ++i) {
    ASSERT_TRUE(task->acquire_input(0));
    CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get(), 4 * sizeof(float),
                               cudaMemcpyHostToDevice, producer.get()));
    CUDA_CHECK(cudaStreamSynchronize(producer.get()));
    ASSERT_EQ(task->try_push(push), OpResult::Ok);
  }
  // One submitted decomposition and three incomplete windows are discarded by replacement.
  task = factory.update(std::move(task), inputs, settings, {producer.get(), consumer.get()});
  task->bind_storage_access(&storage);
  AsyncTicket read;
  EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
  EXPECT_EQ(task->owned_input_pointers(0)->size(), 12U);
  EXPECT_EQ(task->owned_output_pointer_sequence(0)->cycle,
            task->owned_output_pointer_sequence(1)->cycle);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_TRUE(task->acquire_input(0));
    CUDA_CHECK(cudaMemcpyAsync(in[0].data(), values.get(), 4 * sizeof(float),
                               cudaMemcpyHostToDevice, producer.get()));
    CUDA_CHECK(cudaStreamSynchronize(producer.get()));
    ASSERT_EQ(task->try_push(push), OpResult::Ok);
    if (i < 3)
      EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
  }
  std::array<AsyncTicket, 2> reads;
  std::array<std::byte *, 2> pointers;
  for (size_t i = 0; i < 2; ++i) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto       result   = OpResult::NotReady;
    while (result == OpResult::NotReady && std::chrono::steady_clock::now() < deadline)
      result = task->reserve_pop(pop, reads[i]);
    ASSERT_EQ(result, OpResult::Ok);
    pointers[i] = out[0].data();
  }
  EXPECT_NE(pointers[0], pointers[1]);
  EXPECT_THROW(task->release_pop(reads[1]), std::logic_error);
  task->release_pop(reads[0]);
  task->release_pop(reads[1]);
  EXPECT_EQ(task->reserve_pop(pop, read), OpResult::NotReady);
}

__global__ void generate_inputs(float *output, size_t count, size_t samples, int features,
                                size_t first_window) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const auto window  = first_window + i / (samples * features);
  const auto feature = (i / samples) % features;
  const auto sample  = i % samples;
  // Distinct scales and spatial patterns exercise both original-X retention and eigenvectors.
  const float scale = 1.F + static_cast<float>(window % 17) / 32.F;
  output[i]         = scale * (sample % features == feature
                                   ? 2.F + feature / 32.F
                                   : 0.1F * sinf(static_cast<float>((sample + 1) * (feature + 1))));
}

__global__ void generate_u8_inputs(unsigned char *output, size_t count, size_t first) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    unsigned int v = static_cast<unsigned int>(i + first * 5242880 + 1);
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    v ^= v >> 16;
    output[i] = static_cast<unsigned char>(v);
  }
}

struct PipelineState {
  size_t                                features = 2, height = 1, width = 2, target = 64;
  bool                                  performance = false;
  bool                                  mixed       = false;
  size_t                                duration_ms = 10000;
  std::vector<float>                    values;
  std::function<void(size_t)>           hook;
  size_t                                consumed = 0;
  std::chrono::steady_clock::time_point started;
  double                                windows_per_second = 0;
};

class EndpointTask final : public ISyncTask {
public:
  EndpointTask(bool source, cudaStream_t stream, std::shared_ptr<PipelineState> state)
      : source_(source), stream_(stream), state_(std::move(state)) {}
  OpResult execute(SyncCtx &ctx) override {
    if (source_) {
      const auto count = ctx.outputs[0].desc.num_elements();
      if (state_->mixed)
        generate_u8_inputs<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream_>>>(
            reinterpret_cast<unsigned char *>(ctx.outputs[0].data()), count, next_);
      else
        generate_inputs<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream_>>>(
            reinterpret_cast<float *>(ctx.outputs[0].data()), count, state_->height * state_->width,
            static_cast<int>(state_->features), next_);
      CUDA_CHECK(cudaGetLastError());
      next_ += 4;
    } else {
      float value;
      if (ctx.inputs[0].desc.dtype == DType::U8) {
        unsigned char byte;
        CUDA_CHECK(
            cudaMemcpyAsync(&byte, ctx.inputs[0].data(), 1, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        value = float(byte);
      } else if (state_->mixed) {
        __half h;
        CUDA_CHECK(
            cudaMemcpyAsync(&h, ctx.inputs[0].data(), sizeof(h), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        value = __half2float(h);
        EXPECT_TRUE(std::isfinite(value));
      } else
        CUDA_CHECK(cudaMemcpyAsync(&value, ctx.inputs[0].data(), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      ++state_->consumed;
      if (!state_->performance) {
        const auto count  = ctx.inputs[0].desc.num_elements();
        const auto offset = state_->values.size();
        state_->values.resize(offset + (state_->mixed ? 1 : count));
        if (state_->mixed)
          state_->values[offset] = value;
        else
          CUDA_CHECK(cudaMemcpyAsync(state_->values.data() + offset, ctx.inputs[0].data(),
                                     count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (state_->hook)
          state_->hook(state_->consumed);
        if (state_->consumed == state_->target)
          return OpResult::Eof;
      } else {
        const auto now = std::chrono::steady_clock::now();
        if (state_->consumed == 256)
          state_->started = now;
        if (state_->consumed > 256 &&
            now - state_->started >= std::chrono::milliseconds(state_->duration_ms)) {
          state_->windows_per_second = (state_->consumed - 256) /
                                       std::chrono::duration<double>(now - state_->started).count();
          return OpResult::Eof;
        }
      }
    }
    return OpResult::Ok;
  }

private:
  bool                           source_;
  cudaStream_t                   stream_;
  std::shared_ptr<PipelineState> state_;
  size_t                         next_ = 0;
};

class EndpointFactory final : public ISyncTaskFactory {
public:
  EndpointFactory(bool source, std::shared_ptr<PipelineState> state)
      : source_(source), state_(std::move(state)) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (source_)
      return {{},
              {TDesc({4, state_->features, state_->height, state_->width},
                     state_->mixed ? DType::U8 : DType::F32, MemLoc::Device)},
              {},
              {},
              {false},
              TaskKind::Sync};
    return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<EndpointTask>(source_, ctx.stream, state_);
  }

private:
  bool                           source_;
  std::shared_ptr<PipelineState> state_;
};

std::shared_ptr<PipelineState> run_pipeline(bool pipeline, size_t graph_limit, bool pause = false,
                                            bool miss = false, bool stop_paused = false,
                                            size_t features = 2, bool performance = false,
                                            size_t duration_ms      = 10000,
                                            size_t benchmark_height = 320, bool mixed = false) {
  auto state         = std::make_shared<PipelineState>();
  state->mixed       = mixed;
  state->features    = features;
  state->width       = (performance || mixed) ? 512 : features;
  state->height      = (performance || mixed) ? benchmark_height : 1;
  state->performance = performance;
  state->duration_ms = duration_ms;
  if (features >= 256)
    state->target = 16;
  Registry registry;
  registry.register_sync("source", std::make_unique<EndpointFactory>(true, state));
  registry.register_sync("sink", std::make_unique<EndpointFactory>(false, state));
  registry.register_sync("copy", std::make_unique<holonp::CopyFactory>());
  registry.register_sync("pca", std::make_unique<holotask::syncs::PcaFactory>());
  registry.register_sync("projection", std::make_unique<holotask::syncs::PcaProjectionFactory>());
  registry.register_async("batch", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_async("decomposition",
                          std::make_unique<holotask::asyncs::PcaDecompositionQueueFactory>());
  GraphSpec spec;
  auto      source = add_vertex(NodeSpec{"source", "source", {}}, spec);
  auto      input =
      add_vertex(NodeSpec{"input", "batch", holotask::asyncs::BatchQueueSettings{8, 1, 1}}, spec);
  auto copy    = add_vertex(NodeSpec{"copy", "copy", holonp::CopySettings{}}, spec);
  auto grouped = add_vertex(
      pipeline ? NodeSpec{"grouped", "decomposition", QueueSettings{4, 3, mixed}}
               : NodeSpec{"grouped", "batch", holotask::asyncs::BatchQueueSettings{8, 4, 4}},
      spec);
  const holotask::syncs::PcaSettings selection{mixed ? 0 : 1, (performance || mixed) ? 16 : 2,
                                               mixed, mixed ? DType::F16 : DType::F32};
  auto compute = add_vertex(NodeSpec{"compute", pipeline ? "projection" : "pca", selection}, spec);
  auto output =
      add_vertex(NodeSpec{"output", "batch", holotask::asyncs::BatchQueueSettings{8, 1, 1}}, spec);
  auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, spec);
  add_edge(source, input, EdgeSpec{0, 0}, spec);
  add_edge(input, copy, EdgeSpec{0, 0}, spec);
  add_edge(copy, grouped, EdgeSpec{0, 0}, spec);
  add_edge(grouped, compute, EdgeSpec{0, 0}, spec);
  if (pipeline)
    add_edge(grouped, compute, EdgeSpec{1, 1}, spec);
  add_edge(compute, output, EdgeSpec{0, 0}, spec);
  add_edge(output, sink, EdgeSpec{0, 0}, spec);
  Compiler::Config config;
  config.enable_profiling                        = false;
  config.verbose_tracing                         = false;
  config.max_section_cuda_graphs                 = graph_limit;
  auto                                  compiled = Compiler(registry, config).compile(spec);
  static const nvtx3::registered_string pipeline_name{"PCA pipeline run"};
  static const nvtx3::registered_string baseline_name{"PCA baseline run"};
  nvtx3::scoped_range                   measurement_range{pipeline ? pipeline_name : baseline_name};
  Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  if (pause)
    state->hook = [&](size_t consumed) {
      if (consumed == 5 || consumed == 17)
        scheduler.request_pause();
    };
  for (size_t pass = 0; pass < (pause ? 3U : 1U); ++pass) {
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(performance ? duration_ms + 30000 : 20000);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool stopped = scheduler.stop_requested();
    if (!stopped)
      scheduler.request_stop();
    scheduler.wait();
    EXPECT_TRUE(stopped) << "PCA pipeline deadlocked";
    if (!stopped)
      break;
    if (pause && pass == 0 && miss)
      for (const auto &section : compiled->sections)
        if (compiled->graph[section.sync_topo.front()].spec.name == "compute")
          compiled->resources.section_cuda_graphs.at(section.id)->tuple_indices.clear();
    if (stop_paused)
      break;
  }
  if (stop_paused) {
    scheduler.request_stop();
    scheduler.wait();
  }
  if (!performance)
    EXPECT_EQ(state->consumed, stop_paused ? 5U : state->target);
  if (graph_limit && pipeline)
    for (const auto &section : compiled->sections)
      if (compiled->graph[section.sync_topo.front()].spec.name == "compute") {
        const auto &graphs = *compiled->resources.section_cuda_graphs.at(section.id);
        EXPECT_GT(graphs.launches.load(), 0U);
        EXPECT_GT(graphs.prequeue_submissions.load(), 0U);
        EXPECT_EQ(graphs.prequeue_submissions.load(), graphs.prequeue_retirements.load());
        EXPECT_EQ(graphs.tuple_misses.load(), miss ? 1U : 0U);
      }
  return state;
}

TEST(PcaPipeline, MatchesSynchronousPcaAcrossWraparoundPauseFallbackAndStop) {
  const auto baseline = run_pipeline(false, 0);
  for (const auto &actual :
       {run_pipeline(true, 0), run_pipeline(true, 128), run_pipeline(true, 128, true),
        run_pipeline(true, 128, true, true), run_pipeline(true, 128, true, false, true)}) {
    ASSERT_LE(actual->values.size(), baseline->values.size());
    for (size_t i = 0; i < actual->values.size(); ++i)
      EXPECT_NEAR(actual->values[i], baseline->values[i],
                  2e-3F * std::max(1.F, std::abs(baseline->values[i])))
          << "window " << i;
  }
}

TEST(PcaPipeline, MatchesCusolverFallback) {
  const auto baseline = run_pipeline(false, 0, false, false, false, 256);
  const auto actual   = run_pipeline(true, 128, false, false, false, 256);
  ASSERT_EQ(actual->values.size(), baseline->values.size());
  for (size_t i = 0; i < actual->values.size(); ++i)
    EXPECT_NEAR(actual->values[i], baseline->values[i], 2e-3F);
}

size_t benchmark_option(const char *name, size_t fallback) {
  char  *value  = nullptr;
  size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
    return fallback;
  const auto result = std::stoull(value);
  std::free(value);
  return static_cast<size_t>(result);
}

TEST(PcaPipeline, PerformanceOverlap) {
  const auto duration      = benchmark_option("HOLOFLOW_PCA_BENCHMARK_MS", 10000);
  const auto runs          = benchmark_option("HOLOFLOW_PCA_BENCHMARK_RUNS", 5);
  const auto selected_mode = benchmark_option("HOLOFLOW_PCA_BENCHMARK_MODE", 0);
  const auto height        = benchmark_option("HOLOFLOW_PCA_BENCHMARK_HEIGHT", 320);
  std::array<std::vector<double>, 2> rates;
  for (size_t run = 0; run < runs; ++run)
    for (size_t order = 0; order < 2; ++order) {
      const size_t mode = (run + order) % 2; // Alternate order to reduce clock/thermal bias.
      if (selected_mode && selected_mode != mode + 1)
        continue;
      const auto result =
          run_pipeline(mode != 0, 128, false, false, false, 32, true, duration, height);
      rates[mode].push_back(result->windows_per_second);
      EXPECT_GT(result->windows_per_second, 0);
      std::cout << "PCA mode=" << (mode ? "pipeline" : "baseline") << " run=" << run
                << " windows_per_second=" << result->windows_per_second << std::endl;
    }
  for (size_t mode = 0; mode < 2; ++mode) {
    std::sort(rates[mode].begin(), rates[mode].end());
    if (!rates[mode].empty())
      std::cout << "PCA mode=" << (mode ? "pipeline" : "baseline")
                << " median=" << rates[mode][rates[mode].size() / 2]
                << " min=" << rates[mode].front() << " max=" << rates[mode].back() << std::endl;
  }
}

} // namespace

TEST(PcaPipeline, F16QueueWraparoundAndGraphReplay) {
  const auto actual = run_pipeline(true, 128, false, false, false, 32, false, 10000, 320, true);
  EXPECT_EQ(actual->consumed, 64u);
  EXPECT_TRUE(std::all_of(actual->values.begin(), actual->values.end(),
                          [](float v) { return std::isfinite(v); }));
}

TEST(ProductionF16, RecordingBasicPipeline) {
  char  *path   = nullptr;
  size_t length = 0;
  _dupenv_s(&path, &length, "HOLOFLOW_F16_RECORDING_PATH");
  if (!path)
    GTEST_SKIP() << "Set HOLOFLOW_F16_RECORDING_PATH for the recording smoke/performance run";
  std::string recording(path);
  std::free(path);
  const auto mode     = benchmark_option("HOLOFLOW_F16_RECORDING_MODE", 3);
  const bool half_pca = mode == 1 || mode == 3, half_fresnel = mode == 2 || mode == 3;
  auto       state   = std::make_shared<PipelineState>();
  state->performance = true;
  state->duration_ms = benchmark_option("HOLOFLOW_F16_RECORDING_MS", 1000);
  Registry registry;
  registry.register_sync("source", std::make_unique<holotask::sources::HolofileFactory>());
  registry.register_sync("memcpy", std::make_unique<holotask::syncs::MemcpyFactory>());
  registry.register_sync("convert", std::make_unique<holotask::syncs::ConversionFactory>());
  registry.register_sync("reshape", std::make_unique<holonp::ReshapeFactory>());
  registry.register_sync("projection", std::make_unique<holotask::syncs::PcaProjectionFactory>());
  registry.register_sync("fresnel", std::make_unique<holotask::syncs::FresnelDiffractionFactory>());
  registry.register_sync("mean", std::make_unique<holonp::MeanFactory>());
  registry.register_sync("fftshift", std::make_unique<holonp::FFTShiftFactory>());
  registry.register_sync("sink", std::make_unique<EndpointFactory>(false, state));
  registry.register_async("batch", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_async("decomposition",
                          std::make_unique<holotask::asyncs::PcaDecompositionQueueFactory>());
  registry.register_async("dual",
                          std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
  registry.register_async("average", std::make_unique<holotask::asyncs::SlidingAverageFactory>());
  GraphSpec spec;
  auto      node = [&](const char *name, const char *type, nlohmann::json settings) {
    return add_vertex(NodeSpec{name, type, settings}, spec);
  };
  auto edge = [&](auto from, auto to, int out = 0, int in = 0) {
    add_edge(from, to, EdgeSpec{out, in}, spec);
  };
  auto source = node("source", "source",
                     holotask::sources::HolofileSettings{
                         .path        = recording,
                         .load_kind   = holotask::sources::HolofileSettings::LoadKind::CPUCached,
                         .start_frame = 0,
                         .end_frame   = 16384,
                         .batch_size  = 32,
                         .keep_cursor = false});
  auto copy =
      node("memcpy", "memcpy",
           holotask::syncs::MemcpySettings{holotask::syncs::MemcpySettings::Target::Device});
  edge(source, copy);
  auto input = node("input", "batch", holotask::asyncs::BatchQueueSettings{1024, 32, 32});
  edge(copy, input);
  auto previous = input;
  if (half_pca) {
    auto boundary =
        node("queue_copy", "memcpy",
             holotask::syncs::MemcpySettings{holotask::syncs::MemcpySettings::Target::Device});
    edge(previous, boundary);
    previous = boundary;
  }
  if (!half_pca) {
    auto convert = node(
        "widen", "convert",
        holotask::syncs::ConversionSettings{holotask::syncs::ConversionSettings::Target::F32,
                                            holotask::syncs::ConversionSettings::Strategy::Real});
    edge(previous, convert);
    previous = convert;
  }
  auto reshape = node("reshape", "reshape", holonp::ReshapeSettings{{1, 32, 320, 512}, false});
  edge(previous, reshape);
  auto decomposition = node("decomposition", "decomposition", QueueSettings{4, 3, half_pca});
  edge(reshape, decomposition);
  auto projection =
      node("projection", "projection",
           holotask::syncs::PcaSettings{0, 16, half_pca, half_pca ? DType::F16 : DType::F32});
  edge(decomposition, projection);
  edge(decomposition, projection, 1, 1);
  auto dual = node("dual", "dual", holotask::asyncs::DualReaderBatchQueueSettings{8, 64});
  edge(projection, dual);
  auto fresnel =
      node("fresnel", "fresnel",
           holotask::syncs::FresnelDiffractionSettings{.lambda           = 851e-9f,
                                                       .dx               = 20e-6f,
                                                       .dy               = 20e-6f,
                                                       .z                = .49f,
                                                       .axes             = {-2, -1},
                                                       .skip_phase_shift = true,
                                                       .output_magnitude = true,
                                                       .use_fused_f16    = half_fresnel,
                                                       .reduce_magnitude = half_fresnel});
  edge(dual, fresnel);
  previous = fresnel;
  if (!half_fresnel) {
    auto mean = node("component_mean", "mean", holonp::MeanSettings{{-3}, true});
    edge(previous, mean);
    previous = mean;
  }
  auto shift = node("shift", "fftshift", holonp::FFTShiftSettings{{-2, -1}});
  edge(previous, shift);
  auto mean = node("mean", "mean", holonp::MeanSettings{{0}, false});
  edge(shift, mean);
  auto average = node("average", "average", holotask::asyncs::SlidingAverageSettings{8, 64, 0});
  edge(mean, average);
  auto convert = node(
      "display_scale", "convert",
      holotask::syncs::ConversionSettings{holotask::syncs::ConversionSettings::Target::U8,
                                          holotask::syncs::ConversionSettings::Strategy::Scaled});
  edge(average, convert);
  auto output = node("output", "batch", holotask::asyncs::BatchQueueSettings{32, 1, 1});
  edge(convert, output);
  auto sink = node("sink", "sink", nlohmann::json::object());
  edge(output, sink);
  Compiler::Config config;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.max_section_cuda_graphs = 128;
  auto      compiled             = Compiler(registry, config).compile(spec);
  Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  scheduler.start();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool stopped = scheduler.stop_requested();
  if (!stopped)
    scheduler.request_stop();
  scheduler.wait();
  ASSERT_TRUE(stopped);
  EXPECT_GT(state->windows_per_second, 0);
  std::cout << "PRODUCTION_F16 mode=" << mode << " consumed=" << state->consumed
            << " windows_per_second=" << state->windows_per_second << std::endl;
}
