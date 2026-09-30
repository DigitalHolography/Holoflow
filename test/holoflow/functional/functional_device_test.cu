// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <boost/graph/adjacency_list.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::InferResult;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

__global__ void add_kernel(const float *lhs, const float *rhs, float *out, size_t size) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = lhs[i] + rhs[i];
  }
}

__global__ void scale_kernel(const float *input, float factor, float *out, size_t size) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = input[i] * factor;
  }
}

float *device_data(holoflow::core::TView &view) { return reinterpret_cast<float *>(view.data()); }

class DeviceSourceTask final : public holoflow::core::ISyncTask {
public:
  DeviceSourceTask(std::vector<float> values, cudaStream_t stream)
      : values_(std::move(values)), stream_(stream) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUDA_CHECK(cudaMemcpyAsync(device_data(ctx.outputs[0]), values_.data(),
                               values_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    return OpResult::Ok;
  }

private:
  std::vector<float> values_;
  cudaStream_t       stream_;
};

class DeviceAddTask final : public holoflow::core::ISyncTask {
public:
  explicit DeviceAddTask(cudaStream_t stream) : stream_(stream) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto    size       = ctx.outputs[0].desc.num_elements();
    constexpr int block_size = 128;
    const auto    grid_size  = static_cast<unsigned>((size + block_size - 1) / block_size);
    add_kernel<<<grid_size, block_size, 0, stream_>>>(
        device_data(ctx.inputs[0]), device_data(ctx.inputs[1]), device_data(ctx.outputs[0]), size);
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  cudaStream_t stream_;
};

class DeviceScaleTask final : public holoflow::core::ISyncTask {
public:
  DeviceScaleTask(float factor, cudaStream_t stream) : factor_(factor), stream_(stream) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto    size       = ctx.outputs[0].desc.num_elements();
    constexpr int block_size = 128;
    const auto    grid_size  = static_cast<unsigned>((size + block_size - 1) / block_size);
    scale_kernel<<<grid_size, block_size, 0, stream_>>>(device_data(ctx.inputs[0]), factor_,
                                                        device_data(ctx.outputs[0]), size);
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  float        factor_;
  cudaStream_t stream_;
};

class DeviceCollectTask final : public holoflow::core::ISyncTask {
public:
  DeviceCollectTask(std::shared_ptr<holoflow::test::MathState> state, cudaStream_t stream)
      : state_(std::move(state)), stream_(stream) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    state_->collected.resize(ctx.inputs[0].desc.num_elements());
    CUDA_CHECK(cudaMemcpyAsync(state_->collected.data(), device_data(ctx.inputs[0]),
                               state_->collected.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    return OpResult::Eof;
  }

private:
  std::shared_ptr<holoflow::test::MathState> state_;
  cudaStream_t                               stream_;
};

class DeviceAsyncBridgeTask final : public holoflow::core::IAsyncTask {
public:
  DeviceAsyncBridgeTask(size_t bytes, cudaStream_t producer_stream, cudaStream_t consumer_stream,
                        std::shared_ptr<holoflow::test::MathState> state)
      : bytes_(bytes), producer_stream_(producer_stream), consumer_stream_(consumer_stream),
        state_(std::move(state)), buffer_(curaii::make_unique_device_ptr<std::byte>(bytes)) {}

  OpResult try_push(holoflow::core::AsyncPushCtx &ctx) override {
    ++state_->async_push_calls;
    std::lock_guard lock(mutex_);
    if (full_) {
      ++state_->async_push_not_ready;
      return OpResult::NotReady;
    }
    CUDA_CHECK(cudaMemcpyAsync(buffer_.get(), ctx.inputs[0].data(), bytes_,
                               cudaMemcpyDeviceToDevice, producer_stream_));
    CUDA_CHECK(cudaStreamSynchronize(producer_stream_));
    full_ = true;
    return OpResult::Ok;
  }

  OpResult try_pop(holoflow::core::AsyncPopCtx &ctx) override {
    ++state_->async_pop_calls;
    std::lock_guard lock(mutex_);
    if (!full_) {
      ++state_->async_pop_not_ready;
      return OpResult::NotReady;
    }
    CUDA_CHECK(cudaMemcpyAsync(ctx.outputs[0].data(), buffer_.get(), bytes_,
                               cudaMemcpyDeviceToDevice, consumer_stream_));
    CUDA_CHECK(cudaStreamSynchronize(consumer_stream_));
    full_ = false;
    return OpResult::Ok;
  }

private:
  size_t                                     bytes_;
  cudaStream_t                               producer_stream_;
  cudaStream_t                               consumer_stream_;
  std::shared_ptr<holoflow::test::MathState> state_;
  curaii::unique_device_ptr<std::byte>       buffer_;
  std::mutex                                 mutex_;
  bool                                       full_ = false;
};

class DeviceAsyncBridgeFactory final : public holoflow::core::IAsyncTaskFactory {
public:
  explicit DeviceAsyncBridgeFactory(std::shared_ptr<holoflow::test::MathState> state)
      : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1 || inputs[0].mem_loc != MemLoc::Device) {
      throw std::invalid_argument("device bridge requires one device input");
    }
    return {{inputs[0]}, {inputs[0]}, {}, {false}, {false}, TaskKind::Async};
  }

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const TDesc>                inputs, const nlohmann::json &,
         const holoflow::core::AsyncCreateCtx &ctx) const override {
    state_->producer_stream = ctx.producer_stream;
    state_->consumer_stream = ctx.consumer_stream;
    return std::make_unique<DeviceAsyncBridgeTask>(inputs[0].num_bytes(), ctx.producer_stream,
                                                   ctx.consumer_stream, state_);
  }

private:
  std::shared_ptr<holoflow::test::MathState> state_;
};

enum class DeviceOperation { Source, Add, Scale, Sink };

class DeviceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  DeviceFactory(DeviceOperation operation, std::shared_ptr<holoflow::test::MathState> state,
                std::vector<float> values = {}, float factor = 1.F)
      : operation_(operation), state_(std::move(state)), values_(std::move(values)),
        factor_(factor) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    switch (operation_) {
    case DeviceOperation::Source:
      return {{},      {TDesc({values_.size()}, DType::F32, MemLoc::Device)},
              {},      {},
              {false}, TaskKind::Sync};
    case DeviceOperation::Add:
      if (inputs.size() != 2 || inputs[0].shape != inputs[1].shape) {
        throw std::invalid_argument("device add requires matching inputs");
      }
      return {
          {inputs.begin(), inputs.end()}, {inputs[0]}, {}, {false, false}, {false}, TaskKind::Sync};
    case DeviceOperation::Scale:
      if (inputs.size() != 1) {
        throw std::invalid_argument("device scale requires one input");
      }
      return {{inputs[0]}, {inputs[0]}, {}, {false}, {false}, TaskKind::Sync};
    case DeviceOperation::Sink:
      if (inputs.size() != 1) {
        throw std::invalid_argument("device sink requires one input");
      }
      return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
    }
    throw std::invalid_argument("unknown device operation");
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &ctx) const override {
    state_->last_sync_stream = ctx.stream;
    switch (operation_) {
    case DeviceOperation::Source:
      return std::make_unique<DeviceSourceTask>(values_, ctx.stream);
    case DeviceOperation::Add:
      return std::make_unique<DeviceAddTask>(ctx.stream);
    case DeviceOperation::Scale:
      return std::make_unique<DeviceScaleTask>(factor_, ctx.stream);
    case DeviceOperation::Sink:
      return std::make_unique<DeviceCollectTask>(state_, ctx.stream);
    }
    throw std::invalid_argument("unknown device operation");
  }

private:
  DeviceOperation                            operation_;
  std::shared_ptr<holoflow::test::MathState> state_;
  std::vector<float>                         values_;
  float                                      factor_;
};

holoflow::core::GraphSpec device_math_graph() {
  using holoflow::core::EdgeSpec;
  using holoflow::core::NodeSpec;
  holoflow::core::GraphSpec graph;
  auto                      lhs   = add_vertex(NodeSpec{"lhs", "lhs", {}}, graph);
  auto                      rhs   = add_vertex(NodeSpec{"rhs", "rhs", {}}, graph);
  auto                      add   = add_vertex(NodeSpec{"add", "add", {}}, graph);
  auto                      scale = add_vertex(NodeSpec{"scale", "scale", {}}, graph);
  auto                      sink  = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(lhs, add, EdgeSpec{0, 0}, graph);
  add_edge(rhs, add, EdgeSpec{0, 1}, graph);
  add_edge(add, scale, EdgeSpec{0, 0}, graph);
  add_edge(scale, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

holoflow::core::GraphSpec device_async_math_graph() {
  using holoflow::core::EdgeSpec;
  using holoflow::core::NodeSpec;
  holoflow::core::GraphSpec graph;
  auto                      source   = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto                      bridge_a = add_vertex(NodeSpec{"bridge-a", "bridge", {}}, graph);
  auto                      scale_a  = add_vertex(NodeSpec{"scale-a", "double", {}}, graph);
  auto                      bridge_b = add_vertex(NodeSpec{"bridge-b", "bridge", {}}, graph);
  auto                      scale_b  = add_vertex(NodeSpec{"scale-b", "triple", {}}, graph);
  auto                      sink     = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, bridge_a, EdgeSpec{0, 0}, graph);
  add_edge(bridge_a, scale_a, EdgeSpec{0, 0}, graph);
  add_edge(scale_a, bridge_b, EdgeSpec{0, 0}, graph);
  add_edge(bridge_b, scale_b, EdgeSpec{0, 0}, graph);
  add_edge(scale_b, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

} // namespace

TEST(FunctionalPipelineTest, CompilesAndExecutesCudaVectorMath) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("lhs", std::make_unique<DeviceFactory>(DeviceOperation::Source, state,
                                                                std::vector<float>{1, 2, 3, 4}));
  registry.register_sync("rhs",
                         std::make_unique<DeviceFactory>(DeviceOperation::Source, state,
                                                         std::vector<float>{10, 20, 30, 40}));
  registry.register_sync("add", std::make_unique<DeviceFactory>(DeviceOperation::Add, state));
  registry.register_sync("scale", std::make_unique<DeviceFactory>(DeviceOperation::Scale, state,
                                                                  std::vector<float>{}, 0.25F));
  registry.register_sync("sink", std::make_unique<DeviceFactory>(DeviceOperation::Sink, state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto                         output = compiler.compile(device_math_graph());
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  ASSERT_EQ(state->collected.size(), 4);
  EXPECT_FLOAT_EQ(state->collected[0], 2.75F);
  EXPECT_FLOAT_EQ(state->collected[1], 5.5F);
  EXPECT_FLOAT_EQ(state->collected[2], 8.25F);
  EXPECT_FLOAT_EQ(state->collected[3], 11.F);

  const auto metrics = scheduler.metrics();
  EXPECT_GT(metrics.at("add").device_throughput_bytes_per_second, 0.0);
  EXPECT_GT(metrics.at("scale").device_throughput_bytes_per_second, 0.0);
}

TEST(FunctionalPipelineTest, ExecutesCudaMathAcrossThreeAsyncSections) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<DeviceFactory>(DeviceOperation::Source, state,
                                                                   std::vector<float>{1, 2, 3, 4}));
  registry.register_async("bridge", std::make_unique<DeviceAsyncBridgeFactory>(state));
  registry.register_sync("double", std::make_unique<DeviceFactory>(DeviceOperation::Scale, state,
                                                                   std::vector<float>{}, 2.F));
  registry.register_sync("triple", std::make_unique<DeviceFactory>(DeviceOperation::Scale, state,
                                                                   std::vector<float>{}, 3.F));
  registry.register_sync("sink", std::make_unique<DeviceFactory>(DeviceOperation::Sink, state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(device_async_math_graph());
  ASSERT_EQ(output->sections.size(), 3);
  ASSERT_EQ(output->resources.streams.size(), 3);
  EXPECT_NE(output->sections[0].stream, nullptr);
  EXPECT_NE(output->sections[1].stream, nullptr);
  EXPECT_NE(output->sections[2].stream, nullptr);
  EXPECT_NE(output->sections[0].stream, output->sections[1].stream);
  EXPECT_NE(output->sections[0].stream, output->sections[2].stream);
  EXPECT_NE(output->sections[1].stream, output->sections[2].stream);

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  ASSERT_EQ(state->collected.size(), 4);
  EXPECT_FLOAT_EQ(state->collected[0], 6.F);
  EXPECT_FLOAT_EQ(state->collected[1], 12.F);
  EXPECT_FLOAT_EQ(state->collected[2], 18.F);
  EXPECT_FLOAT_EQ(state->collected[3], 24.F);
  EXPECT_GE(state->async_push_calls, 2);
  EXPECT_GE(state->async_pop_calls, 2);

  const auto metrics = scheduler.metrics();
  EXPECT_GT(metrics.at("bridge-a").device_throughput_bytes_per_second, 0.0);
  EXPECT_GT(metrics.at("bridge-b").device_throughput_bytes_per_second, 0.0);
}
