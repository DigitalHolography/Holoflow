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
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "curaii/cuda.hh"
#include "holoflow/runtime/compiler.hh"
#include "holonp/arange.hh"
#include "holonp/fft2.hh"
#include "holonp/fftshift.hh"
#include "holonp/reshape.hh"
#include "holotask/syncs/mean_abs.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::OpResult;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

struct PipelineState {
  std::mutex mutex;
  float      value = 0.0F;
};

class ScalarSink final : public holoflow::core::ISyncTask {
public:
  ScalarSink(std::shared_ptr<PipelineState> state, cudaStream_t stream)
      : state_(std::move(state)), stream_(stream) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    float value = 0.0F;
    CUDA_CHECK(cudaMemcpyAsync(&value, ctx.inputs[0].data(), sizeof(value), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    {
      std::lock_guard lock(state_->mutex);
      state_->value = value;
    }
    return OpResult::Eof;
  }

private:
  std::shared_ptr<PipelineState> state_;
  cudaStream_t                   stream_;
};

class ScalarSinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit ScalarSinkFactory(std::shared_ptr<PipelineState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1 || inputs[0].mem_loc != MemLoc::Device ||
        inputs[0].dtype != DType::F32 || inputs[0].num_elements() != 1) {
      throw std::invalid_argument("scalar sink requires one device F32 scalar");
    }
    return InferResult{.input_descs   = {inputs[0]},
                       .output_descs  = {},
                       .in_place      = {},
                       .owned_inputs  = {false},
                       .owned_outputs = {},
                       .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override {
    (void)infer(inputs, settings);
    return std::make_unique<ScalarSink>(state_, ctx.stream);
  }

private:
  std::shared_ptr<PipelineState> state_;
};

GraphSpec production_graph() {
  GraphSpec graph;
  const auto arange = add_vertex(
      NodeSpec{"arange", "Arange",
               nlohmann::json(holonp::ArangeSettings{
                   .start = 1.0,
                   .stop = 5.0,
                   .step = 1.0,
                   .dtype = DType::CF32,
                   .device = MemLoc::Device,
               })},
      graph);
  const auto reshape = add_vertex(
      NodeSpec{"reshape", "Reshape", nlohmann::json(holonp::ReshapeSettings{
                                             .shape = {2, 2}, .copy = std::nullopt})},
      graph);
  const auto fft2 = add_vertex(NodeSpec{"fft2", "FFT2", nlohmann::json(holonp::FFT2Settings{})},
                               graph);
  const auto shift = add_vertex(
      NodeSpec{"shift", "FFTShiftNp", nlohmann::json(holonp::FFTShiftSettings{})}, graph);
  const auto mean_abs = add_vertex(
      NodeSpec{"mean-abs", "MeanAbs",
               nlohmann::json(holotask::syncs::MeanAbsSettings{.axis = {}, .keepdims = true})},
      graph);
  const auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, graph);

  add_edge(arange, reshape, EdgeSpec{0, 0}, graph);
  add_edge(reshape, fft2, EdgeSpec{0, 0}, graph);
  add_edge(fft2, shift, EdgeSpec{0, 0}, graph);
  add_edge(shift, mean_abs, EdgeSpec{0, 0}, graph);
  add_edge(mean_abs, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

} // namespace

TEST(ProductionPipelineTest, ExecutesArangeReshapeFftShiftAndMeanAbsEndToEnd) {
  auto state = std::make_shared<PipelineState>();
  holoflow::core::Registry registry;
  registry.register_sync("Arange", std::make_unique<holonp::ArangeFactory>());
  registry.register_sync("Reshape", std::make_unique<holonp::ReshapeFactory>());
  registry.register_sync("FFT2", std::make_unique<holonp::FFT2Factory>());
  registry.register_sync("FFTShiftNp", std::make_unique<holonp::FFTShiftFactory>());
  registry.register_sync("MeanAbs", std::make_unique<holotask::syncs::MeanAbsFactory>());
  registry.register_sync("sink", std::make_unique<ScalarSinkFactory>(state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(production_graph());
  ASSERT_EQ(output->resources.tasks.size(), 6U);

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  // The unnormalised 2x2 FFT of [[1, 2], [3, 4]] has magnitudes {10, 2, 4, 0};
  // FFTShift only reorders them, so the mean absolute value is exactly 4.
  std::lock_guard lock(state->mutex);
  EXPECT_NEAR(state->value, 4.0F, 1e-5F);
}
