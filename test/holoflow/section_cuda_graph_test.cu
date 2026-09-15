// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include "holoflow/runtime/compiler.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

struct State {
  int                frame              = 0;
  int                frames             = 13;
  int                executions         = 0;
  int                recordings         = 0;
  int                acquisitions       = 0;
  bool               unexpected_pointer = false;
  std::vector<float> results;
};

class Boundary : public IAsyncTask {
public:
  Boundary(bool source, size_t count, TDesc desc, cudaStream_t stream, std::shared_ptr<State> state,
           bool bad)
      : source_(source), count_(count), desc_(desc), stream_(stream), state_(state), bad_(bad) {
    for (size_t i = 0; i <= count; ++i)
      buffers_.push_back(curaii::make_unique_device_ptr<float>(2));
  }
  std::optional<std::vector<std::byte *>> owned_input_pointers(size_t) const override {
    return pointers();
  }
  std::optional<std::vector<std::byte *>> owned_output_pointers(size_t) const override {
    return pointers();
  }
  std::optional<TView> acquire_input(int) override {
    ++state_->acquisitions;
    auto &storage = storage_access().owned_input_storage(0);
    storage.ptr   = reinterpret_cast<std::byte *>(buffers_[state_->frame % count_].get());
    return TView{desc_, &storage};
  }
  OpResult try_pop(AsyncPopCtx &ctx) override {
    auto        &storage = storage_access().owned_output_storage(0);
    const size_t slot =
        state_->unexpected_pointer && state_->frame == 2 ? count_ : state_->frame % count_;
    storage.ptr = reinterpret_cast<std::byte *>(buffers_[slot].get());
    value_      = static_cast<float>(state_->frame);
    CUDA_CHECK(cudaMemcpyAsync(storage.ptr + desc_.offset, &value_, sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    ctx.outputs[0] = {desc_, &storage};
    return OpResult::Ok;
  }
  OpResult try_push(AsyncPushCtx &ctx) override {
    float result;
    CUDA_CHECK(cudaMemcpyAsync(&result, ctx.inputs[0].data(), sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    state_->results.push_back(result);
    storage_access().owned_input_storage(0).ptr = nullptr;
    return state_->results.size() == static_cast<size_t>(state_->frames) ? OpResult::Eof
                                                                         : OpResult::Ok;
  }
  void release_output(int) override {
    storage_access().owned_output_storage(0).ptr = nullptr;
    ++state_->frame;
  }

private:
  std::vector<std::byte *> pointers() const {
    std::vector<std::byte *> result;
    for (size_t i = 0; i < count_; ++i)
      result.push_back(reinterpret_cast<std::byte *>(buffers_[i].get()));
    if (bad_)
      result[0] = nullptr;
    return result;
  }
  bool                                          source_;
  size_t                                        count_;
  TDesc                                         desc_;
  cudaStream_t                                  stream_;
  std::shared_ptr<State>                        state_;
  bool                                          bad_;
  float                                         value_ = 0;
  std::vector<curaii::unique_device_ptr<float>> buffers_;
};

class BoundaryFactory : public IAsyncTaskFactory {
public:
  BoundaryFactory(bool source, std::shared_ptr<State> state) : source_(source), state_(state) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    const auto  count    = settings.value("count", size_t{2});
    const auto  declared = settings.value("declared", count);
    const TDesc desc     = source_ ? TDesc({1}, DType::F32, MemLoc::Device, size_t{4}) : inputs[0];
    InferResult result;
    result.kind = TaskKind::Async;
    if (source_) {
      result.output_descs  = {desc};
      result.owned_outputs = {true};
      if (!settings.value("unknown", false))
        result.owned_output_pointer_counts = {declared};
    } else {
      result.input_descs                = {desc};
      result.owned_inputs               = {true};
      result.owned_input_pointer_counts = {declared};
    }
    return result;
  }
  std::unique_ptr<IAsyncTask> create(std::span<const TDesc> inputs, const nlohmann::json &settings,
                                     const AsyncCreateCtx &ctx) const override {
    const auto inferred = infer(inputs, settings);
    return std::make_unique<Boundary>(
        source_, settings.value("count", size_t{2}), source_ ? inferred.output_descs[0] : inputs[0],
        source_ ? ctx.consumer_stream : ctx.producer_stream, state_, settings.value("bad", false));
  }

private:
  bool                   source_;
  std::shared_ptr<State> state_;
};

__global__ void affine(const float *input, float *output, float scale, float bias) {
  *output = *input * scale + bias;
}

__global__ void condition(cudaGraphConditionalHandle handle, const float *input) {
  cudaGraphSetConditional(handle, *input >= 4 ? 1U : 0U);
}

class Compute : public ISyncTask {
public:
  Compute(cudaStream_t stream, std::shared_ptr<State> state, nlohmann::json settings)
      : stream_(stream), state_(state), settings_(settings) {}
  bool supports_cuda_graph() const noexcept override {
    return !settings_.value("unsupported", false);
  }
  OpResult execute(SyncCtx &ctx) override {
    ++state_->executions;
    enqueue(ctx.inputs, ctx.outputs, stream_);
    return OpResult::Ok;
  }
  void record_cuda_graph(CudaGraphCtx &ctx) override {
    ++state_->recordings;
    if (settings_.value("invalidate", false)) {
      CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    if (settings_.value("fail_after", 0) &&
        state_->recordings >= settings_["fail_after"].get<int>())
      throw std::runtime_error("deliberate recording failure");
    if (settings_.value("child", false)) {
      cudaGraph_t child;
      CUDA_CHECK(cudaGraphCreate(&child, 0));
      cudaGraphNode_t                node;
      std::vector<cudaGraphNode_t>   deps;
      std::vector<cudaGraphEdgeData> edges;
      ctx.dependencies(deps, edges);
      const auto result =
          cudaGraphAddChildGraphNode(&node, ctx.graph, deps.data(), deps.size(), child);
      CUDA_CHECK_NT(cudaGraphDestroy(child));
      CUDA_CHECK(result);
      ctx.set_dependencies(std::span{&node, size_t{1}});
    } else if (settings_.value("conditional", false)) {
      cudaGraphConditionalHandle handle;
      CUDA_CHECK(
          cudaGraphConditionalHandleCreate(&handle, ctx.graph, 0, cudaGraphCondAssignDefault));
      auto *input = reinterpret_cast<float *>(ctx.inputs[0].data());
      condition<<<1, 1, 0, ctx.stream>>>(handle, input);
      std::vector<cudaGraphNode_t>   deps;
      std::vector<cudaGraphEdgeData> edges;
      ctx.dependencies(deps, edges);
      cudaGraphNodeParams params{};
      params.type               = cudaGraphNodeTypeConditional;
      params.conditional.handle = handle;
      params.conditional.type   = cudaGraphCondTypeIf;
      params.conditional.size   = 1;
      cudaGraphNode_t node;
      CUDA_CHECK(
          cudaGraphAddNode(&node, ctx.graph, deps.data(), edges.data(), deps.size(), &params));
      curaii::CudaStream body_stream;
      CUDA_CHECK(cudaStreamBeginCaptureToGraph(body_stream.get(), params.conditional.phGraph_out[0],
                                               nullptr, nullptr, 0,
                                               cudaStreamCaptureModeThreadLocal));
      affine<<<1, 1, 0, body_stream.get()>>>(input, input, 1, 10);
      cudaGraph_t body;
      CUDA_CHECK(cudaStreamEndCapture(body_stream.get(), &body));
      ctx.set_dependencies(std::span{&node, size_t{1}});
    } else {
      enqueue(ctx.inputs, ctx.outputs, ctx.stream);
    }
  }

private:
  void enqueue(std::span<TView> inputs, std::span<TView> outputs, cudaStream_t stream) {
    affine<<<1, 1, 0, stream>>>(reinterpret_cast<float *>(inputs[0].data()),
                                reinterpret_cast<float *>(outputs[0].data()),
                                settings_.value("scale", 1.F), settings_.value("bias", 0.F));
    CUDA_CHECK(cudaGetLastError());
  }
  cudaStream_t           stream_;
  std::shared_ptr<State> state_;
  nlohmann::json         settings_;
};

class ComputeFactory : public ISyncTaskFactory {
public:
  explicit ComputeFactory(std::shared_ptr<State> state) : state_(state) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    auto output = inputs[0];
    if (!settings.value("alias", false))
      output.offset = 0;
    return {{inputs[0]},
            {output},
            settings.value("alias", false) ? std::vector<InPlace>{{0, 0}} : std::vector<InPlace>{},
            {false},
            {false},
            TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &settings,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<Compute>(ctx.stream, state_, settings);
  }

private:
  std::shared_ptr<State> state_;
};

class SectionCudaGraphTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.register_async("source", std::make_unique<BoundaryFactory>(true, state));
    registry.register_async("sink", std::make_unique<BoundaryFactory>(false, state));
    registry.register_sync("compute", std::make_unique<ComputeFactory>(state));
    source = add_vertex(NodeSpec{"source", "source", {{"count", 2}}}, spec);
    first  = add_vertex(NodeSpec{"first", "compute", {{"scale", 2.F}}}, spec);
    last   = add_vertex(NodeSpec{"last", "compute", {{"bias", 1.F}}}, spec);
    sink   = add_vertex(NodeSpec{"sink", "sink", {{"count", 3}}}, spec);
    add_edge(source, first, EdgeSpec{0, 0}, spec);
    add_edge(first, last, EdgeSpec{0, 0}, spec);
    add_edge(last, sink, EdgeSpec{0, 0}, spec);
  }
  std::unique_ptr<CompilerOutput> compile(size_t                          limit = 128,
                                          std::unique_ptr<CompilerOutput> prev  = {}) {
    Compiler::Config config;
    config.max_section_cuda_graphs = limit;
    config.verbose_tracing         = false;
    config.enable_profiling        = false;
    config.dump_dot_on_failure     = false;
    return Compiler(registry, config).compile(spec, std::move(prev));
  }
  void run(CompilerOutput &out) {
    Scheduler scheduler(out.graph, out.sections, out.resources);
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    scheduler.request_stop();
    scheduler.wait();
    EXPECT_EQ(state->results.size(), static_cast<size_t>(state->frames));
    metrics         = scheduler.metrics();
    section_metrics = scheduler.section_graph_metrics();
  }
  std::shared_ptr<State>             state = std::make_shared<State>();
  Registry                           registry;
  GraphSpec                          spec;
  GraphSpec::vertex_descriptor       source, first, last, sink;
  std::map<std::string, NodeMetrics> metrics, section_metrics;
};

TEST_F(SectionCudaGraphTest, EagerProductReplaysRotatingPointersAndOffsets) {
  auto out = compile(6);
  ASSERT_EQ(out->sections.size(), 1U);
  const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 6U);
  EXPECT_EQ(state->recordings, 12);
  EXPECT_EQ(state->frame, 0);
  EXPECT_EQ(state->acquisitions, 0);
  EXPECT_EQ(state->executions, 0);
  run(*out);
  EXPECT_EQ(state->executions, 0);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
  EXPECT_FALSE(metrics.at("first").individual_timing_available);
  EXPECT_NE(fmt::format("{}", metrics.at("first")).find("n/a (section graph)"), std::string::npos);
  EXPECT_EQ(metrics.at("first").sample_count, 13U);
  EXPECT_EQ(section_metrics.begin()->second.sample_count, 13U);
}

TEST_F(SectionCudaGraphTest, CapAndDisabledUseOrdinaryExecution) {
  for (size_t limit : {size_t{0}, size_t{5}}) {
    auto out = compile(limit);
    EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
    EXPECT_EQ(state->recordings, 0);
    state->frame = 0;
    state->results.clear();
    run(*out);
    for (size_t i = 0; i < state->results.size(); ++i)
      EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
  }
}

TEST_F(SectionCudaGraphTest, UnknownDomainAndIncompatibleTaskFallBack) {
  spec[source].settings["unknown"] = true;
  auto out                         = compile();
  EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
  EXPECT_EQ(state->recordings, 0);
  spec[source].settings.erase("unknown");
  spec[first].settings["unsupported"] = true;
  out                                 = compile();
  EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
  run(*out);
  EXPECT_GT(state->executions, 0);
}

TEST_F(SectionCudaGraphTest, ProductOverflowIsRejectedBeforeRecordingOrEnumeration) {
  spec[source].settings["declared"] = (std::numeric_limits<size_t>::max)();
  auto out                          = compile((std::numeric_limits<size_t>::max)());
  EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
  EXPECT_EQ(state->recordings, 0);
}

TEST_F(SectionCudaGraphTest, MalformedDomainsFailCompilation) {
  spec[source].settings["bad"] = true;
  EXPECT_THROW(compile(), std::invalid_argument);
  spec[source].settings.erase("bad");
  spec[source].settings["declared"] = 3;
  EXPECT_THROW(compile(), std::invalid_argument);
}

TEST_F(SectionCudaGraphTest, RecordingFailureDiscardsPartialSetWithoutExecuting) {
  spec[last].settings["fail_after"] = 4;
  auto        out                   = compile();
  const auto &graphs                = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.enabled);
  EXPECT_TRUE(graphs.executables.empty());
  EXPECT_EQ(state->frame, 0);
  EXPECT_EQ(state->executions, 0);
  run(*out);
}

TEST_F(SectionCudaGraphTest, EmbeddedChildGraphFallsBack) {
  spec[first].settings["child"] = true;
  auto        out               = compile();
  const auto &graphs            = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.enabled);
  EXPECT_NE(graphs.fallback_reason.find("child graph"), std::string::npos);
  run(*out);
}

TEST_F(SectionCudaGraphTest, InvalidatedCaptureRestoresStreamForFallback) {
  spec[first].settings["invalidate"] = true;
  auto out                           = compile();
  EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
  cudaStreamCaptureStatus status;
  CUDA_CHECK(cudaStreamIsCapturing(out->sections[0].stream, &status));
  EXPECT_EQ(status, cudaStreamCaptureStatusNone);
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, ConditionalNodeComposesWithSurroundingTasks) {
  remove_edge(first, last, spec);
  auto conditional = add_vertex(
      NodeSpec{"conditional", "compute", {{"conditional", true}, {"alias", true}}}, spec);
  add_edge(first, conditional, EdgeSpec{0, 0}, spec);
  add_edge(conditional, last, EdgeSpec{0, 0}, spec);
  auto        out    = compile();
  const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 6U);
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + (i >= 2 ? 10 : 0) + 1);
}

TEST_F(SectionCudaGraphTest, UnexpectedPointerFallsBackBeforeSubmitting) {
  auto out                  = compile();
  state->unexpected_pointer = true;
  run(*out);
  EXPECT_FALSE(out->resources.section_cuda_graphs.begin()->second->enabled);
  EXPECT_EQ(state->executions, 22);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, RecompilationChangesParametersAndRebuildsGraphs) {
  auto out                      = compile();
  spec[first].settings["scale"] = 3.F;
  out                           = compile(128, std::move(out));
  ASSERT_TRUE(out->resources.section_cuda_graphs.begin()->second->enabled);
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 3.F * i + 1);
}

TEST_F(SectionCudaGraphTest, AliasedRotatingStorageIsOnlyOneProductDimension) {
  spec[first].settings["alias"] = true;
  auto        out               = compile(6);
  const auto &graphs            = *out->resources.section_cuda_graphs.begin()->second;
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.storage_ids.size(), 2U);
  EXPECT_EQ(graphs.executables.size(), 6U);
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, SingletonDomainsAndResumeRetainExecutableSet) {
  spec[source].settings["count"] = 1;
  spec[sink].settings["count"]   = 1;
  auto        out                = compile(1);
  const auto &graphs             = *out->resources.section_cuda_graphs.begin()->second;
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 1U);
  run(*out);
  state->frames = 26;
  run(*out);
  EXPECT_EQ(state->recordings, 2);
  EXPECT_EQ(state->executions, 0);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

} // namespace
