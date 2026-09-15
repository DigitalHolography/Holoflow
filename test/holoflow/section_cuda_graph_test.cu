// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include "holoflow/runtime/compiler.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

struct State {
  int                   frame        = 0;
  int                   frames       = 13;
  int                   executions   = 0;
  int                   recordings   = 0;
  int                   acquisitions = 0;
  int                   source_uses = 0, sink_uses = 0;
  bool                  stop_after_pop             = false;
  bool                  recorded_after_acquisition = false;
  bool                  unexpected_pointer         = false;
  bool                  unexpected_tuple           = false;
  std::vector<float>    results;
  std::function<void()> sequence_query_hook;
};

class Boundary : public IAsyncTask {
public:
  Boundary(bool source, size_t count, TDesc desc, cudaStream_t stream, std::shared_ptr<State> state,
           bool bad, nlohmann::json settings)
      : source_(source), count_(count), desc_(desc), stream_(stream), state_(state), bad_(bad),
        settings_(settings) {
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
    storage.ptr   = reinterpret_cast<std::byte *>(buffers_[slot(state_->sink_uses)].get());
    return TView{desc_, &storage};
  }
  OpResult try_pop(AsyncPopCtx &ctx) override {
    auto        &storage = storage_access().owned_output_storage(0);
    const size_t slot = state_->unexpected_pointer && state_->frame == 2
                            ? count_
                            : this->slot(state_->source_uses +
                                         (state_->unexpected_tuple && state_->frame == 2 ? 1 : 0));
    storage.ptr       = reinterpret_cast<std::byte *>(buffers_[slot].get());
    value_            = static_cast<float>(state_->frame);
    CUDA_CHECK(cudaMemcpyAsync(storage.ptr + desc_.offset, &value_, sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    ctx.outputs[0] = {desc_, &storage};
    if (state_->stop_after_pop) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      state_->stop_after_pop = false;
      ctx.cancelled->store(true);
    }
    return OpResult::Ok;
  }
  OpResult try_push(AsyncPushCtx &ctx) override {
    float result;
    CUDA_CHECK(cudaMemcpyAsync(&result, ctx.inputs[0].data(), sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    state_->results.push_back(result);
    ++state_->sink_uses;
    storage_access().owned_input_storage(0).ptr = nullptr;
    return state_->results.size() == static_cast<size_t>(state_->frames) ? OpResult::Eof
                                                                         : OpResult::Ok;
  }
  void release_output(int) override {
    storage_access().owned_output_storage(0).ptr = nullptr;
    ++state_->frame;
    ++state_->source_uses;
  }

  std::optional<PointerSequence> owned_input_pointer_sequence(size_t) const override {
    return sequence();
  }
  std::optional<PointerSequence> owned_output_pointer_sequence(size_t) const override {
    return sequence();
  }

private:
  PointerSequence base_sequence() const {
    PointerSequence result;
    if (settings_.contains("prefix"))
      result.prefix = settings_["prefix"].get<std::vector<size_t>>();
    if (settings_.contains("cycle"))
      result.cycle = settings_["cycle"].get<std::vector<size_t>>();
    else
      for (size_t i = 0; i < count_; ++i)
        result.cycle.push_back((settings_.value("phase", size_t{0}) + i) % count_);
    return result;
  }
  size_t slot(size_t use) const {
    const auto s = base_sequence();
    return use < s.prefix.size() ? s.prefix[use]
                                 : s.cycle[(use - s.prefix.size()) % s.cycle.size()];
  }
  std::optional<PointerSequence> sequence() const {
    if (state_->sequence_query_hook)
      state_->sequence_query_hook();
    if (!settings_.value("ordered", false))
      return std::nullopt;
    auto         result = base_sequence();
    const size_t used   = source_ ? state_->source_uses : state_->sink_uses;
    if (used < result.prefix.size())
      result.prefix.erase(result.prefix.begin(), result.prefix.begin() + used);
    else {
      const size_t phase =
          result.cycle.empty() ? 0 : (used - result.prefix.size()) % result.cycle.size();
      result.prefix.clear();
      std::rotate(result.cycle.begin(), result.cycle.begin() + phase, result.cycle.end());
    }
    return result;
  }
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
  nlohmann::json                                settings_;
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
    return std::make_unique<Boundary>(source_, settings.value("count", size_t{2}),
                                      source_ ? inferred.output_descs[0] : inputs[0],
                                      source_ ? ctx.consumer_stream : ctx.producer_stream, state_,
                                      settings.value("bad", false), settings);
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
    if (settings_.value("unused_handle", false)) {
      // CUDA rejects an unused conditional handle during instantiation, after capture succeeds.
      cudaGraphConditionalHandle handle;
      CUDA_CHECK(
          cudaGraphConditionalHandleCreate(&handle, ctx.graph, 0, cudaGraphCondAssignDefault));
    }
    state_->recorded_after_acquisition |= state_->acquisitions != 0 && state_->source_uses == 0;
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
    config.log_dir                 = log_directory;
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
    diagnostics     = scheduler.section_graph_diagnostics();
  }
  std::shared_ptr<State>             state = std::make_shared<State>();
  Registry                           registry;
  GraphSpec                          spec;
  GraphSpec::vertex_descriptor       source, first, last, sink;
  std::map<std::string, NodeMetrics> metrics, section_metrics;
  nlohmann::json                     diagnostics;
  std::filesystem::path              log_directory;
};

TEST_F(SectionCudaGraphTest, EagerProductReplaysRotatingPointersAndOffsets) {
  auto out = compile(6);
  ASSERT_EQ(out->sections.size(), 1U);
  const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_EQ(state->recordings, 0);
  EXPECT_TRUE(graphs.executables.empty());
  run(*out);
  EXPECT_FALSE(state->recorded_after_acquisition);
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 6U);
  EXPECT_EQ(state->recordings, 12);
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
  EXPECT_FALSE(graphs.enabled);
  EXPECT_TRUE(graphs.executables.empty());
}

TEST_F(SectionCudaGraphTest, EmbeddedChildGraphFallsBack) {
  spec[first].settings["child"] = true;
  auto        out               = compile();
  const auto &graphs            = *out->resources.section_cuda_graphs.begin()->second;
  run(*out);
  EXPECT_FALSE(graphs.enabled);
  EXPECT_NE(graphs.fallback_reason.find("child graph"), std::string::npos);
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
  run(*out);
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 6U);
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
  auto out = compile();
  run(*out); // Populate the old executable cache before replacing task resources.
  ASSERT_FALSE(out->resources.section_cuda_graphs.begin()->second->executables.empty());
  state->frame = state->source_uses = state->sink_uses = 0;
  state->results.clear();
  spec[first].settings["scale"] = 3.F;
  out                           = compile(128, std::move(out));
  run(*out);
  ASSERT_TRUE(out->resources.section_cuda_graphs.begin()->second->enabled);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 3.F * i + 1);
}

TEST_F(SectionCudaGraphTest, AliasedRotatingStorageIsOnlyOneProductDimension) {
  spec[first].settings["alias"] = true;
  auto        out               = compile(6);
  const auto &graphs            = *out->resources.section_cuda_graphs.begin()->second;
  run(*out);
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.storage_ids.size(), 2U);
  EXPECT_EQ(graphs.executables.size(), 6U);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, SingletonDomainsAndResumeRetainExecutableSet) {
  spec[source].settings["count"] = 1;
  spec[sink].settings["count"]   = 1;
  auto        out                = compile(1);
  const auto &graphs             = *out->resources.section_cuda_graphs.begin()->second;
  run(*out);
  ASSERT_TRUE(graphs.enabled) << graphs.fallback_reason;
  EXPECT_EQ(graphs.executables.size(), 1U);
  state->frames = 26;
  run(*out);
  EXPECT_EQ(state->recordings, 2);
  EXPECT_EQ(state->executions, 0);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, ExactCyclesPruneBeforeApplyingCapAndReuseOnResume) {
  spec[source].settings = {{"count", 4}, {"ordered", true}, {"phase", 1}};
  spec[sink].settings   = {{"count", 6}, {"ordered", true}, {"phase", 3}};
  auto  out             = compile(12);
  auto &graphs          = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_EQ(graphs.snapshot()["raw_cartesian_count"], 24);
  EXPECT_EQ(graphs.snapshot()["pruned_count"], 12);
  run(*out);
  ASSERT_TRUE(graphs.enabled);
  EXPECT_EQ(graphs.executables.size(), 12U);
  const auto handles = graphs.executables;
  state->frames      = 26;
  run(*out);
  EXPECT_EQ(graphs.snapshot()["reused"], 12);
  EXPECT_EQ(graphs.snapshot()["created"], 0);
  EXPECT_EQ(graphs.snapshot()["refresh_count"], 2);
  EXPECT_EQ(state->executions, 0);
  EXPECT_EQ(graphs.snapshot()["tuple_misses"], 0);
  for (auto handle : handles)
    EXPECT_NE(std::find(graphs.executables.begin(), graphs.executables.end(), handle),
              graphs.executables.end());
}

TEST_F(SectionCudaGraphTest, StartupPrefixesAndRepeatedCycleIndicesAreDeduplicated) {
  spec[source].settings = {
      {"count", 3}, {"ordered", true}, {"prefix", {2, 2, 0}}, {"cycle", {0, 0, 1}}};
  spec[sink].settings = {{"count", 2}, {"ordered", true}, {"prefix", {1}}, {"cycle", {1, 0}}};
  auto out            = compile(6);
  run(*out);
  EXPECT_TRUE(out->resources.section_cuda_graphs.begin()->second->enabled);
  EXPECT_EQ(state->executions, 0);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
  state->frames = 26;
  run(*out);
  EXPECT_EQ(diagnostics[0]["pruned_count"], 4);
  EXPECT_EQ(diagnostics[0]["discarded"], 1);
  EXPECT_EQ(diagnostics[0]["reused"], 4);
  EXPECT_EQ(diagnostics[0]["created"], 0);
}

TEST_F(SectionCudaGraphTest, UnspecifiedDimensionsRemainCartesian) {
  spec[source].settings        = {{"count", 4}, {"ordered", true}};
  spec[sink].settings["count"] = 6;
  auto out                     = compile(24);
  EXPECT_EQ(out->resources.section_cuda_graphs.begin()->second->snapshot()["pruned_count"], 24);
  run(*out);
  EXPECT_EQ(state->executions, 0);
}

TEST_F(SectionCudaGraphTest, InvalidSequencesFailWithDiagnosticReport) {
  spec[source].settings = {{"count", 2}, {"ordered", true}, {"cycle", nlohmann::json::array()}};
  EXPECT_THROW(compile(), std::invalid_argument);
  spec[source].settings["cycle"] = {0, 2};
  EXPECT_THROW(compile(), std::invalid_argument);
}

TEST_F(SectionCudaGraphTest, DiagnosticsIncludeAllDomainsAndBlockersOnFailure) {
  spec[first].settings["unsupported"] = true;
  spec[last].settings["unsupported"]  = true;
  spec[source].settings["unknown"]    = true;
  auto       out                      = compile(1);
  const auto report = out->resources.section_cuda_graphs.begin()->second->snapshot();
  EXPECT_EQ(report["tasks"].size(), 2U);
  EXPECT_EQ(report["domains"].size(), 3U);
  EXPECT_EQ(report["blockers"].size(), 3U);
  EXPECT_EQ(report["raw_count_status"], "unknown");
  EXPECT_EQ(report["domains"][2]["owner"], "sink");
  EXPECT_EQ(report["domains"][2]["declared_count"], 3);
  EXPECT_EQ(report["domains"][2]["enumeration"], "skipped");
}

TEST_F(SectionCudaGraphTest, PlanningBudgetUsesConservativeCartesianFallback) {
  // Coprime periods exceed the planning budget, although the pointer domains are tiny.
  std::vector<size_t> a(1009, 0), b(1013, 0);
  a.back()              = 1;
  b.back()              = 1;
  spec[source].settings = {{"count", 2}, {"ordered", true}, {"cycle", a}};
  spec[sink].settings   = {{"count", 2}, {"ordered", true}, {"cycle", b}};
  auto       out        = compile(4);
  const auto report     = out->resources.section_cuda_graphs.begin()->second->snapshot();
  EXPECT_EQ(report["planning_mode"], "cartesian");
  EXPECT_EQ(report["pruned_count"], 4);
  EXPECT_TRUE(report.contains("planning_note"));
  run(*out);
  EXPECT_EQ(state->executions, 0);
}

TEST_F(SectionCudaGraphTest, ResumeRefreshesPhaseAfterCooperativeStopBetweenPopAndPush) {
  spec[source].settings = {{"count", 4}, {"ordered", true}};
  spec[sink].settings   = {{"count", 4}, {"ordered", true}};
  auto out              = compile(4);
  state->stop_after_pop = true;
  {
    Scheduler scheduler(out->graph, out->sections, out->resources);
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    scheduler.request_stop();
    scheduler.wait();
  }
  ASSERT_EQ(state->source_uses, 1);
  ASSERT_EQ(state->sink_uses, 0);
  ASSERT_TRUE(state->results.empty());
  run(*out);
  const auto report = out->resources.section_cuda_graphs.begin()->second->snapshot();
  EXPECT_EQ(report["created"], 4);
  EXPECT_EQ(report["discarded"], 4);
  EXPECT_EQ(report["reused"], 0);
  EXPECT_EQ(report["tuple_misses"], 0);
  EXPECT_EQ(report["ordinary_iterations"], 0);
  EXPECT_EQ(state->executions, 0);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * (i + 1) + 1);
}

TEST_F(SectionCudaGraphTest, TupleMissIsDistinctFromPointerMissAndFallsBackBeforeLaunch) {
  spec[source].settings   = {{"count", 4}, {"ordered", true}};
  spec[sink].settings     = {{"count", 4}, {"ordered", true}};
  auto out                = compile(4);
  state->unexpected_tuple = true;
  run(*out);
  EXPECT_EQ(diagnostics[0]["tuple_misses"], 1);
  EXPECT_EQ(diagnostics[0]["pointer_misses"], 0);
  EXPECT_EQ(diagnostics[0]["launches"], 2);
  EXPECT_EQ(diagnostics[0]["ordinary_iterations"], 11);
  EXPECT_EQ(state->executions, 22);
  EXPECT_EQ(diagnostics[0]["failure_stage"], "lookup");
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, JsonReportSurvivesInvalidEnumerationAndRecordingFailure) {
  log_directory = std::filesystem::temp_directory_path() /
                  ("holoflow-section-diagnostics-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  spec[source].settings["bad"] = true;
  EXPECT_THROW(compile(), std::invalid_argument);
  auto read_report = [&]() {
    std::ifstream  file(log_directory / "section_cuda_graphs.json");
    nlohmann::json report;
    file >> report;
    return report;
  };
  auto report = read_report();
  EXPECT_EQ(report[0]["status"], "error");
  EXPECT_EQ(report[0]["domains"].size(), 3U);
  EXPECT_EQ(report[0]["domains"][0]["enumerated_count"], 2);
  EXPECT_EQ(report[0]["domains"][2]["enumerated_count"], 3);
  spec[source].settings.erase("bad");
  spec[last].settings["fail_after"] = 4;
  auto out                          = compile();
  EXPECT_EQ(read_report()[0]["status"], "planned");
  run(*out);
  report = read_report();
  EXPECT_EQ(report[0]["status"], "fallback");
  EXPECT_EQ(report[0]["failure_stage"], "record");
  EXPECT_EQ(report[0]["recording_task"], "last");
  EXPECT_EQ(report[0]["variants"], 0);
  EXPECT_EQ(report[0]["created"], 1);
  EXPECT_EQ(report[0]["discarded"], 1);
  EXPECT_EQ(report[0]["ordinary_iterations"], 13);
}

TEST_F(SectionCudaGraphTest, InstantiationFailureRetainsDomainsAndRestoresOrdinaryExecution) {
  spec[first].settings["unused_handle"] = true;
  auto out                              = compile();
  run(*out);
  EXPECT_EQ(diagnostics[0]["status"], "fallback");
  EXPECT_EQ(diagnostics[0]["failure_stage"], "instantiate");
  EXPECT_EQ(diagnostics[0]["domains"].size(), 3U);
  EXPECT_EQ(diagnostics[0]["variants"], 0);
  EXPECT_EQ(diagnostics[0]["launches"], 0);
  EXPECT_EQ(diagnostics[0]["ordinary_iterations"], 13);
  EXPECT_EQ(state->executions, 26);
}

TEST_F(SectionCudaGraphTest, StopRequestedDuringPreparationIsNotLost) {
  auto out = compile();
  {
    Scheduler scheduler(out->graph, out->sections, out->resources);
    state->sequence_query_hook = [&] { scheduler.request_stop(); };
    scheduler.start();
    EXPECT_TRUE(scheduler.stop_requested());
    scheduler.wait();
    state->sequence_query_hook = {};
  }
  EXPECT_EQ(state->frame, 0);
  EXPECT_EQ(state->acquisitions, 0);
  run(*out);
  EXPECT_EQ(diagnostics[0]["reused"], 6);
  EXPECT_EQ(state->executions, 0);
}

} // namespace
