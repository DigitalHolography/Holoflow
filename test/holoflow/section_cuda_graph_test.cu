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

#include "../../src/holoflow/src/runtime/diagnostics_file_writer.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/tracing.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

struct State {
  bool                              cancel_sync_once = false;
  int                               frame            = 0;
  int                               frames           = 13;
  int                               executions       = 0;
  int                               recordings       = 0;
  int                               acquisitions     = 0;
  int                               source_uses = 0, sink_uses = 0;
  bool                              stop_after_pop             = false;
  bool                              recorded_after_acquisition = false;
  bool                              unexpected_pointer         = false;
  bool                              unexpected_tuple           = false;
  std::vector<float>                results;
  std::function<void()>             sequence_query_hook;
  std::function<void()>             recording_hook;
  std::function<void(const char *)> operation_hook;
  bool                              blocked_pop      = false;
  bool                              blocked_push     = false;
  bool                              blocked_acquire  = false;
  bool                              explicit_updates = true;
  const void                       *watched_task     = nullptr;
  std::function<void()>             destruction_probe;
};

class Boundary : public IAsyncTask {
public:
  Boundary(bool source, size_t count, TDesc desc, cudaStream_t stream, std::shared_ptr<State> state,
           bool bad, nlohmann::json settings)
      : source_(source), count_(count), desc_(desc), stream_(stream), state_(state), bad_(bad),
        settings_(settings) {
    for (size_t i = 0; i < (count + 1) * settings_.value("ports", 1); ++i)
      buffers_.push_back(curaii::make_unique_device_ptr<float>(2));
  }
  std::optional<std::vector<std::byte *>> owned_input_pointers(size_t index) const override {
    return pointers(index);
  }
  std::optional<std::vector<std::byte *>> owned_output_pointers(size_t index) const override {
    return pointers(index);
  }
  std::optional<TView> acquire_input(int index) override {
    if (state_->blocked_acquire) {
      if (state_->operation_hook)
        state_->operation_hook("blocked acquire");
      return std::nullopt;
    }
    ++state_->acquisitions;
    auto &storage = storage_access().owned_input_storage(index);
    storage.ptr   = reinterpret_cast<std::byte *>(
        buffers_[index * (count_ + 1) + slot(state_->sink_uses)].get());
    if (state_->operation_hook)
      state_->operation_hook(settings_.value("ports", 1) == 2 && index == 0 ? "acquire first"
                                                                            : "acquire");
    return TView{desc_, &storage};
  }
  OpResult try_pop(AsyncPopCtx &ctx) override {
    if (state_->blocked_pop) {
      if (state_->operation_hook)
        state_->operation_hook("blocked pop");
      return OpResult::NotReady;
    }
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
    if (settings_.value("ports", 1) == 2) {
      auto &second = storage_access().owned_output_storage(1);
      second.ptr   = reinterpret_cast<std::byte *>(buffers_[count_ + 1 + slot].get());
      CUDA_CHECK(cudaMemcpyAsync(second.ptr + desc_.offset, &value_, sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
      ctx.outputs[1] = {desc_, &second};
    }
    if (state_->operation_hook)
      state_->operation_hook("pop");
    if (state_->stop_after_pop) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      state_->stop_after_pop = false;
      ctx.cancelled->store(true);
    }
    return OpResult::Ok;
  }
  OpResult try_push(AsyncPushCtx &ctx) override {
    if (state_->blocked_push) {
      if (state_->operation_hook)
        state_->operation_hook("blocked push");
      return OpResult::NotReady;
    }
    float result;
    CUDA_CHECK(cudaMemcpyAsync(&result, ctx.inputs[0].data(), sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    state_->results.push_back(result);
    ++state_->sink_uses;
    if (state_->operation_hook)
      state_->operation_hook("push");
    storage_access().owned_input_storage(0).ptr = nullptr;
    if (settings_.value("ports", 1) == 2)
      storage_access().owned_input_storage(1).ptr = nullptr;
    return state_->results.size() == static_cast<size_t>(state_->frames) ? OpResult::Eof
                                                                         : OpResult::Ok;
  }
  void release_output(int index) override {
    storage_access().owned_output_storage(index).ptr = nullptr;
    if (index + 1 == settings_.value("ports", 1)) {
      ++state_->frame;
      ++state_->source_uses;
    }
    if (state_->operation_hook)
      state_->operation_hook(settings_.value("ports", 1) == 2 && index == 0 ? "release first"
                                                                            : "release");
  }

  std::optional<PointerSequence> owned_input_pointer_sequence(size_t) const override {
    return sequence();
  }
  std::optional<PointerSequence> owned_output_pointer_sequence(size_t) const override {
    return sequence();
  }
  bool compatible(const nlohmann::json &settings) const {
    return settings.value("count", size_t{2}) == count_ && settings.value("bad", false) == bad_;
  }
  void reconfigure(nlohmann::json settings, cudaStream_t stream, TDesc desc) {
    settings_ = std::move(settings);
    stream_   = stream;
    desc_     = std::move(desc);
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
  std::vector<std::byte *> pointers(size_t port) const {
    std::vector<std::byte *> result;
    for (size_t i = 0; i < count_; ++i)
      result.push_back(reinterpret_cast<std::byte *>(buffers_[port * (count_ + 1) + i].get()));
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
  ExecutionUpdatePolicy execution_update_policy() const noexcept override {
    return state_->explicit_updates ? ExecutionUpdatePolicy::ExplicitInvalidation
                                    : ExecutionUpdatePolicy::AlwaysInvalidate;
  }
  BoundaryFactory(bool source, std::shared_ptr<State> state) : source_(source), state_(state) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    const auto  count    = settings.value("count", size_t{2});
    const auto  declared = settings.value("declared", count);
    const TDesc desc =
        source_ ? TDesc({1}, DType::F32, MemLoc::Device, settings.value("offset", size_t{4}))
                : inputs[0];
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
    if (settings.value("ports", 1) == 2) {
      if (source_) {
        result.output_descs.push_back(desc);
        result.owned_outputs.push_back(true);
        result.owned_output_pointer_counts.push_back(declared);
      } else {
        result.input_descs.push_back(desc);
        result.owned_inputs.push_back(true);
        result.owned_input_pointer_counts.push_back(declared);
      }
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
  std::unique_ptr<IAsyncTask> update(std::unique_ptr<IAsyncTask> old_task,
                                     std::span<const TDesc> inputs, const nlohmann::json &settings,
                                     const AsyncCreateCtx &ctx) const override {
    ExecutionUpdateGuard guard(ctx.execution_invalidation);
    if (!state_->explicit_updates)
      EXPECT_TRUE(ctx.execution_invalidation->invalidated);
    auto *old = dynamic_cast<Boundary *>(old_task.get());
    if (old && old->compatible(settings)) {
      const auto inferred = infer(inputs, settings);
      old->reconfigure(settings, source_ ? ctx.consumer_stream : ctx.producer_stream,
                       source_ ? inferred.output_descs[0] : inputs[0]);
      return old_task;
    }
    ctx.invalidate_execution();
    return create(inputs, settings, ctx);
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
  ~Compute() override {
    if (state_->watched_task == this) {
      state_->watched_task = nullptr;
      state_->destruction_probe();
    }
  }
  const nlohmann::json &settings() const { return settings_; }
  void                  update_stream(cudaStream_t stream) { stream_ = stream; }
  bool                  supports_cuda_graph() const noexcept override {
    return !settings_.value("unsupported", false);
  }
  OpResult execute(SyncCtx &ctx) override {
    if (state_->cancel_sync_once) {
      state_->cancel_sync_once = false;
      if (state_->operation_hook)
        state_->operation_hook("cancel sync");
      return OpResult::Cancelled;
    }
    ++state_->executions;
    enqueue(ctx.inputs, ctx.outputs, stream_);
    if (state_->operation_hook)
      state_->operation_hook("sync");
    return OpResult::Ok;
  }
  void record_cuda_graph(CudaGraphCtx &ctx) override {
    ++state_->recordings;
    if (state_->recording_hook)
      state_->recording_hook();
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
  ExecutionUpdatePolicy execution_update_policy() const noexcept override {
    return state_->explicit_updates ? ExecutionUpdatePolicy::ExplicitInvalidation
                                    : ExecutionUpdatePolicy::AlwaysInvalidate;
  }
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
  std::unique_ptr<ISyncTask> update(std::unique_ptr<ISyncTask> old_task,
                                    std::span<const TDesc> inputs, const nlohmann::json &settings,
                                    const SyncCreateCtx &ctx) const override {
    ExecutionUpdateGuard guard(ctx.execution_invalidation);
    if (settings.value("throw_update", false))
      throw std::runtime_error("deliberate update failure");
    auto *old = dynamic_cast<Compute *>(old_task.get());
    if (old && old->settings() == settings) {
      old->update_stream(ctx.stream);
      return old_task;
    }
    ctx.invalidate_execution();
    return create(inputs, settings, ctx);
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

TEST_F(SectionCudaGraphTest, CompatibleRecompilationRetainsExecutableHandles) {
  auto out = compile();
  run(*out);
  const auto handles    = out->resources.section_cuda_graphs.begin()->second->executables;
  const int  recordings = state->recordings;
  out                   = compile(128, std::move(out));
  auto &graphs          = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_EQ(graphs.executables, handles);
  EXPECT_TRUE(graphs.snapshot()["carried_from_previous_compilation"]);
  EXPECT_EQ(graphs.snapshot()["compilation_generation"], 2);
  EXPECT_EQ(graphs.snapshot()["launches"], 0);
  state->frames = 26;
  run(*out);
  EXPECT_EQ(graphs.snapshot()["created"], 0);
  EXPECT_EQ(graphs.snapshot()["reused"], handles.size());
  EXPECT_EQ(state->recordings, recordings);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, DefaultPolicyInvalidatesBeforeTaskDestruction) {
  auto out = compile();
  run(*out);
  auto &graphs                  = *out->resources.section_cuda_graphs.begin()->second;
  state->explicit_updates       = false;
  state->watched_task           = out->resources.tasks.at("first").get();
  state->destruction_probe      = [&] { EXPECT_TRUE(graphs.executables.empty()); };
  spec[first].settings["scale"] = 3.F;
  out                           = compile(128, std::move(out));
  EXPECT_EQ(state->watched_task, nullptr);
  EXPECT_FALSE(
      out->resources.section_cuda_graphs.begin()->second->carried_from_previous_compilation);
  state->frames = 26;
  run(*out);
  for (size_t i = 13; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 3.F * i + 1);
}

TEST_F(SectionCudaGraphTest, ThrowingExplicitUpdateInvalidatesBeforeTaskDestruction) {
  auto out = compile();
  run(*out);
  auto &graphs             = *out->resources.section_cuda_graphs.begin()->second;
  bool  probed             = false;
  state->watched_task      = out->resources.tasks.at("first").get();
  state->destruction_probe = [&] {
    probed = true;
    EXPECT_TRUE(graphs.executables.empty());
  };
  spec[first].settings["throw_update"] = true;
  EXPECT_THROW(compile(128, std::move(out)), std::runtime_error);
  EXPECT_TRUE(probed);
}

TEST_F(SectionCudaGraphTest, DescriptorOffsetChangeRejectsCacheCarryover) {
  auto out = compile();
  run(*out);
  spec[source].settings["offset"] = 0;
  out                             = compile(128, std::move(out));
  const auto &graphs              = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.carried_from_previous_compilation);
  EXPECT_EQ(graphs.compilation_invalidation_reason, "descriptor_or_binding_change");
  state->frames = 26;
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, RecompilationReusesOverlappingReachableTuples) {
  spec[source].settings = {{"count", 2}, {"ordered", true}};
  spec[sink].settings   = {{"count", 2}, {"ordered", true}};
  auto out              = compile(4);
  run(*out);
  const auto old_handles = out->resources.section_cuda_graphs.begin()->second->executables;
  ASSERT_EQ(old_handles.size(), 2);
  spec[sink].settings["cycle"] = {0, 0, 1, 1};
  out                          = compile(4, std::move(out));
  state->frames                = 26;
  run(*out);
  const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_EQ(graphs.snapshot()["reused"], 2);
  EXPECT_EQ(graphs.snapshot()["created"], 2);
  EXPECT_EQ(graphs.snapshot()["discarded"], 0);
  EXPECT_EQ(graphs.snapshot()["tuple_misses"], 0);
  for (auto handle : old_handles)
    EXPECT_NE(std::find(graphs.executables.begin(), graphs.executables.end(), handle),
              graphs.executables.end());
}

TEST_F(SectionCudaGraphTest, StreamReplacementRejectsCacheCarryover) {
  auto out = compile();
  run(*out);
  curaii::CudaStream replacement;
  out->resources.streams.begin()->second = std::move(replacement);
  out                                    = compile(128, std::move(out));
  const auto &graphs                     = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.carried_from_previous_compilation);
  EXPECT_EQ(graphs.compilation_invalidation_reason, "stream_change");
  state->frames = 26;
  run(*out);
}

TEST_F(SectionCudaGraphTest, RenumberedStorageRejectsCacheCarryover) {
  auto out = compile();
  run(*out);
  add_vertex(NodeSpec{"unused_source", "source", {{"count", 1}}}, spec);
  out                = compile(128, std::move(out));
  const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.carried_from_previous_compilation);
  EXPECT_EQ(graphs.compilation_invalidation_reason, "descriptor_or_binding_change");
  state->frames = 26;
  run(*out);
}

TEST_F(SectionCudaGraphTest, RewiredSectionRejectsCacheCarryover) {
  auto out = compile();
  run(*out);
  remove_edge(source, first, spec);
  remove_edge(first, last, spec);
  remove_edge(last, sink, spec);
  add_edge(source, last, EdgeSpec{0, 0}, spec);
  add_edge(last, first, EdgeSpec{0, 0}, spec);
  add_edge(first, sink, EdgeSpec{0, 0}, spec);
  out = compile(128, std::move(out));
  EXPECT_FALSE(
      out->resources.section_cuda_graphs.begin()->second->carried_from_previous_compilation);
  state->frames = 26;
  run(*out);
  for (size_t i = 13; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * (i + 1));
}

TEST_F(SectionCudaGraphTest, AliasingChangeRejectsCacheCarryover) {
  auto out = compile();
  run(*out);
  spec[first].settings["alias"] = true;
  out                           = compile(128, std::move(out));
  EXPECT_FALSE(
      out->resources.section_cuda_graphs.begin()->second->carried_from_previous_compilation);
  state->frames = 26;
  run(*out);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, UnrelatedSectionSurvivesCapturedArgumentChange) {
  auto source2  = add_vertex(NodeSpec{"source2", "source", {{"count", 1}}}, spec);
  auto compute2 = add_vertex(NodeSpec{"compute2", "compute", {{"scale", 4.F}}}, spec);
  auto sink2    = add_vertex(NodeSpec{"sink2", "sink", {{"count", 1}}}, spec);
  add_edge(source2, compute2, EdgeSpec{0, 0}, spec);
  add_edge(compute2, sink2, EdgeSpec{0, 0}, spec);
  auto out = compile();
  // Prepare all sections without racing two synthetic sinks sharing the fixture's frame cursor.
  {
    Scheduler scheduler(out->graph, out->sections, out->resources);
    // Request cancellation during preparation, so both caches are populated but no work executes.
    state->recording_hook = [&] { scheduler.request_stop(); };
    scheduler.start();
    scheduler.wait();
  }
  state->recording_hook = {};
  std::vector<cudaGraphExec_t> handles;
  int                          preserved_id = -1;
  for (const auto &section : out->sections)
    if (out->graph[section.sync_topo.front()].spec.name == "compute2") {
      preserved_id = section.id;
      handles      = out->resources.section_cuda_graphs.at(section.id)->executables;
    }
  ASSERT_FALSE(handles.empty());
  spec[first].settings["scale"] = 3.F;
  out                           = compile(128, std::move(out));
  EXPECT_EQ(out->resources.section_cuda_graphs.at(preserved_id)->executables, handles);
  EXPECT_TRUE(
      out->resources.section_cuda_graphs.at(preserved_id)->carried_from_previous_compilation);
  for (const auto &[id, graphs] : out->resources.section_cuda_graphs)
    if (id != preserved_id)
      EXPECT_FALSE(graphs->carried_from_previous_compilation);
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

TEST_F(SectionCudaGraphTest, PauseRetainsEveryCompletedOperationAndGraphVariant) {
  // The 28/40 cycle has 280 reachable tuples; even a one-sided phase change loses them all.
  for (const bool use_graphs : {false, true}) {
    for (const auto point : {"acquire", "pop", "sync", "blocked push", "push", "release",
                             "blocked acquire", "blocked pop", "cancel sync"}) {
      if (use_graphs && (std::string(point) == "sync" || std::string(point) == "cancel sync"))
        continue;
      SCOPED_TRACE(std::string(point) + (use_graphs ? " graphs" : " ordinary"));
      state->frame = state->source_uses = state->sink_uses = state->executions = 0;
      state->acquisitions                                                      = 0;
      state->results.clear();
      spec[source].settings = {{"count", 28}, {"ordered", true}};
      spec[sink].settings   = {{"count", 40}, {"ordered", true}};
      auto      out         = compile(use_graphs ? 280 : 0);
      Scheduler scheduler(out->graph, out->sections, out->resources);
      bool      triggered     = false;
      state->cancel_sync_once = std::string(point) == "cancel sync";
      state->blocked_acquire  = std::string(point) == "blocked acquire";
      state->blocked_pop      = std::string(point) == "blocked pop";
      state->blocked_push     = std::string(point) == "blocked push";
      state->operation_hook   = [&](const char *operation) {
        if (!triggered && std::string(operation) == point) {
          triggered = true;
          scheduler.request_pause();
        }
      };
      scheduler.start();
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (!scheduler.stop_requested())
        scheduler.request_stop();
      scheduler.wait();
      ASSERT_TRUE(triggered);
      const auto recordings  = state->recordings;
      const auto handles     = use_graphs
                                   ? out->resources.section_cuda_graphs.begin()->second->executables
                                   : std::vector<cudaGraphExec_t>{};
      state->operation_hook  = {};
      state->blocked_acquire = state->blocked_pop = state->blocked_push = false;
      scheduler.start();
      deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (!scheduler.stop_requested())
        scheduler.request_stop();
      scheduler.wait();
      ASSERT_EQ(state->results.size(), state->frames);
      EXPECT_EQ(state->source_uses, state->frames);
      EXPECT_EQ(state->sink_uses, state->frames);
      EXPECT_EQ(state->acquisitions, state->frames);
      for (size_t i = 0; i < state->results.size(); ++i)
        EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
      EXPECT_EQ(state->recordings, recordings);
      if (use_graphs) {
        const auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
        EXPECT_EQ(graphs.executables, handles);
        EXPECT_EQ(graphs.snapshot()["tuple_misses"], 0);
        EXPECT_EQ(graphs.snapshot()["ordinary_iterations"], 0);
        EXPECT_EQ(graphs.snapshot()["launches"], state->frames);
      } else {
        EXPECT_EQ(state->executions, 2 * state->frames);
      }
    }
  }
}

TEST_F(SectionCudaGraphTest, PauseRetainsPartiallyAcquiredAndReleasedPorts) {
  spec[source].settings["ports"] = 2;
  spec[sink].settings["ports"]   = 2;
  const auto branch = add_vertex(NodeSpec{"branch", "compute", nlohmann::json::object()}, spec);
  add_edge(source, branch, EdgeSpec{1, 0}, spec);
  add_edge(branch, sink, EdgeSpec{0, 1}, spec);
  auto      out = compile(0);
  Scheduler scheduler(out->graph, out->sections, out->resources);
  int       pauses      = 0;
  state->operation_hook = [&](const char *point) {
    if ((pauses == 0 && std::string(point) == "acquire first") ||
        (pauses == 1 && std::string(point) == "release first")) {
      ++pauses;
      scheduler.request_pause();
    }
  };
  for (int run = 0; run < 3; ++run) {
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!scheduler.stop_requested())
      scheduler.request_stop();
    scheduler.wait();
    if (run == 0)
      EXPECT_EQ(state->acquisitions, 1);
    if (run == 1)
      EXPECT_EQ(state->source_uses, 0);
  }
  state->operation_hook = {};
  EXPECT_EQ(pauses, 2);
  ASSERT_EQ(state->results.size(), state->frames);
  EXPECT_EQ(state->acquisitions, 2 * state->frames);
  EXPECT_EQ(state->source_uses, state->frames);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, PausePreservesOrdinaryFallbackAfterTupleMiss) {
  spec[source].settings   = {{"count", 4}, {"ordered", true}};
  spec[sink].settings     = {{"count", 4}, {"ordered", true}};
  auto out                = compile(4);
  state->unexpected_tuple = true;
  Scheduler scheduler(out->graph, out->sections, out->resources);
  state->operation_hook = [&](const char *point) {
    if (std::string(point) == "push" && state->results.size() == 3)
      scheduler.request_pause();
  };
  scheduler.start();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!scheduler.stop_requested())
    scheduler.request_stop();
  scheduler.wait();
  ASSERT_EQ(state->results.size(), 3U);
  auto &graphs = *out->resources.section_cuda_graphs.begin()->second;
  EXPECT_FALSE(graphs.enabled);
  const auto recordings = state->recordings;
  state->operation_hook = {};
  scheduler.start();
  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!scheduler.stop_requested())
    scheduler.request_stop();
  scheduler.wait();
  EXPECT_FALSE(graphs.enabled);
  EXPECT_EQ(state->recordings, recordings);
  EXPECT_EQ(graphs.snapshot()["tuple_misses"], 1);
  EXPECT_EQ(graphs.snapshot()["launches"], 2);
  ASSERT_EQ(state->results.size(), state->frames);
  for (size_t i = 0; i < state->results.size(); ++i)
    EXPECT_FLOAT_EQ(state->results[i], 2.F * i + 1);
}

TEST_F(SectionCudaGraphTest, EofOverridesConcurrentPauseAndReleasesOutputs) {
  spec[source].settings = {{"count", 4}, {"ordered", true}};
  spec[sink].settings   = {{"count", 4}, {"ordered", true}};
  state->frames         = 1;
  auto      out         = compile(4);
  Scheduler scheduler(out->graph, out->sections, out->resources);
  state->operation_hook = [&](const char *point) {
    if (std::string(point) == "push")
      scheduler.request_pause();
  };
  scheduler.start();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!scheduler.stop_requested())
    scheduler.request_stop();
  scheduler.wait();
  EXPECT_EQ(state->source_uses, 1);
  state->operation_hook      = {};
  state->frames              = 2;
  int queries                = 0;
  state->sequence_query_hook = [&] { ++queries; };
  scheduler.start();
  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!scheduler.stop_requested())
    scheduler.request_stop();
  scheduler.wait();
  state->sequence_query_hook = {};
  EXPECT_GT(queries, 0);
  ASSERT_EQ(state->results.size(), 2U);
  EXPECT_FLOAT_EQ(state->results[0], 1.F);
  EXPECT_FLOAT_EQ(state->results[1], 3.F);
  EXPECT_EQ(state->source_uses, 2);
}

TEST_F(SectionCudaGraphTest, DestroyingPausedSchedulerReleasesHeldOutputsOnce) {
  auto out = compile();
  {
    Scheduler scheduler(out->graph, out->sections, out->resources);
    state->operation_hook = [&](const char *point) {
      if (std::string(point) == "pop")
        scheduler.request_pause();
    };
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!scheduler.stop_requested())
      scheduler.request_stop();
    scheduler.wait();
    ASSERT_TRUE(state->results.empty());
    EXPECT_EQ(state->source_uses, 0);
    state->operation_hook = {};
  }
  EXPECT_EQ(state->source_uses, 1);
  EXPECT_EQ(state->sink_uses, 0);
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
    section_diagnostics_file_writer().flush();
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

TEST_F(SectionCudaGraphTest, FirstStartupUsesCompiledInspectionAndRestartReinspects) {
  int queries                 = 0;
  state->sequence_query_hook  = [&] { ++queries; };
  auto       out              = compile();
  const auto compiled_queries = queries;
  ASSERT_GT(compiled_queries, 0);
  ASSERT_NE(out->resources.section_cuda_graph_inspection, nullptr);
  EXPECT_EQ(state->recordings, 0);
  run(*out);
  EXPECT_EQ(queries, compiled_queries);
  EXPECT_EQ(out->resources.section_cuda_graph_inspection, nullptr);
  EXPECT_TRUE(diagnostics[0]["inspection_reused"].get<bool>());
  EXPECT_EQ(diagnostics[0]["inspection_ms"], 0.0);
  EXPECT_TRUE(diagnostics[0].contains("compiled_inspection_ms"));
  state->frames = 26;
  run(*out);
  EXPECT_GT(queries, compiled_queries);
  EXPECT_FALSE(diagnostics[0]["inspection_reused"].get<bool>());
  EXPECT_EQ(diagnostics[0]["refresh_count"], 2);
  state->sequence_query_hook = {};
}

TEST_F(SectionCudaGraphTest, RecompilationRetainsFreshInspectionForStartup) {
  int queries                = 0;
  state->sequence_query_hook = [&] { ++queries; };
  auto out                   = compile();
  run(*out);
  ASSERT_EQ(out->resources.section_cuda_graph_inspection, nullptr);
  out                         = compile(128, std::move(out));
  const auto compiled_queries = queries;
  ASSERT_NE(out->resources.section_cuda_graph_inspection, nullptr);
  state->frames = 26;
  run(*out);
  EXPECT_EQ(queries, compiled_queries);
  EXPECT_TRUE(diagnostics[0]["inspection_reused"].get<bool>());
  EXPECT_EQ(diagnostics[0]["created"], 0);
  EXPECT_EQ(diagnostics[0]["reused"], 6);
  state->sequence_query_hook = {};
}

TEST_F(SectionCudaGraphTest, ChangedGraphCapInvalidatesCompiledInspection) {
  auto out                               = compile();
  int  queries                           = 0;
  state->sequence_query_hook             = [&] { ++queries; };
  out->resources.max_section_cuda_graphs = 1;
  run(*out);
  EXPECT_GT(queries, 0);
  EXPECT_FALSE(diagnostics[0]["inspection_reused"].get<bool>());
  EXPECT_FALSE(diagnostics[0]["enabled"].get<bool>());
  EXPECT_GT(state->executions, 0);
  state->sequence_query_hook = {};
}

TEST_F(SectionCudaGraphTest, StopRequestedDuringPreparationIsNotLost) {
  auto out = compile();
  {
    Scheduler scheduler(out->graph, out->sections, out->resources);
    state->recording_hook = [&] { scheduler.request_stop(); };
    scheduler.start();
    EXPECT_TRUE(scheduler.stop_requested());
    scheduler.wait();
    state->recording_hook = {};
  }
  EXPECT_EQ(state->frame, 0);
  EXPECT_EQ(state->acquisitions, 0);
  run(*out);
  EXPECT_EQ(diagnostics[0]["reused"], 6);
  EXPECT_EQ(state->executions, 0);
}

} // namespace

TEST_F(SectionCudaGraphTest, CapturesStartupVariantConstructionAndReuse) {
  if (!tracing::Session::available())
    GTEST_SKIP() << "SDK disabled";
  auto       out = compile(6);
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("holoflow-startup-detail-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  for (bool details : {true, false}) {
    auto session = tracing::Session::start({.include_details = details});
    ASSERT_NE(session, nullptr);
    for (int start = 0; start < 2; ++start) {
      Scheduler scheduler(out->graph, out->sections, out->resources);
      scheduler.start();
      scheduler.request_stop();
      scheduler.wait();
    }
    const auto report = out->resources.section_cuda_graphs.begin()->second->snapshot();
    EXPECT_EQ(report["reused"], 6);
    EXPECT_EQ(report["created"], 0);
    const auto path = directory / (details ? "detailed.perfetto-trace" : "filtered.perfetto-trace");
    session->stop_and_save(path);
    EXPECT_GT(std::filesystem::file_size(path), 0U);
  }
  EXPECT_EQ(state->recordings, 12);
}
