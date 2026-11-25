// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <thread>

#include "holofile/holofile.hh"
#include "holoflow/runtime/compiler.hh"
#include "holonp/reshape.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/pca_decomposition_queue.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

class TemporaryHolofile {
public:
  TemporaryHolofile(size_t width = 16, size_t height = 8, size_t frames = 13) {
    static std::atomic<size_t> sequence{0};
    path = std::filesystem::temp_directory_path() /
           ("holoflow-prequeue-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(sequence++) + ".holo");
    holofile::Header header{};
    header.magic_number       = holofile::Header::MAGIC_NUMBER_LE;
    header.version            = holofile::Header::CURRENT_VERSION;
    header.bits_per_pixel     = 8;
    header.frame_width        = static_cast<uint32_t>(width);
    header.frame_height       = static_cast<uint32_t>(height);
    header.frame_count        = static_cast<uint32_t>(frames);
    header.data_size_in_bytes = width * height * frames;
    header.endianness         = holofile::Header::LITTLE_ENDIAN;
    pixels                    = width * height;
    values.resize(pixels * frames);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<uint8_t>((i / pixels) * 13 + i % pixels);
    holofile::Writer writer(path.string(), header, holofile::Footer{nlohmann::json::object()});
    writer.write_frames(values.data(), frames);
    writer.write_footer();
  }
  ~TemporaryHolofile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  std::filesystem::path path;
  size_t                pixels;
  std::vector<uint8_t>  values;
};

struct PlaybackState {
  bool                                  performance = false;
  size_t                                duration_ms = 10000, target = 32, batches = 0;
  std::chrono::steady_clock::time_point started;
  double                                batches_per_second = 0;
  std::vector<uint8_t>                  values;
  std::function<void(size_t)>           hook;
};

class PlaybackSink final : public ISyncTask {
public:
  PlaybackSink(cudaStream_t stream, std::shared_ptr<PlaybackState> state)
      : stream_(stream), state_(std::move(state)) {}
  OpResult execute(SyncCtx &ctx) override {
    if (state_->performance) {
      uint8_t value;
      CUDA_CHECK(cudaMemcpyAsync(&value, ctx.inputs[0].data(), 1, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else {
      const auto offset = state_->values.size();
      const auto bytes  = ctx.inputs[0].desc.num_bytes();
      state_->values.resize(offset + bytes);
      CUDA_CHECK(cudaMemcpyAsync(state_->values.data() + offset, ctx.inputs[0].data(), bytes,
                                 cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    ++state_->batches;
    if (state_->hook)
      state_->hook(state_->batches);
    if (!state_->performance)
      return state_->batches == state_->target ? OpResult::Eof : OpResult::Ok;
    const auto now = std::chrono::steady_clock::now();
    if (state_->batches == 128)
      state_->started = now;
    if (state_->batches > 128 &&
        now - state_->started >= std::chrono::milliseconds(state_->duration_ms)) {
      state_->batches_per_second =
          (state_->batches - 128) / std::chrono::duration<double>(now - state_->started).count();
      return OpResult::Eof;
    }
    return OpResult::Ok;
  }

private:
  cudaStream_t                   stream_;
  std::shared_ptr<PlaybackState> state_;
};

class PlaybackSinkFactory final : public ISyncTaskFactory {
public:
  explicit PlaybackSinkFactory(std::shared_ptr<PlaybackState> state) : state_(std::move(state)) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<PlaybackSink>(ctx.stream, state_);
  }

private:
  std::shared_ptr<PlaybackState> state_;
};

std::shared_ptr<PlaybackState> run_playback(const TemporaryHolofile &file, bool prequeue,
                                            bool pause = false, bool stop_paused = false,
                                            size_t capacity = 8, bool performance = false,
                                            bool pca = false, size_t duration_ms = 10000) {
  auto state         = std::make_shared<PlaybackState>();
  state->performance = performance;
  state->duration_ms = duration_ms;
  Registry registry;
  registry.register_sync("source", std::make_unique<holotask::sources::HolofileFactory>());
  registry.register_sync("memcpy", std::make_unique<holotask::syncs::MemcpyFactory>());
  registry.register_sync("sink", std::make_unique<PlaybackSinkFactory>(state));
  registry.register_async("batch", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_sync("float", std::make_unique<holotask::syncs::ConversionFactory>());
  registry.register_sync("reshape", std::make_unique<holonp::ReshapeFactory>());
  registry.register_sync("projection", std::make_unique<holotask::syncs::PcaProjectionFactory>());
  registry.register_async("decomposition",
                          std::make_unique<holotask::asyncs::PcaDecompositionQueueFactory>());
  const int                                 batch = performance ? 32 : 2;
  const int                                 begin = performance ? 0 : 3;
  const int                                 end   = performance ? 128 : 12;
  const holotask::sources::HolofileSettings settings{
      file.path.string(),
      holotask::sources::HolofileSettings::LoadKind::CPUCached,
      begin,
      end,
      batch,
      std::nullopt,
      true};
  GraphSpec  spec;
  const auto source = add_vertex(NodeSpec{"source", "source", settings}, spec);
  const auto copy   = add_vertex(
      NodeSpec{"memcpy", "memcpy",
               holotask::syncs::MemcpySettings{holotask::syncs::MemcpySettings::Target::Device}},
      spec);
  const auto input = add_vertex(
      NodeSpec{"input", "batch",
               holotask::asyncs::BatchQueueSettings{static_cast<int>(capacity), batch, batch}},
      spec);
  auto output = input;
  add_edge(source, copy, EdgeSpec{0, 0}, spec);
  add_edge(copy, input, EdgeSpec{0, 0}, spec);
  if (pca) {
    auto convert = add_vertex(NodeSpec{"float", "float",
                                       holotask::syncs::ConversionSettings{
                                           holotask::syncs::ConversionSettings::Target::F32,
                                           holotask::syncs::ConversionSettings::Strategy::Real}},
                              spec);
    auto reshape = add_vertex(
        NodeSpec{"reshape", "reshape", holonp::ReshapeSettings{{1, 32, 320, 512}, false}}, spec);
    auto decomposition = add_vertex(NodeSpec{"decomposition", "decomposition",
                                             holotask::asyncs::PcaDecompositionQueueSettings{4, 3}},
                                    spec);
    auto projection =
        add_vertex(NodeSpec{"projection", "projection", holotask::syncs::PcaSettings{1, 16}}, spec);
    output = add_vertex(NodeSpec{"output", "batch", holotask::asyncs::BatchQueueSettings{8, 1, 1}},
                        spec);
    add_edge(input, convert, EdgeSpec{0, 0}, spec);
    add_edge(convert, reshape, EdgeSpec{0, 0}, spec);
    add_edge(reshape, decomposition, EdgeSpec{0, 0}, spec);
    add_edge(decomposition, projection, EdgeSpec{0, 0}, spec);
    add_edge(decomposition, projection, EdgeSpec{1, 1}, spec);
    add_edge(projection, output, EdgeSpec{0, 0}, spec);
  }
  const auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, spec);
  add_edge(output, sink, EdgeSpec{0, 0}, spec);
  Compiler::Config config;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.dump_dot_on_failure     = false;
  config.max_section_cuda_graphs = performance ? 128 : 0;
  auto compiled                  = Compiler(registry, config).compile(spec);
  if (!prequeue)
    for (auto &section : compiled->sections)
      if (compiled->graph[section.sync_topo.front()].spec.name == "source")
        section.supports_prequeue = false; // Isolate section-0 in matched performance runs.
  Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  if (pause)
    state->hook = [&](size_t count) {
      if (count == 5 || count == 11)
        scheduler.request_pause();
    };
  static const nvtx3::registered_string queued_range{"Cached playback prequeued"};
  static const nvtx3::registered_string serial_range{"Cached playback serial"};
  nvtx3::scoped_range                   range{prequeue ? queued_range : serial_range};
  for (size_t pass = 0; pass < (pause ? 3U : 1U); ++pass) {
    scheduler.start();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(performance ? duration_ms + 30000 : 10000);
    while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool stopped = scheduler.stop_requested();
    if (!stopped)
      scheduler.request_stop();
    scheduler.wait();
    EXPECT_TRUE(stopped) << "Cached playback deadlocked";
    if (!stopped || stop_paused)
      break;
  }
  if (stop_paused) {
    scheduler.request_stop();
    scheduler.wait();
  }
  if (!performance) {
    EXPECT_EQ(state->batches, stop_paused ? 5U : state->target);
    const size_t batch_bytes = file.pixels * batch;
    for (size_t i = 0; i < state->batches; ++i) {
      const size_t frame = begin + (i % ((end - begin) / batch)) * batch;
      EXPECT_EQ(state->values.size(), state->batches * batch_bytes);
      if (state->values.size() != state->batches * batch_bytes)
        return state;
      EXPECT_TRUE(std::equal(file.values.begin() + frame * file.pixels,
                             file.values.begin() + frame * file.pixels + batch_bytes,
                             state->values.begin() + i * batch_bytes))
          << "batch " << i;
    }
  }
  for (const auto &section : compiled->sections)
    if (compiled->graph[section.sync_topo.front()].spec.name == "source") {
      const auto &graphs = *compiled->resources.section_cuda_graphs.at(section.id);
      EXPECT_EQ(graphs.snapshot()["prequeue_execution_mode"], prequeue ? "ordinary" : "serial");
      EXPECT_EQ(graphs.launches.load(), 0U);
      EXPECT_EQ(graphs.prequeue_submissions.load(), graphs.prequeue_retirements.load());
      if (prequeue)
        EXPECT_GT(graphs.prequeue_submissions.load(), 0U);
    }
  return state;
}

TEST(OrdinaryPrequeue, CachedPlaybackPreservesBytesAcrossWraparoundPauseAndStop) {
  TemporaryHolofile file;
  run_playback(file, false);
  run_playback(file, true);
  run_playback(file, true, true);
  run_playback(file, true, true, true);
  run_playback(file, true, false, false, 2);
}

class SourceStorage final : public IOStorageAccess {
public:
  explicit SourceStorage(const TDesc &desc) : output{desc.mem_loc, desc.num_bytes(), nullptr} {}
  Storage &owned_input_storage(size_t) override { throw std::out_of_range("source input"); }
  Storage &owned_output_storage(size_t) override { return output; }
  Storage  output;
};

TEST(OrdinaryPrequeue, CachedSourceUpdatesPreserveCursorAndPacingEligibility) {
  TemporaryHolofile                   file;
  holotask::sources::HolofileFactory  factory;
  holotask::sources::HolofileSettings settings{
      file.path.string(),
      holotask::sources::HolofileSettings::LoadKind::CPUCached,
      3,
      12,
      2,
      std::nullopt,
      true};
  const auto         infer = factory.infer({}, settings);
  SourceStorage      storage(infer.output_descs[0]);
  std::array         outputs{TView{infer.output_descs[0], &storage.output}};
  std::atomic<bool>  cancelled{false};
  SyncCtx            execution{{}, outputs, &cancelled, nullptr, nullptr};
  curaii::CudaStream stream;
  auto               task = factory.create({}, settings, {stream.get()});
  task->bind_storage_access(&storage);
  EXPECT_TRUE(task->supports_prequeued_execution());
  ASSERT_EQ(task->execute(execution), OpResult::Ok);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(outputs[0].data()), file.values[3 * file.pixels]);
  task->release_output(0);
  task = factory.update(std::move(task), {}, settings, {stream.get()});
  task->bind_storage_access(&storage);
  ASSERT_EQ(task->execute(execution), OpResult::Ok);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(outputs[0].data()), file.values[5 * file.pixels]);
  task->release_output(0);
  settings.keep_cursor = false;
  task                 = factory.update(std::move(task), {}, settings, {stream.get()});
  task->bind_storage_access(&storage);
  ASSERT_EQ(task->execute(execution), OpResult::Ok);
  EXPECT_EQ(*reinterpret_cast<uint8_t *>(outputs[0].data()), file.values[3 * file.pixels]);
  task->release_output(0);
  settings.max_fps = 1000;
  task             = factory.update(std::move(task), {}, settings, {stream.get()});
  EXPECT_FALSE(task->supports_prequeued_execution());
  settings.max_fps.reset();
  settings.load_kind = holotask::sources::HolofileSettings::LoadKind::Live;
  task               = factory.update(std::move(task), {}, settings, {stream.get()});
  EXPECT_FALSE(task->supports_prequeued_execution());
  settings.load_kind     = holotask::sources::HolofileSettings::LoadKind::GPUCached;
  task                   = factory.update(std::move(task), {}, settings, {stream.get()});
  storage.output.mem_loc = MemLoc::Device;
  outputs[0].desc        = factory.infer({}, settings).output_descs[0];
  task->bind_storage_access(&storage);
  EXPECT_TRUE(task->supports_prequeued_execution());
  ASSERT_EQ(task->execute(execution), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  uint8_t first;
  CUDA_CHECK(cudaMemcpy(&first, outputs[0].data(), 1, cudaMemcpyDeviceToHost));
  EXPECT_EQ(first, file.values[3 * file.pixels]);
  task->release_output(0);
  holotask::syncs::MemcpyFactory memcpy;
  const std::array               host{TDesc({1}, DType::U8, MemLoc::Host)};
  auto                           host_copy = memcpy.create(
      host, holotask::syncs::MemcpySettings{holotask::syncs::MemcpySettings::Target::Host},
      {stream.get()});
  EXPECT_FALSE(host_copy->supports_prequeued_execution());
}

struct LoanState {
  LoanState() : cache(curaii::make_unique_host_ptr<uint8_t>(3)) {
    cache.get()[0] = 11;
    cache.get()[1] = 22;
    cache.get()[2] = 33;
    for (auto &event : copied)
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  ~LoanState() {
    for (auto event : copied)
      CUDA_CHECK_NT(cudaEventDestroy(event));
  }
  std::array<cudaEvent_t, 3>       copied{};
  curaii::unique_host_ptr<uint8_t> cache;
  std::atomic<size_t>              submissions{0}, releases{0};
  std::atomic<bool>                gate{false}, bad_release{false};
  std::atomic<bool> refill_allowed{true}, second_started{false}, refill_wait_observed{false};
  bool              hold_second = false;
  size_t            fail_after  = 0;
};

void CUDART_CB hold_copy(void *argument) {
  auto &state = *static_cast<LoanState *>(argument);
  while (!state.gate.load())
    std::this_thread::yield();
}

void CUDART_CB hold_second_copy(void *argument) {
  static_cast<LoanState *>(argument)->second_started.store(true);
  hold_copy(argument);
}

class LoanTask final : public ISyncTask {
public:
  LoanTask(bool source, cudaStream_t stream, std::shared_ptr<LoanState> state)
      : source_(source), stream_(stream), state_(std::move(state)) {}
  bool     supports_prequeued_execution() const noexcept override { return true; }
  OpResult execute(SyncCtx &ctx) override {
    if (source_) {
      const auto next = state_->submissions.load();
      storage_access().owned_output_storage(0).ptr =
          reinterpret_cast<std::byte *>(state_->cache.get() + next % 3);
      state_->submissions.fetch_add(1);
    } else {
      ++submitted_;
      if (state_->hold_second && submitted_ == 2)
        CUDA_CHECK(cudaLaunchHostFunc(stream_, hold_second_copy, state_.get()));
      else if (!state_->hold_second && submitted_ == 1)
        CUDA_CHECK(cudaLaunchHostFunc(stream_, hold_copy, state_.get()));
      CUDA_CHECK(cudaMemcpyAsync(ctx.outputs[0].data(), ctx.inputs[0].data(), 1,
                                 cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaEventRecord(state_->copied[(submitted_ - 1) % 3], stream_));
      if (state_->fail_after == submitted_)
        throw std::runtime_error("injected submission failure");
    }
    return OpResult::Ok;
  }
  void release_output(int) override {
    const auto next = state_->releases.load();
    if (cudaEventQuery(state_->copied[next % 3]) != cudaSuccess ||
        storage_access().owned_output_storage(0).ptr !=
            reinterpret_cast<std::byte *>(state_->cache.get() + next % 3))
      state_->bad_release.store(true);
    state_->releases.fetch_add(1);
    storage_access().owned_output_storage(0).ptr = nullptr;
  }

private:
  bool                       source_;
  cudaStream_t               stream_;
  std::shared_ptr<LoanState> state_;
  size_t                     submitted_ = 0;
};

class LoanFactory final : public ISyncTaskFactory {
public:
  LoanFactory(bool source, std::shared_ptr<LoanState> state)
      : source_(source), state_(std::move(state)) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (source_)
      return {{}, {TDesc({1}, DType::U8, MemLoc::Host)}, {}, {}, {true}, TaskKind::Sync};
    return {{inputs[0]},   {TDesc({1}, DType::U8, MemLoc::Device)}, {}, {false}, {false},
            TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<LoanTask>(source_, ctx.stream, state_);
  }

private:
  bool                       source_;
  std::shared_ptr<LoanState> state_;
};

// Simulate an independently changing downstream readiness condition, without changing the real
// BatchQueue's reservations, capacity, or publication protocol.
class GatedQueue final : public IAsyncTask {
public:
  GatedQueue(std::unique_ptr<IAsyncTask> inner, std::shared_ptr<LoanState> state)
      : inner_(std::move(inner)), state_(std::move(state)) {}
  OpResult reserve_push(AsyncPushCtx &ctx, AsyncTicket &ticket) override {
    if (ctx.cancelled && ctx.cancelled->load())
      return OpResult::Cancelled;
    if (state_->submissions.load() >= 2 && !state_->refill_allowed.load()) {
      state_->refill_wait_observed.store(true);
      return OpResult::NotReady;
    }
    return bind().reserve_push(ctx, ticket);
  }
  void     enqueue_push(AsyncTicket ticket) override { bind().enqueue_push(ticket); }
  void     commit_push(AsyncTicket ticket) override { bind().commit_push(ticket); }
  void     cancel_push(AsyncTicket ticket) override { bind().cancel_push(ticket); }
  OpResult reserve_pop(AsyncPopCtx &ctx, AsyncTicket &ticket) override {
    return bind().reserve_pop(ctx, ticket);
  }
  void                 release_pop(AsyncTicket ticket) override { bind().release_pop(ticket); }
  void                 cancel_pop(AsyncTicket ticket) override { bind().cancel_pop(ticket); }
  OpResult             try_push(AsyncPushCtx &ctx) override { return bind().try_push(ctx); }
  OpResult             try_pop(AsyncPopCtx &ctx) override { return bind().try_pop(ctx); }
  std::optional<TView> acquire_input(int port) override { return bind().acquire_input(port); }
  void                 release_output(int port) override { bind().release_output(port); }

private:
  IAsyncTask &bind() {
    inner_->bind_storage_access(&storage_access());
    return *inner_;
  }
  std::unique_ptr<IAsyncTask> inner_;
  std::shared_ptr<LoanState>  state_;
};

class GatedQueueFactory final : public IAsyncTaskFactory {
public:
  explicit GatedQueueFactory(std::shared_ptr<LoanState> state) : state_(std::move(state)) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    return holotask::asyncs::BatchQueueFactory{}.infer(inputs, settings);
  }
  std::unique_ptr<IAsyncTask> create(std::span<const TDesc> inputs, const nlohmann::json &settings,
                                     const AsyncCreateCtx &ctx) const override {
    return std::make_unique<GatedQueue>(
        holotask::asyncs::BatchQueueFactory{}.create(inputs, settings, ctx), state_);
  }

private:
  std::shared_ptr<LoanState> state_;
};

std::unique_ptr<CompilerOutput> loan_graph(Registry                         &registry,
                                           const std::shared_ptr<LoanState> &state, size_t capacity,
                                           bool gated_refill = false) {
  registry.register_sync("source", std::make_unique<LoanFactory>(true, state));
  registry.register_sync("copy", std::make_unique<LoanFactory>(false, state));
  registry.register_sync("sink",
                         std::make_unique<PlaybackSinkFactory>(std::make_shared<PlaybackState>()));
  if (gated_refill)
    registry.register_async("queue", std::make_unique<GatedQueueFactory>(state));
  else
    registry.register_async("queue", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  GraphSpec  spec;
  const auto source = add_vertex(NodeSpec{"source", "source", {}}, spec);
  const auto copy   = add_vertex(NodeSpec{"copy", "copy", {}}, spec);
  const auto queue =
      add_vertex(NodeSpec{"queue", "queue",
                          holotask::asyncs::BatchQueueSettings{static_cast<int>(capacity), 1, 1}},
                 spec);
  const auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, spec);
  add_edge(source, copy, EdgeSpec{0, 0}, spec);
  add_edge(copy, queue, EdgeSpec{0, 0}, spec);
  add_edge(queue, sink, EdgeSpec{0, 0}, spec);
  Compiler::Config config;
  config.max_section_cuda_graphs = 0;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.dump_dot_on_failure     = false;
  return Compiler(registry, config).compile(spec);
}

TEST(OrdinaryPrequeue, SubmitsTwoBeforeCompletionAndRetainsSourceLoans) {
  for (size_t capacity : {size_t{1}, size_t{4}}) {
    auto      state = std::make_shared<LoanState>();
    Registry  registry;
    auto      compiled = loan_graph(registry, state, capacity);
    Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
    scheduler.start();
    const size_t expected = capacity == 1 ? 1 : 2;
    const auto   deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state->submissions.load() < expected && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    const auto submitted = state->submissions.load();
    const auto released  = state->releases.load();
    scheduler.request_pause();
    state->gate.store(true); // Unblock before any assertion could leave the worker waiting.
    scheduler.wait();
    scheduler.request_stop();
    scheduler.wait();
    EXPECT_EQ(submitted, expected);
    EXPECT_EQ(released, 0U);
    EXPECT_EQ(state->releases.load(), state->submissions.load());
    EXPECT_FALSE(state->bad_release.load());
  }
}

TEST(OrdinaryPrequeue, SubmissionFailureDrainsBeforeReleasingSource) {
  EXPECT_EXIT(
      {
        auto state        = std::make_shared<LoanState>();
        state->fail_after = 2;
        state->gate.store(true);
        Registry  registry;
        auto      compiled = loan_graph(registry, state, 4);
        Scheduler scheduler(
            compiled->graph, compiled->sections, compiled->resources,
            std::chrono::milliseconds{1000}, [&](auto, auto, auto) {
              std::_Exit(state->releases.load() == 2 && !state->bad_release.load() ? 0 : 3);
            });
        scheduler.start();
        scheduler.wait();
        std::_Exit(4);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(OrdinaryPrequeue, RefillsWhileRemainingIterationIsIncomplete) {
  auto state         = std::make_shared<LoanState>();
  state->hold_second = true;
  state->refill_allowed.store(false);
  Registry  registry;
  auto      compiled = loan_graph(registry, state, 4, true);
  Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  static const nvtx3::registered_string trial_name{"Delayed refill policy trial"};
  nvtx3::scoped_range                   trial{trial_name};
  scheduler.start();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while ((!state->second_started.load() || !state->refill_wait_observed.load()) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool second_started       = state->second_started.load();
  const bool refill_wait_observed = state->refill_wait_observed.load();
  // The previous policy blocked retiring the second iteration and could not observe this
  // newly available buffer until the GPU gate opened. The new policy must submit iteration 3.
  state->refill_allowed.store(true);
  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (state->submissions.load() < 3 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const auto submitted     = state->submissions.load();
  const auto second_status = cudaEventQuery(state->copied[1]);
  scheduler.request_pause();
  state->gate.store(true); // Always unblock before assertions or teardown.
  scheduler.wait();
  scheduler.request_stop();
  scheduler.wait();
  EXPECT_TRUE(second_started);
  EXPECT_TRUE(refill_wait_observed);
  EXPECT_EQ(submitted, 3U);
  EXPECT_EQ(second_status, cudaErrorNotReady);
  EXPECT_EQ(state->releases.load(), state->submissions.load());
  EXPECT_FALSE(state->bad_release.load());
}

size_t benchmark_option(const char *name, size_t fallback) {
  char  *value = nullptr;
  size_t size  = 0;
  if (_dupenv_s(&value, &size, name) != 0 || !value)
    return fallback;
  const auto result = std::stoull(value);
  std::free(value);
  return static_cast<size_t>(result);
}

void benchmark_playback(bool pca) {
  TemporaryHolofile file(512, 320, 128);
  const auto        runs     = benchmark_option("HOLOFLOW_SOURCE_BENCHMARK_RUNS", 5);
  const auto        duration = benchmark_option("HOLOFLOW_SOURCE_BENCHMARK_MS", 10000);
  const auto        selected = benchmark_option("HOLOFLOW_SOURCE_BENCHMARK_MODE", 0);
  std::array<std::vector<double>, 2> rates;
  for (size_t run = 0; run < runs; ++run)
    for (size_t order = 0; order < 2; ++order) {
      const size_t mode = (run + order) % 2;
      if (selected && selected != mode + 1)
        continue;
      const auto state = run_playback(file, mode != 0, false, false, 128, true, pca, duration);
      EXPECT_GT(state->batches_per_second, 0);
      rates[mode].push_back(state->batches_per_second);
      std::cout << "Cached playback pca=" << pca << " prequeue=" << mode << " run=" << run
                << " windows_per_second=" << state->batches_per_second << std::endl;
    }
  for (size_t mode = 0; mode < 2; ++mode) {
    std::sort(rates[mode].begin(), rates[mode].end());
    if (!rates[mode].empty())
      std::cout << "Cached playback pca=" << pca << " prequeue=" << mode
                << " median=" << rates[mode][rates[mode].size() / 2]
                << " min=" << rates[mode].front() << " max=" << rates[mode].back() << std::endl;
  }
}

TEST(OrdinaryPrequeue, PerformanceCachedCopy) { benchmark_playback(false); }
TEST(OrdinaryPrequeue, PerformanceCachedPcaPipeline) { benchmark_playback(true); }

} // namespace
