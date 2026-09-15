// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

#include "holoflow/runtime/compiler.hh"
#include "holonp/fftshift.hh"
#include "holonp/mean.hh"
#include "holonp/reshape.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/dual_reader_batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/flatfield.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"

namespace {
using namespace holoflow::core;
using namespace holoflow::runtime;

struct ReferenceState {
  size_t                                             target_frames = 96;
  std::vector<std::vector<unsigned char>>            frames;
  std::vector<std::chrono::steady_clock::time_point> arrival;
};

class ReferenceTask : public ISyncTask {
public:
  ReferenceTask(bool source, cudaStream_t stream, std::shared_ptr<ReferenceState> state)
      : source_(source), stream_(stream), state_(state) {}
  OpResult execute(SyncCtx &ctx) override {
    if (source_) {
      auto *output = reinterpret_cast<unsigned char *>(ctx.outputs[0].data());
      for (size_t i = 0; i < ctx.outputs[0].desc.num_elements(); ++i) {
        // Deterministic changing input, independent of queue scheduling.
        random_ ^= random_ << 13;
        random_ ^= random_ >> 17;
        random_ ^= random_ << 5;
        output[i] = static_cast<unsigned char>(random_);
      }
      return OpResult::Ok;
    }
    std::vector<unsigned char> frame(ctx.inputs[0].desc.num_bytes());
    CUDA_CHECK(cudaMemcpyAsync(frame.data(), ctx.inputs[0].data(), frame.size(),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    state_->frames.push_back(std::move(frame));
    state_->arrival.push_back(std::chrono::steady_clock::now());
    return state_->frames.size() >= state_->target_frames ? OpResult::Eof : OpResult::Ok;
  }

private:
  bool                            source_;
  cudaStream_t                    stream_;
  std::shared_ptr<ReferenceState> state_;
  uint32_t                        random_ = 42;
};

class ReferenceFactory : public ISyncTaskFactory {
public:
  ReferenceFactory(bool source, std::shared_ptr<ReferenceState> state)
      : source_(source), state_(state) {}
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &settings) const override {
    if (source_) {
      return {{},
              {TDesc({settings.at("batch_size").get<size_t>(), settings.at("height").get<size_t>(),
                      settings.at("width").get<size_t>()},
                     DType::U8, MemLoc::Host)},
              {},
              {},
              {false},
              TaskKind::Sync};
    }
    return {{inputs.begin(), inputs.end()}, {}, {}, {false}, {}, TaskKind::Sync};
  }
  std::unique_ptr<ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                    const SyncCreateCtx &ctx) const override {
    return std::make_unique<ReferenceTask>(source_, ctx.stream, state_);
  }

private:
  bool                            source_;
  std::shared_ptr<ReferenceState> state_;
};

Registry reference_registry(const std::shared_ptr<ReferenceState> &state) {
  Registry registry;
  registry.register_sync("ReferenceSource", std::make_unique<ReferenceFactory>(true, state));
  registry.register_sync("ReferenceSink", std::make_unique<ReferenceFactory>(false, state));
  registry.register_sync("Memcpy", std::make_unique<holotask::syncs::MemcpyFactory>());
  registry.register_sync("Conversion", std::make_unique<holotask::syncs::ConversionFactory>());
  registry.register_sync("Reshape", std::make_unique<holonp::ReshapeFactory>());
  registry.register_sync("Pca", std::make_unique<holotask::syncs::PcaFactory>());
  registry.register_sync("Filter2D", std::make_unique<holotask::syncs::Filter2DFactory>());
  registry.register_sync("FresnelDiffraction",
                         std::make_unique<holotask::syncs::FresnelDiffractionFactory>());
  registry.register_sync("Mean", std::make_unique<holonp::MeanFactory>());
  registry.register_sync("FFTShiftNp", std::make_unique<holonp::FFTShiftFactory>());
  registry.register_sync("Flatfield", std::make_unique<holotask::syncs::FlatfieldFactory>());
  registry.register_sync("PctClip", std::make_unique<holotask::syncs::PctClipFactory>());
  registry.register_async("BatchQueue", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  registry.register_async("DualReaderBatchQueue",
                          std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
  registry.register_async("SlidingAverage",
                          std::make_unique<holotask::asyncs::SlidingAverageFactory>());
  return registry;
}

GraphSpec reference_spec(bool full) {
  std::ifstream  file(HOLOFLOW_REFERENCE_SPEC);
  nlohmann::json json;
  file >> json;
  if (!full) {
    auto &nodes                                                     = json["nodes"];
    nodes["source_0"]["params"]["height"]                           = 8;
    nodes["source_0"]["params"]["width"]                            = 16;
    nodes["reshape_4"]["params"]["shape"]                           = {1, 32, 8, 16};
    nodes["batch_queue_2"]["params"]["target_capacity"]             = 64;
    nodes["batch_queue_5"]["params"]["target_capacity"]             = 8;
    nodes["dual_reader_batch_queue_8"]["params"]["target_capacity"] = 2;
    nodes["dual_reader_batch_queue_8"]["params"]["window_size"]     = 3;
    nodes["slide_avg_14"]["params"]["target_capacity"]              = 2;
    nodes["slide_avg_14"]["params"]["window_size"]                  = 3;
    nodes["batch_queue_17"]["params"]["target_capacity"]            = 3;
  }
  return holoflow::core::from_json(json);
}

std::vector<std::vector<unsigned char>> run_reference(bool full, size_t limit) {
  auto             state    = std::make_shared<ReferenceState>();
  auto             registry = reference_registry(state);
  Compiler::Config config;
  config.max_section_cuda_graphs = limit;
  config.enable_profiling        = false;
  config.verbose_tracing         = false;
  config.dump_dot_on_failure     = false;
  size_t free_before, free_after, total;
  CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
  const auto   started = std::chrono::steady_clock::now();
  auto         out     = Compiler(registry, config).compile(reference_spec(full));
  const double compile_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
  Scheduler  scheduler(out->graph, out->sections, out->resources);
  const auto preparing = std::chrono::steady_clock::now();
  scheduler.start();
  const double start_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preparing)
          .count();
  CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
  while (!scheduler.stop_requested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  scheduler.request_stop();
  scheduler.wait();
  size_t              enabled = 0;
  std::vector<size_t> products;
  for (const auto &sec : out->sections) {
    const auto &graphs = *out->resources.section_cuda_graphs.at(sec.id);
    if (graphs.enabled) {
      ++enabled;
      products.push_back(graphs.executables.size());
    } else if (limit && !sec.sync_topo.empty() &&
               out->graph[sec.sync_topo.front()].spec.kind != "ReferenceSource" &&
               out->graph[sec.sync_topo.front()].spec.kind != "ReferenceSink") {
      ADD_FAILURE() << sec.name << ": " << graphs.fallback_reason;
    }
  }
  EXPECT_EQ(enabled, limit ? 4U : 0U);
  if (full && limit) {
    std::sort(products.begin(), products.end());
    EXPECT_EQ(products, (std::vector<size_t>{21, 264, 504, 792}));
  }

  EXPECT_EQ(state->frames.size(), state->target_frames);
  if (state->arrival.size() > 16) {
    const double seconds =
        std::chrono::duration<double>(state->arrival.back() - state->arrival[16]).count();
    std::cout << "Reference full=" << full << " cap=" << limit << " compile_ms=" << compile_ms
              << " start_ms=" << start_ms
              << " allocated_MiB=" << (double(free_before) - double(free_after)) / (1024 * 1024)
              << " steady_frames_per_second=" << double(state->arrival.size() - 17) / seconds
              << '\n';
  }
  return state->frames;
}

void compare_reference(bool full) {
  const auto ordinary = run_reference(full, 0);
  const auto graphs   = run_reference(full, full ? 4096 : 128);
  ASSERT_EQ(ordinary.size(), graphs.size());
  for (size_t i = 0; i < ordinary.size(); ++i) {
    ASSERT_EQ(ordinary[i].size(), graphs[i].size());
    int maximum_error = 0;
    for (size_t j = 0; j < ordinary[i].size(); ++j)
      maximum_error = std::max(maximum_error, std::abs(int(ordinary[i][j]) - int(graphs[i][j])));
    EXPECT_LE(maximum_error, 2) << "frame " << i;
  }
}

TEST(SectionGraphReference, SmallPipelineMatchesOrdinaryExecution) { compare_reference(false); }
TEST(SectionGraphReference, PerformanceFullReference) { compare_reference(true); }

} // namespace
