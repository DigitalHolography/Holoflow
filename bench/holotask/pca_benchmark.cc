// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/pca.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::ISyncTask;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::SyncCtx;
using holoflow::core::TDesc;
using holoflow::core::Tensor;
using holoflow::core::TView;

constexpr size_t kFeatures = 32;
constexpr size_t kHeight   = 320;
constexpr size_t kWidth    = 512;
constexpr int    kBegin    = 24;
constexpr int    kEnd      = 32;

struct TaskInvocation {
  std::unique_ptr<ISyncTask> task;
  std::vector<TView>         inputs;
  std::vector<TView>         outputs;
  std::atomic<bool>          cancelled{false};

  void execute() {
    SyncCtx ctx{
        .inputs       = inputs,
        .outputs      = outputs,
        .cancelled    = &cancelled,
        .event_writer = nullptr,
        .event_reader = nullptr,
    };
    if (task->execute(ctx) != OpResult::Ok) {
      throw std::runtime_error("PCA benchmark task did not complete");
    }
  }
};

double elapsed_seconds(cudaEvent_t start, cudaEvent_t stop) {
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  return static_cast<double>(milliseconds) / 1000.0;
}

void fill_input(Tensor &input, cudaStream_t stream) {
  std::vector<std::uint8_t> host(input.desc().num_elements());
  for (size_t index = 0; index < host.size(); ++index) {
    host[index] = static_cast<std::uint8_t>((index * 17 + index / 31 + 3) % 251);
  }
  CUDA_CHECK(
      cudaMemcpyAsync(input.data(), host.data(), host.size(), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void fused_u8_pca(benchmark::State &state) {
  const auto         batches = static_cast<size_t>(state.range(0));
  curaii::CudaStream stream;
  const TDesc        input_desc({batches, kFeatures, kHeight, kWidth}, DType::U8, MemLoc::Device);
  const holotask::syncs::PcaSettings settings{.begin = kBegin, .end = kEnd};
  holotask::syncs::PcaFactory        factory;
  const auto inferred = factory.infer(std::vector<TDesc>{input_desc}, settings);

  Tensor input(input_desc);
  Tensor output(inferred.output_descs[0]);
  fill_input(input, stream.get());

  TaskInvocation invocation{
      .task    = factory.create(std::vector<TDesc>{input_desc}, settings, {stream.get()}),
      .inputs  = {input.view()},
      .outputs = {output.view()},
  };
  invocation.task->bind_logger(spdlog::default_logger());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  invocation.execute();
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  for (auto _ : state) {
    (void)_;
    CUDA_CHECK(cudaEventRecord(start, stream.get()));
    invocation.execute();
    CUDA_CHECK(cudaEventRecord(stop, stream.get()));
    state.SetIterationTime(elapsed_seconds(start, stop));
  }

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(input_desc.num_bytes()));
}

void conversion_plus_f32_pca(benchmark::State &state) {
  const auto         batches = static_cast<size_t>(state.range(0));
  curaii::CudaStream stream;
  const TDesc        input_desc({batches, kFeatures, kHeight, kWidth}, DType::U8, MemLoc::Device);
  const holotask::syncs::ConversionSettings conversion_settings{
      .target   = holotask::syncs::ConversionSettings::Target::F32,
      .strategy = holotask::syncs::ConversionSettings::Strategy::Real,
  };
  const holotask::syncs::PcaSettings pca_settings{.begin = kBegin, .end = kEnd};
  holotask::syncs::ConversionFactory conversion_factory;
  holotask::syncs::PcaFactory        pca_factory;
  const auto                         conversion_inferred =
      conversion_factory.infer(std::vector<TDesc>{input_desc}, conversion_settings);
  const auto f32_desc     = conversion_inferred.output_descs[0];
  const auto pca_inferred = pca_factory.infer(std::vector<TDesc>{f32_desc}, pca_settings);

  Tensor input(input_desc);
  Tensor converted(f32_desc);
  Tensor output(pca_inferred.output_descs[0]);
  fill_input(input, stream.get());

  TaskInvocation conversion{
      .task    = conversion_factory.create(std::vector<TDesc>{input_desc}, conversion_settings,
                                           {stream.get()}),
      .inputs  = {input.view()},
      .outputs = {converted.view()},
  };
  TaskInvocation pca{
      .task    = pca_factory.create(std::vector<TDesc>{f32_desc}, pca_settings, {stream.get()}),
      .inputs  = {converted.view()},
      .outputs = {output.view()},
  };
  conversion.task->bind_logger(spdlog::default_logger());
  pca.task->bind_logger(spdlog::default_logger());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  conversion.execute();
  pca.execute();
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  for (auto _ : state) {
    (void)_;
    CUDA_CHECK(cudaEventRecord(start, stream.get()));
    conversion.execute();
    pca.execute();
    CUDA_CHECK(cudaEventRecord(stop, stream.get()));
    state.SetIterationTime(elapsed_seconds(start, stop));
  }

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(input_desc.num_bytes()));
}

BENCHMARK(fused_u8_pca)->Arg(1)->Arg(8)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(conversion_plus_f32_pca)->Arg(1)->Arg(8)->UseManualTime()->Unit(benchmark::kMillisecond);

} // namespace
