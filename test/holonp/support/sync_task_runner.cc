// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sync_task_runner.hh"

#include <atomic>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "tensor_test_buffer.hh"

namespace holonp_test {

namespace {

RunResult execute_task(holoflow::core::ISyncTask              &task,
                       std::span<const holoflow::core::TDesc>  input_descs,
                       std::span<const std::vector<std::byte>> input_host_data,
                       std::span<const holoflow::core::TDesc> output_descs, cudaStream_t stream) {
  // Upload inputs.
  std::vector<TensorTestBuffer> in_bufs;
  in_bufs.reserve(input_descs.size());
  for (size_t i = 0; i < input_descs.size(); ++i) {
    in_bufs.emplace_back(input_descs[i]);
    in_bufs.back().upload(input_host_data[i]);
  }

  // Allocate output buffers (uninitialized).
  std::vector<TensorTestBuffer> out_bufs;
  out_bufs.reserve(output_descs.size());
  for (const auto &odesc : output_descs) {
    out_bufs.emplace_back(odesc);
  }

  // Assemble views.
  std::vector<holoflow::core::TView> in_views;
  std::vector<holoflow::core::TView> out_views;
  in_views.reserve(in_bufs.size());
  out_views.reserve(out_bufs.size());
  for (auto &buf : in_bufs)
    in_views.push_back(buf.view());
  for (auto &buf : out_bufs)
    out_views.push_back(buf.view());

  // Execute.
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = in_views,
      .outputs      = out_views,
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  task.bind_logger(spdlog::default_logger());
  (void)task.execute(ctx);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Download outputs.
  RunResult result;
  result.output_descs = {output_descs.begin(), output_descs.end()};
  result.output_bytes.reserve(out_bufs.size());
  for (auto &buf : out_bufs) {
    result.output_bytes.push_back(buf.download());
  }
  return result;
}

} // namespace

RunResult run_sync_factory(const holoflow::core::ISyncTaskFactory &factory,
                           std::span<const holoflow::core::TDesc>  input_descs,
                           std::span<const std::vector<std::byte>> input_host_data,
                           const nlohmann::json                   &jsettings) {
  const auto infer = factory.infer(input_descs, jsettings);

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto                                task = factory.create(input_descs, jsettings, create_ctx);

  return execute_task(*task, input_descs, input_host_data, infer.output_descs, stream.get());
}

RunResult run_sync_factory_update(const holoflow::core::ISyncTaskFactory &factory,
                                  std::span<const holoflow::core::TDesc>  input_descs,
                                  std::span<const std::vector<std::byte>> input_host_data,
                                  const nlohmann::json                   &jsettings) {
  const auto infer = factory.infer(input_descs, jsettings);

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  auto                                task = factory.create(input_descs, jsettings, create_ctx);
  task = factory.update(std::move(task), input_descs, jsettings, create_ctx);

  return execute_task(*task, input_descs, input_host_data, infer.output_descs, stream.get());
}

} // namespace holonp_test
