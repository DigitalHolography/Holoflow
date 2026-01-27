// Copyright 2025 Digital Holography Foundation
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

#include "holotask/syncs/transpose.hh"

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <map>
#include <spdlog/fmt/ranges.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const TransposeSettings &s) { j = nlohmann::json{}; }

void from_json(const nlohmann::json &j, TransposeSettings &s) {
  // No settings to parse
}

__global__ void kernel_transpose_2d(holoflow::core::TView Source, holoflow::core::TView Destination,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Transpose: (y, x) -> (x, y)
  Destination.data[x * height + y] = Source.data[y * width + x];
}

__global__ void kernel_transpose_3d(holoflow::core::TView Source, holoflow::core::TView Destination,
                                    int depth, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  if (x >= width || y >= height || z >= depth)
    return;

  // For 3D with depth=1: (z, y, x) -> (z, x, y)
  // Since z dimension is 1, we're just transposing the last two dimensions
  Destination.data[z * (width * height) + x * height + y] =
      Source.data[z * (height * width) + y * width + x];
}

void transpose_2d(cudaStream_t stream_, holoflow::core::TView input, holoflow::core::TView output,
                  int height, int width) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_transpose_2d<<<grid, block, 0, stream_>>>(input, output, width, height);
}

void transpose_3d(cudaStream_t stream_, holoflow::core::TView input, holoflow::core::TView output,
                  int depth, int height, int width) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, depth);

  kernel_transpose_3d<<<grid, block, 0, stream_>>>(input, output, depth, width, height);
}

Transpose::Transpose(const TransposeSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Transpose::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
    return holoflow::core::OpResult::Cancelled;
  }

  if (ctx.inputs.empty() || ctx.outputs.empty()) {
    return holoflow::core::OpResult::NotReady;
  }

  holoflow::core::TView &input  = ctx.inputs[0];
  holoflow::core::TView &output = ctx.outputs[0];


  int depth  = input.desc.shape[0];
  int height = input.desc.shape[1];
  int width  = input.desc.shape[2];

  CUDA_CHECK(cudaGetLastError());

  if (depth == 1) // 2d transpose
  {
    //logger()->debug("[Transpose::execute] 2d transpose");
    transpose_2d(stream_, input, output, height, width);
  } else // 3d transpose (with depth=1)
  {
    //logger()->debug("[Transpose::execute] 3d transpose with depth={}", depth);
    transpose_3d(stream_, input, output, depth, height, width);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
TransposeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[TransposeFactory::infer] error: {}", msg);
      throw std::invalid_argument("TransposeFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<TransposeSettings>();

  check(input_descs.size() == 1, "Expected exactly one input tensor");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input tensor must be in device memory");
  check(idesc.rank() == 3, "Input tensor must be 3D (including batch dimension of 1)");

  auto odesc = idesc;

  // Swap the last two dimensions: (1, y, x) -> (1, x, y)
  int height = idesc.shape[1];
  int width  = idesc.shape[2];

  odesc.shape[1] = width;
  odesc.shape[2] = height;

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
TransposeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer_result = infer(input_descs, jsettings);
  auto settings     = jsettings.get<TransposeSettings>();

  auto *task = new Transpose(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs