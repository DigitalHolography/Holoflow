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

#include "holotask/syncs/rotation.hh"

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <map>
#include <spdlog/fmt/ranges.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const RotationSettings &s) {
  j = nlohmann::json{{"angle", s.angle}, {"axis", s.axis}};
}

void from_json(const nlohmann::json &j, RotationSettings &s) {
  j.at("angle").get_to(s.angle);
  j.at("axis").get_to(s.axis);
}

namespace {

__global__ void rotate_kernel(holoflow::core::TView Source, holoflow::core::TView Destination,
                              int sizeX, int sizeY, float deg) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sizeX || y >= sizeY)
    return;

  int newX = sizeX - 1 - y;

  Destination.data[x * sizeY + newX] = Source.data[y * sizeX + x];
}
} // namespace

Rotation::Rotation(const RotationSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Rotation::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
    return holoflow::core::OpResult::Cancelled;
  }

  if (ctx.inputs.empty() || ctx.outputs.empty()) {
    return holoflow::core::OpResult::NotReady;
  }

  holoflow::core::TView &input  = ctx.inputs[0];
  holoflow::core::TView &output = ctx.outputs[0];

  logger()->trace(
      "[Rotation::execute] Rotating tensor of shape {} by {} degrees for a new tensor of shape {}",
      input.desc.shape, settings_.angle, output.desc.shape);

  int height = input.desc.shape[1];
  int width  = input.desc.shape[2];

  CUDA_CHECK(cudaGetLastError());

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  rotate_kernel<<<grid, block, 0, stream_>>>(input, output, width, height, settings_.angle);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
RotationFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[RotationFactory::infer] error: {}", msg);
      throw std::invalid_argument("RotationFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<RotationSettings>();

  settings.angle = settings.angle % 360;
  logger()->debug("[RotationFactory::infer] angle: {}", settings.angle);

  check(input_descs.size() == 1, "Expected exactly one input tensor");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input tensor must be in device memory");
  check(settings.angle % 90 == 0, "Rotation angle must be a multiple of 90 degrees");
  check(settings.angle % 360 != 0, "Rotation angle cannot be 0 or multiples of 360 degrees");
  check(settings.axis == RotationSettings::Axis::Z || idesc.rank() == 2,
        "Rotation axis other than Z is only valid for 3D tensors");

  auto odesc = idesc;

  auto temp      = odesc.shape[1];
  odesc.shape[1] = idesc.shape[2];
  odesc.shape[2] = temp;

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
RotationFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer_result = infer(input_descs, jsettings);
  auto settings     = jsettings.get<RotationSettings>();

  auto *task = new Rotation(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs