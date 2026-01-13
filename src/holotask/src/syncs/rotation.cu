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

__global__ void kernel_2d_270(holoflow::core::TView Source, holoflow::core::TView Destination,
                              int sizeX, int sizeY, float deg) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sizeX || y >= sizeY)
    return;

  int newX = y;
  int newY = sizeY - 1 - x;

  Destination.data[newY * sizeY + newX] = Source.data[y * sizeX + x];
}
__global__ void kernel_2d_180(holoflow::core::TView Source, holoflow::core::TView Destination,
                              int sizeX, int sizeY, float deg) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sizeX || y >= sizeY)
    return;

  int newX = sizeY - 1 - x;
  int newY = sizeX - 1 - y;

  Destination.data[newY * sizeY + newX] = Source.data[y * sizeX + x];
}
__global__ void kernel_2d_90(holoflow::core::TView Source, holoflow::core::TView Destination,
                             int sizeX, int sizeY, float deg) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sizeX || y >= sizeY)
    return;

  int newX = sizeX - 1 - y;
  int newY = x;

  Destination.data[newY * sizeY + newX] = Source.data[y * sizeX + x];
}

void rotate_2d(cudaStream_t stream_, holoflow::core::TView input, holoflow::core::TView output,
               int height, int width, float deg) {

  int  angle = static_cast<int>(deg);
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  switch (angle) {
  case 90:
    kernel_2d_90<<<grid, block, 0, stream_>>>(input, output, width, height, deg);
    break;
  case 180:
    kernel_2d_180<<<grid, block, 0, stream_>>>(input, output, width, height, deg);
    break;
  case 270:
    kernel_2d_270<<<grid, block, 0, stream_>>>(input, output, width, height, deg);
    break;
  default:
    logger()->error("[rotate_2d] Invalid angle impossible case");
    break;
  }
}

__global__ void kernel_3d_90_xaxis(holoflow::core::TView Source, holoflow::core::TView Destination,
                                   int sizeZ, int sizeX, int sizeY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  if (x >= sizeX || y >= sizeY || z >= sizeZ)
    return;

  int newX = x;
  int newY = z;
  int newZ = sizeY - 1 - y;

  Destination.data[newZ * (sizeX * sizeZ) + newY * sizeX + newX] =
      Source.data[z * (sizeX * sizeY) + y * sizeX + x];
}

__global__ void kernel_3d_90_yaxis(holoflow::core::TView Source, holoflow::core::TView Destination,
                                   int sizeZ, int sizeX, int sizeY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  if (x >= sizeX || y >= sizeY || z >= sizeZ)
    return;

  int newX = z;
  int newY = y;
  int newZ = sizeX - 1 - x;

  Destination.data[newZ * (sizeZ * sizeY) + newY * sizeZ + newX] =
      Source.data[z * (sizeX * sizeY) + y * sizeX + x];
}

__global__ void kernel_3d_90_zaxis(holoflow::core::TView Source, holoflow::core::TView Destination,
                                   int sizeZ, int sizeX, int sizeY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  if (x >= sizeX || y >= sizeY || z >= sizeZ)
    return;

  int newX = sizeX - 1 - y;
  int newY = x;
  int newZ = z;

  Destination.data[newZ * (sizeX * sizeY) + newY * sizeY + newX] =
      Source.data[z * (sizeX * sizeY) + y * sizeX + x];
}

void rotate_3d(cudaStream_t stream_, holoflow::core::TView input, holoflow::core::TView output,
               int depth, int height, int width, float deg, RotationSettings::Axis axis) {

  int  it = deg / 90;
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, depth);

  switch (axis) {
  case RotationSettings::Axis::Z:
    for (int i = 0; i < it; i++) {
      kernel_3d_90_zaxis<<<grid, block, 0, stream_>>>(input, output, depth, width, height);
    }
    break;
  case RotationSettings::Axis::Y:
    for (int i = 0; i < it; i++) {
      kernel_3d_90_yaxis<<<grid, block, 0, stream_>>>(input, output, depth, width, height);
    }
    break;
  case RotationSettings::Axis::X:
    for (int i = 0; i < it; i++) {
      kernel_3d_90_xaxis<<<grid, block, 0, stream_>>>(input, output, depth, width, height);
    }
    break;
  default:
    logger()->error("[rotate_3d] Invalid axis impossible case");
  }
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

  int depth  = input.desc.shape[0];
  int height = input.desc.shape[1];
  int width  = input.desc.shape[2];

  CUDA_CHECK(cudaGetLastError());

  if (depth == 1) // 2d rotation
  {
    logger()->trace("[Rotation::execute] 2d rotation");
    rotate_2d(stream_, input, output, width, height, settings_.angle);

  } else // 3d rotation
  {
    logger()->trace("[Rotation::execute] 3d rotation");
    rotate_3d(stream_, input, output, depth, width, height, settings_.angle, settings_.axis);
  }
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

  settings.angle = ((settings.angle % 360) + 360) % 360;
  logger()->debug("[RotationFactory::infer] angle: {}", settings.angle);

  check(input_descs.size() == 1, "Expected exactly one input tensor");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input tensor must be in device memory");
  check(settings.angle % 90 == 0, "Rotation angle must be a multiple of 90 degrees");
  check(settings.angle % 360 != 0, "Rotation angle cannot be 0 or multiples of 360 degrees");
  check(settings.axis == RotationSettings::Axis::Z || idesc.rank() == 2,
        "Rotation axis other than Z is only valid for 3D tensors");

  auto odesc  = idesc;
  int  depth  = idesc.shape[0];
  int  height = idesc.shape[1];
  int  width  = idesc.shape[2];

  if (depth == 1) {

    auto temp      = odesc.shape[1];
    odesc.shape[1] = idesc.shape[2];
    odesc.shape[2] = temp;
  } else {
    switch (settings.axis) {
      size_t temp;
    case RotationSettings::Axis::Z:
      temp           = odesc.shape[1];
      odesc.shape[1] = idesc.shape[2];
      odesc.shape[2] = temp;
      break;
    case RotationSettings::Axis::Y:
      temp           = odesc.shape[2];
      odesc.shape[2] = idesc.shape[0];
      odesc.shape[0] = temp;
      break;
    case RotationSettings::Axis::X:
      temp           = odesc.shape[1];
      odesc.shape[1] = idesc.shape[0];
      odesc.shape[0] = temp;
      break;
    default:
      logger()->error("[RotationFactory::infer] Invalid axis impossible case");
    }
  }

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