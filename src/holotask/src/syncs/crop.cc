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

#include "holotask/syncs/crop.hh"

#include "logger.hh"
#include <cuda_runtime.h>

namespace holotask::syncs {

void to_json(nlohmann::json &j, const CropSettings &settings) {
  j = nlohmann::json{{"origin", settings.origin}, {"shape", settings.shape}};
}

void from_json(const nlohmann::json &j, CropSettings &settings) {
  j.at("origin").get_to(settings.origin);
  j.at("shape").get_to(settings.shape);
}

Crop::Crop(const CropSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Crop::execute(holoflow::core::SyncCtx &ctx) {

  holoflow::core::TView &input  = ctx.inputs[0];
  holoflow::core::TView &output = ctx.outputs[0];

  const std::size_t input_depth   = input.desc.shape[0];
  const std::size_t input_height  = input.desc.shape[1];
  const std::size_t input_width   = input.desc.shape[2];
  const std::size_t output_depth  = output.desc.shape[0];
  const std::size_t output_height = output.desc.shape[1];
  const std::size_t output_width  = output.desc.shape[2];

  const std::size_t origin_z = settings_.origin[0];
  const std::size_t origin_y = settings_.origin[1];
  const std::size_t origin_x = settings_.origin[2];

  logger()->trace("[Crop] 3D crop: input shape = [{}, {}, {}], output shape = [{}, {}, {}], origin "
                  "= [{}, {}, {}]",
                  input_depth, input_height, input_width, output_depth, output_height, output_width,
                  origin_z, origin_y, origin_x);

  if (origin_z + output_depth > input_depth || origin_y + output_height > input_height ||
      origin_x + output_width > input_width) {
    logger()->error("[Crop] Crop region exceeds input bounds");
    return holoflow::core::OpResult::Cancelled;
  }

  cudaMemcpy3DParms params = {0};

  params.srcPtr = make_cudaPitchedPtr((void *)input.data, input_width * sizeof(float), input_width,
                                      input_height);

  params.dstPtr = make_cudaPitchedPtr((void *)output.data, output_width * sizeof(float),
                                      output_width, output_height);

  params.srcPos = make_cudaPos(origin_x, origin_y, origin_z);

  params.dstPos = make_cudaPos(0, 0, 0);

  params.extent = make_cudaExtent(output_width * sizeof(float), output_height, output_depth);

  params.kind = cudaMemcpyDeviceToDevice;

  CUDA_CHECK(cudaMemcpy3DAsync(&params, stream_));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult CropFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {

  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[CropFactory::infer] error: {}", msg);
      throw std::invalid_argument("CropFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<CropSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  check(input_descs[0].dtype == holoflow::core::DType::F32, "only Float32 data type is supported");
  const auto &idesc = input_descs[0];

  logger()->debug("[CropFactory::infer] input shape z,y,x: {}, {}, {} | output shape: {}, {}, {}", idesc.shape[0],
                  idesc.shape[1], idesc.shape[2], settings.shape[0], settings.shape[1],
                  settings.shape[2]);

  for (size_t i = 0; i < idesc.rank(); ++i) {
    check(settings.origin[i] + settings.shape[i] <= idesc.shape[i],
          "Crop region exceeds input bounds in dimension " + std::to_string(i));
  }

  auto odesc = idesc;
  odesc.shape.assign(settings.shape.begin(), settings.shape.end());

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
CropFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {

  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<CropSettings>();

  auto *task = new Crop(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs