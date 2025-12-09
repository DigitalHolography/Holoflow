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

#include "holotask/syncs/reshape.hh"

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(json &j, const ReshapeSettings &s) { j = json{{"shape", s.shape}}; }

void from_json(const json &j, ReshapeSettings &s) { j.at("shape").get_to(s.shape); }

Reshape::Reshape(ReshapeSettings settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Reshape::execute(holoflow::core::SyncCtx &ctx) {
  auto &iview = ctx.inputs[0];
  auto &oview = ctx.outputs[0];

  CUDA_CHECK(cudaMemcpyAsync(oview.data, iview.data, oview.desc.num_bytes(),
                             cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ReshapeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[ReshapeFactory::infer] error: {}", msg);
      throw std::invalid_argument("ReshapeFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<ReshapeSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];

  auto has_zero = std::ranges::any_of(settings.shape, [](std::size_t dim) { return dim == 0; });
  check(!has_zero, "new shape should not have zeros");

  auto nb_elements = std::accumulate(settings.shape.begin(), settings.shape.end(), 1ULL,
                                     std::multiplies<std::size_t>());

  check(idesc.num_elements() == nb_elements,
        "new shape does not have same number of elements as input");

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
ReshapeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  infer(input_descs, jsettings);
  auto settings = jsettings.get<ReshapeSettings>();

  auto *sync = new Reshape(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(sync);
}

} // namespace holotask::syncs