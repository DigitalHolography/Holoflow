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

#include "holotask/syncs/cuda_stream_synchronize.hh"

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const CudaStreamSynchronizeSettings &) { j = nlohmann::json{}; }

void from_json(const nlohmann::json &, CudaStreamSynchronizeSettings &) {}

CudaStreamSynchronize::CudaStreamSynchronize(cudaStream_t stream) : stream_(stream) {}

holoflow::core::OpResult CudaStreamSynchronize::execute(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
CudaStreamSynchronizeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[CudaStreamSynchronizeFactory::infer] error: {}", msg);
      throw std::invalid_argument("CudaStreamSynchronizeFactory inference error: " + msg);
    }
  };

  (void)jsettings;
  check(input_descs.size() == 1, "CudaStreamSynchronize task must have exactly one input");

  return holoflow::core::InferResult{
      .input_descs   = {input_descs[0]},
      .output_descs  = {input_descs[0]},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CudaStreamSynchronizeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                     const nlohmann::json                  &jsettings,
                                     const holoflow::core::SyncCreateCtx   &ctx) const {
  this->infer(input_descs, jsettings);
  return std::unique_ptr<holoflow::core::ISyncTask>(new CudaStreamSynchronize(ctx.stream));
}

} // namespace holotask::syncs
