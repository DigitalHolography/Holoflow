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

#include "holotask/syncs/log.hh"

#include <cmath>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const LogSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, LogSettings &) {}

namespace {

__global__ void log_kernel(const float *idata, float *odata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = log10f(idata[idx]);
  }
}

} // namespace

Log::Log(const LogSettings &settings, cudaStream_t stream) : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Log::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  int   size          = static_cast<int>(idesc.num_elements());
  int   block_size    = 256;
  int   grid_size     = (size + block_size - 1) / block_size;
  auto *in_ptr        = reinterpret_cast<const float *>(idata);
  auto *out_ptr       = reinterpret_cast<float *>(odata);

  log_kernel<<<grid_size, block_size, 0, stream_>>>(in_ptr, out_ptr, size);
  CUDA_CHECK(cudaGetLastError());

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult LogFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[LogFactory::infer] error: {}", msg);
      throw std::invalid_argument("LogFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<LogSettings>();
  (void)settings;

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
LogFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<LogSettings>();
  (void)infer;

  auto *task = new Log(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs
