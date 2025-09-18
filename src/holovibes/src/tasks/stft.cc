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

#include "tasks/stft.hh"

#include "logger.hh"

namespace holovibes::tasks {

void to_json(nlohmann::json &j, const StftSettings &ss) {
  // No settings for now.
  (void)ss;
  j = nlohmann::json::object();
}

void from_json(const nlohmann::json &j, StftSettings &ss) {
  // No settings for now.
  (void)j;
  (void)ss;
}

Stft::Stft(const StftSettings &settings, curaii::CufftHandle &&plan)
    : settings_(settings), plan_(std::move(plan)) {}

holoflow::core::OpResult Stft::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data);
  auto *odata = reinterpret_cast<float *>(ctx.outputs[0].data);
  CUFFT_CHECK(cufftXtExec(plan_.get(), idata, odata, CUFFT_FORWARD));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult StftFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[StftFactory::infer] error: {}", msg);
      throw std::invalid_argument("StftFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<StftSettings>();
  (void)settings; // No settings for now.

  // Validate input
  check(input_descs.size() == 1, "STFT task must have exactly one input");
  auto idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::CF32, "Input dtype must be CF32");
  check(idesc.rank() == 3, "Input rank must be 3");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input memory location must be Device");

  // Success
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
StftFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<StftSettings>();
  (void)settings; // No settings for now.

  auto idesc = input_descs[0];
  int  B     = static_cast<int>(idesc.shape[0]);
  int  H     = static_cast<int>(idesc.shape[1]);
  int  W     = static_cast<int>(idesc.shape[2]);

  // Create plan
  int           rank          = 1;
  long long int n[1]          = {B};
  long long int inembed[1]    = {B};
  int           istride       = H * W;
  int           idist         = 1;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[1]    = {B};
  int           ostride       = H * W;
  int           odist         = 1;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = H * W;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), rank, n, inembed, istride, idist, inputtype, onembed,
                                  ostride, odist, outputtype, batch, &work_size, executiontype));

  // Success
  auto *task = new Stft(settings, std::move(plan));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks