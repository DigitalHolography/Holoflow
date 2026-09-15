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

#include "holonp/exp.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ExpSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, ExpSettings &) {}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Exp: " + msg);
  }
}

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

__global__ void exp_f32_kernel(const float *__restrict__ in, float *__restrict__ out,
                               std::int64_t total_out) {
  const auto idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < total_out) {
    out[idx] = expf(in[idx]);
  }
}

__global__ void exp_cf32_kernel(const cuFloatComplex *__restrict__ in,
                                cuFloatComplex *__restrict__ out, std::int64_t total_out) {
  const auto idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < total_out) {
    const auto  value     = in[idx];
    const float magnitude = expf(value.x);
    out[idx] = make_cuFloatComplex(magnitude * cosf(value.y), magnitude * sinf(value.y));
  }
}

// -------------------------------------------------------------------------------------------------
// Exp task implementation
// -------------------------------------------------------------------------------------------------

class Exp : public holoflow::core::ISyncTask {
public:
  explicit Exp(ExpSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  void update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ExpSettings  settings_;
  cudaStream_t stream_;
};

holoflow::core::OpResult Exp::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata     = ctx.inputs[0].data();
  auto       *odata     = ctx.outputs[0].data();
  const auto &idesc     = ctx.inputs[0].desc;
  const auto  total_out = static_cast<std::int64_t>(idesc.num_elements());

  constexpr int block = 256;
  const int     grid  = static_cast<int>((total_out + block - 1) / block);

  switch (idesc.dtype) {
  case holoflow::core::DType::F32:
    exp_f32_kernel<<<grid, block, 0, stream_>>>(reinterpret_cast<const float *>(idata),
                                                reinterpret_cast<float *>(odata), total_out);
    break;
  case holoflow::core::DType::CF32:
    exp_cf32_kernel<<<grid, block, 0, stream_>>>(reinterpret_cast<const cuFloatComplex *>(idata),
                                                 reinterpret_cast<cuFloatComplex *>(odata),
                                                 total_out);
    break;
  default:
    logger()->error("[Exp::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// ExpFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult ExpFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  (void)jsettings;

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");
  check(idesc.num_elements() > 0, "input tensor has zero elements");
  check(is_contiguous(idesc), "input tensor must be contiguous");

  holoflow::core::TDesc odesc(idesc.shape, idesc.dtype, holoflow::core::MemLoc::Device);

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
ExpFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);

  const auto settings = jsettings.get<ExpSettings>();
  return std::make_unique<Exp>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ExpFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_exp = dynamic_cast<Exp *>(old_task.get());
  if (old_exp == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  old_exp->update_stream(ctx.stream);
  return old_task;
}

} // namespace holonp
