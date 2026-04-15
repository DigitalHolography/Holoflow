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

#include "holonp/conj.hh"

#include <cstdint>
#include <stdexcept>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ConjSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, ConjSettings &) {}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Conj: " + msg);
  }
}

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

template <typename T>
__global__ void conj_identity_kernel(const T *__restrict__ in, T *__restrict__ out,
                                     std::int64_t n) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

__global__ void conj_cf32_kernel(const cuFloatComplex *__restrict__ in,
                                 cuFloatComplex *__restrict__ out, std::int64_t n) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = cuConjf(in[idx]);
  }
}

// -------------------------------------------------------------------------------------------------
// Conj task implementation
// -------------------------------------------------------------------------------------------------

class Conj : public holoflow::core::ISyncTask {
public:
  explicit Conj(ConjSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  void update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ConjSettings settings_;
  cudaStream_t stream_;
};

} // namespace

holoflow::core::OpResult Conj::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;
  const auto  n     = static_cast<std::int64_t>(idesc.num_elements());

  if (n == 0) {
    return holoflow::core::OpResult::Ok;
  }

  constexpr int block = 256;
  const int     grid  = static_cast<int>((n + block - 1) / block);

  switch (idesc.dtype) {
  case holoflow::core::DType::U8: {
    auto *in  = reinterpret_cast<const std::uint8_t *>(idata);
    auto *out = reinterpret_cast<std::uint8_t *>(odata);
    conj_identity_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::U16: {
    auto *in  = reinterpret_cast<const std::uint16_t *>(idata);
    auto *out = reinterpret_cast<std::uint16_t *>(odata);
    conj_identity_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::F32: {
    auto *in  = reinterpret_cast<const float *>(idata);
    auto *out = reinterpret_cast<float *>(odata);
    conj_identity_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::CF32: {
    auto *in  = reinterpret_cast<const cuFloatComplex *>(idata);
    auto *out = reinterpret_cast<cuFloatComplex *>(odata);
    conj_cf32_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  default:
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// ConjFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult ConjFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  (void)jsettings;

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");
  check(is_contiguous(idesc), "input tensor must be contiguous");

  holoflow::core::TDesc odesc = idesc;

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
ConjFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);

  const auto settings = jsettings.get<ConjSettings>();
  return std::make_unique<Conj>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ConjFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_conj = dynamic_cast<Conj *>(old_task.get());
  if (old_conj == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  old_conj->update_stream(ctx.stream);
  return old_task;
}

} // namespace holonp
