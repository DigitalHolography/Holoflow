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

#include "holotask/syncs/wrap2pi.hh"

#include <math_constants.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const Wrap2PiSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, Wrap2PiSettings &) {}

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[Wrap2PiFactory::infer] error: {}", msg);
    throw std::invalid_argument("Wrap2PiFactory inference error: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

__global__ void wrap2pi_kernel(const float *idata, float *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    constexpr float two_pi = 2.0f * CUDART_PI_F;
    float           val    = fmodf(idata[idx], two_pi);
    if (val < 0.0f) {
      val += two_pi;
    }
    odata[idx] = val;
  }
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Wrap2Pi task implementation
// -------------------------------------------------------------------------------------------------

class Wrap2Pi : public holoflow::core::ISyncTask {
public:
  explicit Wrap2Pi(Wrap2PiSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *idata = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto *odata = reinterpret_cast<float *>(ctx.outputs[0].data());
    int   size  = static_cast<int>(ctx.inputs[0].desc.num_elements());

    constexpr int block_size = 256;
    const int     grid_size  = (size + block_size - 1) / block_size;
    wrap2pi_kernel<<<grid_size, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());

    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) { stream_ = stream; }
  const Wrap2PiSettings &settings() const { return settings_; }

private:
  Wrap2PiSettings settings_;
  cudaStream_t    stream_;
};

// -------------------------------------------------------------------------------------------------
// Wrap2PiFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
Wrap2PiFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  (void)jsettings.get<Wrap2PiSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(is_c_contiguous(idesc), "input must be C-contiguous");

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
Wrap2PiFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<Wrap2PiSettings>();

  return std::make_unique<Wrap2Pi>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
Wrap2PiFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_wrap2pi = dynamic_cast<Wrap2Pi *>(old_task.get());
  if (old_wrap2pi == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<Wrap2PiSettings>();
  if (settings == old_wrap2pi->settings()) {
    old_wrap2pi->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
