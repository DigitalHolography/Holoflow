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

#include "holotask/syncs/correct_phase.hh"

#include <cuComplex.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const CorrectPhaseSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, CorrectPhaseSettings &) {}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[CorrectPhaseFactory::infer] error: {}", msg);
    throw std::invalid_argument("CorrectPhaseFactory inference error: " + msg);
  }
}

size_t leading_image_count(const holoflow::core::TDesc &desc) {
  size_t count = 1;
  for (size_t i = 0; i + 2 < desc.shape.size(); ++i) {
    count *= desc.shape[i];
  }
  return count;
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

// -------------------------------------------------------------------------------------------------
// Device kernels
// -------------------------------------------------------------------------------------------------

__global__ void correct_phase_f32_to_cf32_kernel(const float *input, const float *phase_mask,
                                                 cuFloatComplex *output, int total_size,
                                                 int plane_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    const float intensity = input[idx];
    const float phase     = phase_mask[idx % plane_size];

    float sin_phase;
    float cos_phase;
    sincosf(phase, &sin_phase, &cos_phase);

    // E_out = I * exp(-i * phi) = I * cos(phi) - i * I * sin(phi)
    output[idx] = make_cuFloatComplex(intensity * cos_phase, -intensity * sin_phase);
  }
}

__global__ void correct_phase_cf32_kernel(const cuFloatComplex *input, const float *phase_mask,
                                          cuFloatComplex *output, int total_size, int plane_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_size) {
    const float phase = phase_mask[idx % plane_size];
    float       sin_phase;
    float       cos_phase;
    sincosf(phase, &sin_phase, &cos_phase);

    const cuFloatComplex value = input[idx];
    output[idx]                = make_cuFloatComplex(value.x * cos_phase + value.y * sin_phase,
                                                     value.y * cos_phase - value.x * sin_phase);
  }
}

// -------------------------------------------------------------------------------------------------
// CorrectPhase task implementation
// -------------------------------------------------------------------------------------------------

class CorrectPhase : public holoflow::core::ISyncTask {
public:
  CorrectPhase(CorrectPhaseSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto &idesc      = ctx.inputs[0].desc;
    const int   total_size = static_cast<int>(idesc.num_elements());
    const int   plane_size =
        static_cast<int>(idesc.shape[idesc.rank() - 2] * idesc.shape[idesc.rank() - 1]);

    constexpr int block_size = 256;
    const int     grid_size  = (total_size + block_size - 1) / block_size;

    auto *phase_mask = reinterpret_cast<const float *>(ctx.inputs[1].data());
    auto *output     = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());

    if (idesc.dtype == holoflow::core::DType::F32) {
      auto *input = reinterpret_cast<const float *>(ctx.inputs[0].data());
      correct_phase_f32_to_cf32_kernel<<<grid_size, block_size, 0, stream_>>>(
          input, phase_mask, output, total_size, plane_size);
    } else {
      auto *input = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data());
      correct_phase_cf32_kernel<<<grid_size, block_size, 0, stream_>>>(input, phase_mask, output,
                                                                       total_size, plane_size);
    }

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) { stream_ = stream; }

  const CorrectPhaseSettings &settings() const { return settings_; }

private:
  CorrectPhaseSettings settings_;
  cudaStream_t         stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// CorrectPhaseFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
CorrectPhaseFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings) const {
  (void)jsettings.get<CorrectPhaseSettings>();

  check(input_descs.size() == 2, "expected exactly two inputs");

  const auto &idesc = input_descs[0];
  const auto &mdesc = input_descs[1];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(mdesc.mem_loc == holoflow::core::MemLoc::Device, "phase mask must be in device memory");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "input dtype must be F32 or CF32");
  check(mdesc.dtype == holoflow::core::DType::F32, "phase mask dtype must be F32");
  check(idesc.rank() >= 2, "input rank must be at least 2");
  check(mdesc.rank() >= 2, "phase mask rank must be at least 2");
  check(idesc.shape[idesc.rank() - 2] == mdesc.shape[mdesc.rank() - 2],
        "phase mask height must match input height");
  check(idesc.shape[idesc.rank() - 1] == mdesc.shape[mdesc.rank() - 1],
        "phase mask width must match input width");
  check(leading_image_count(mdesc) == 1, "phase mask must contain exactly one image");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(is_c_contiguous(mdesc), "phase mask must be C-contiguous");

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::CF32, idesc.mem_loc);

  return holoflow::core::InferResult{
      .input_descs   = {idesc, mdesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CorrectPhaseFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings,
                            const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  auto settings = jsettings.get<CorrectPhaseSettings>();

  return std::make_unique<CorrectPhase>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
CorrectPhaseFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                            std::span<const holoflow::core::TDesc>     input_descs,
                            const nlohmann::json                      &jsettings,
                            const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_correct_phase = dynamic_cast<CorrectPhase *>(old_task.get());
  if (old_correct_phase == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<CorrectPhaseSettings>();
  if (settings == old_correct_phase->settings()) {
    old_correct_phase->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
