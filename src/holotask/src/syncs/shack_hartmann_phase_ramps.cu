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

#include "holotask/syncs/shack_hartmann_phase_ramps.hh"

#include <math_constants.h>

#include <string>

#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ShackHartmannPhaseRampsSettings &s) {
  j = nlohmann::json{
      {"sub_w", s.sub_w},
      {"sub_h", s.sub_h},
      {"dx", s.dx},
      {"dy", s.dy},
      {"nx_subabs", s.nx_subabs},
      {"ny_subabs", s.ny_subabs},
      {"R", s.R},
      {"lambda", s.lambda},
  };
}

void from_json(const nlohmann::json &j, ShackHartmannPhaseRampsSettings &s) {
  j.at("sub_w").get_to(s.sub_w);
  j.at("sub_h").get_to(s.sub_h);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("nx_subabs").get_to(s.nx_subabs);
  j.at("ny_subabs").get_to(s.ny_subabs);
  if (j.contains("R")) {
    j.at("R").get_to(s.R);
  } else {
    j.at("radius_of_curvature").get_to(s.R);
  }
  j.at("lambda").get_to(s.lambda);
}

namespace {

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ShackHartmannPhaseRampsFactory::infer] error: {}", msg);
    throw std::invalid_argument("ShackHartmannPhaseRampsFactory inference error: " + msg);
  }
}

__global__ void fill_phase_ramps_kernel(cuFloatComplex *ramps, size_t sub_w, size_t sub_h,
                                        float dx, float dy, size_t nx_subabs, size_t ny_subabs,
                                        float radius, float lambda) {
  auto idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto tile_size   = static_cast<unsigned long long>(sub_w) * sub_h;
  auto subap_count = static_cast<unsigned long long>(nx_subabs) * ny_subabs;
  auto total       = subap_count * tile_size;
  if (idx >= total)
    return;

  auto subap_idx = idx / tile_size;
  auto local_idx = idx % tile_size;

  size_t gy = static_cast<size_t>(subap_idx / nx_subabs);
  size_t gx = static_cast<size_t>(subap_idx % nx_subabs);
  size_t j  = static_cast<size_t>(local_idx / sub_w);
  size_t i  = static_cast<size_t>(local_idx % sub_w);

  const float k       = 2.0f * CUDART_PI_F / lambda;
  const float pitch_x = static_cast<float>(sub_w) * dx;
  const float pitch_y = static_cast<float>(sub_h) * dy;

  const float center_sub_x = 0.5f * (static_cast<float>(nx_subabs) - 1.0f);
  const float center_sub_y = 0.5f * (static_cast<float>(ny_subabs) - 1.0f);
  const float center_x     = 0.5f * (static_cast<float>(sub_w) - 1.0f);
  const float center_y     = 0.5f * (static_cast<float>(sub_h) - 1.0f);

  const float Xc      = (static_cast<float>(gx) - center_sub_x) * pitch_x;
  const float Yc      = (static_cast<float>(gy) - center_sub_y) * pitch_y;
  const float theta_x = Xc / radius;
  const float theta_y = Yc / radius;
  const float x       = (static_cast<float>(i) - center_x) * dx;
  const float y       = (static_cast<float>(j) - center_y) * dy;

  const float phase = k * (theta_x * x + theta_y * y);
  float       sin_phase;
  float       cos_phase;
  sincosf(phase, &sin_phase, &cos_phase);
  ramps[idx] = make_cuFloatComplex(cos_phase, sin_phase);
}

DevPtr<cuFloatComplex> make_phase_ramps(const ShackHartmannPhaseRampsSettings &settings,
                                        cudaStream_t stream) {
  const size_t count = settings.nx_subabs * settings.ny_subabs * settings.sub_h * settings.sub_w;
  auto         ramps = curaii::make_unique_device_ptr<cuFloatComplex>(count, stream);

  const int block_size = 256;
  const int grid_size =
      static_cast<int>((static_cast<unsigned long long>(count) + block_size - 1) / block_size);

  fill_phase_ramps_kernel<<<grid_size, block_size, 0, stream>>>(
      ramps.get(), settings.sub_w, settings.sub_h, settings.dx, settings.dy, settings.nx_subabs,
      settings.ny_subabs, settings.R, settings.lambda);
  CUDA_CHECK(cudaGetLastError());
  return ramps;
}

} // namespace

ShackHartmannPhaseRamps::ShackHartmannPhaseRamps(
    const ShackHartmannPhaseRampsSettings &settings, DevPtr<cuFloatComplex> &&d_ramps, size_t count,
    cudaStream_t stream)
    : settings_(settings), d_ramps_(std::move(d_ramps)), count_(count), stream_(stream) {}

holoflow::core::OpResult ShackHartmannPhaseRamps::execute(holoflow::core::SyncCtx &ctx) {
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
  CUDA_CHECK(cudaMemcpyAsync(odata, d_ramps_.get(), count_ * sizeof(cuFloatComplex),
                             cudaMemcpyDeviceToDevice, stream_));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ShackHartmannPhaseRampsFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ShackHartmannPhaseRampsSettings>();

  check(input_descs.empty(), "expected zero input tensors");
  check(settings.sub_w > 0, "sub_w must be positive");
  check(settings.sub_h > 0, "sub_h must be positive");
  check(settings.nx_subabs > 0, "nx_subabs must be positive");
  check(settings.ny_subabs > 0, "ny_subabs must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.lambda > 0.0f, "lambda must be positive");
  check(settings.R != 0.0f, "R must be non-zero");

  holoflow::core::TDesc odesc({settings.ny_subabs, settings.nx_subabs, settings.sub_h,
                               settings.sub_w},
                              holoflow::core::DType::CF32, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ShackHartmannPhaseRampsFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                       const nlohmann::json                  &jsettings,
                                       const holoflow::core::SyncCreateCtx   &ctx) const {
  infer(input_descs, jsettings);
  const auto settings = jsettings.get<ShackHartmannPhaseRampsSettings>();
  auto       d_ramps  = make_phase_ramps(settings, ctx.stream);
  const size_t count =
      settings.nx_subabs * settings.ny_subabs * settings.sub_h * settings.sub_w;

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new ShackHartmannPhaseRamps(settings, std::move(d_ramps), count, ctx.stream));
}

std::unique_ptr<holoflow::core::ISyncTask>
ShackHartmannPhaseRampsFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                       std::span<const holoflow::core::TDesc>     input_descs,
                                       const nlohmann::json                       &jsettings,
                                       const holoflow::core::SyncCreateCtx        &ctx) const {
  infer(input_descs, jsettings);

  auto *old_ramps = dynamic_cast<ShackHartmannPhaseRamps *>(old_task.get());
  if (old_ramps != nullptr) {
    const auto settings = jsettings.get<ShackHartmannPhaseRampsSettings>();
    if (settings == old_ramps->get_settings()) {
      old_ramps->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
