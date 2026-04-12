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

#include "holotask/syncs/short_time_fresnel_phase_ramps.hh"

#include <math_constants.h>
#include <stdexcept>
#include <string>

#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ShortTimeFresnelPhaseSettings &s) {
  j = nlohmann::json{{"win_w", s.win_w},
                     {"win_h", s.win_h},
                     {"dx", s.dx},
                     {"dy", s.dy},
                     {"nx_win", s.nx_win},
                     {"ny_win", s.ny_win},
                     {"stride_x", s.stride_x},
                     {"stride_y", s.stride_y},
                     {"field_w", s.field_w},
                     {"field_h", s.field_h},
                     {"z", s.z},
                     {"lambda", s.lambda}};
}

void from_json(const nlohmann::json &j, ShortTimeFresnelPhaseSettings &s) {
  j.at("win_w").get_to(s.win_w);
  j.at("win_h").get_to(s.win_h);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("nx_win").get_to(s.nx_win);
  j.at("ny_win").get_to(s.ny_win);
  j.at("stride_x").get_to(s.stride_x);
  j.at("stride_y").get_to(s.stride_y);
  j.at("field_w").get_to(s.field_w);
  j.at("field_h").get_to(s.field_h);
  j.at("z").get_to(s.z);
  j.at("lambda").get_to(s.lambda);
}

namespace {

// One thread per element of [ny_win, nx_win, win_h, win_w].
//
// For window at grid position (gx, gy), local pixel (i, j):
//
//   Xc = (gx * stride_x + (win_w-1)/2 - (field_w-1)/2) * dx   [global center X, in meters]
//   Yc = (gy * stride_y + (win_h-1)/2 - (field_h-1)/2) * dy   [global center Y, in meters]
//   x' = (i - (win_w-1)/2) * dx                                [local X relative to window center]
//   y' = (j - (win_h-1)/2) * dy
//
//   phase = k/z * (Xc*x' + Yc*y') + k/(2z) * (Xc^2 + Yc^2)
//
__global__ void fill_stft_phase_ramps_kernel(cuFloatComplex *ramps, size_t win_w, size_t win_h,
                                             float dx, float dy, size_t nx_win, size_t ny_win,
                                             size_t stride_x, size_t stride_y, size_t field_w,
                                             size_t field_h, float z, float lambda) {
  const size_t tile_size = win_w * win_h;
  const size_t total     = nx_win * ny_win * tile_size;
  const size_t idx       = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  const size_t subap_idx = idx / tile_size;
  const size_t local_idx = idx % tile_size;

  const size_t gx = subap_idx % nx_win;
  const size_t gy = subap_idx / nx_win;
  const size_t i  = local_idx % win_w;
  const size_t j  = local_idx / win_w;

  const float k = 2.0f * CUDART_PI_F / lambda;

  // Global physical center of this window (field centered at origin)
  const float Xc = (static_cast<float>(gx * stride_x) + 0.5f * static_cast<float>(win_w - 1) -
                    0.5f * static_cast<float>(field_w - 1)) *
                   dx;
  const float Yc = (static_cast<float>(gy * stride_y) + 0.5f * static_cast<float>(win_h - 1) -
                    0.5f * static_cast<float>(field_h - 1)) *
                   dy;

  // Local physical coordinate within window (window centered at origin)
  const float xp = (static_cast<float>(i) - 0.5f * static_cast<float>(win_w - 1)) * dx;
  const float yp = (static_cast<float>(j) - 0.5f * static_cast<float>(win_h - 1)) * dy;

  // phase = k/z * (Xc*x' + Yc*y') + k/(2z) * (Xc^2 + Yc^2)
  const float phase = (k / z) * (Xc * xp + Yc * yp) + (k / (2.0f * z)) * (Xc * Xc + Yc * Yc);

  float sin_phase, cos_phase;
  sincosf(phase, &sin_phase, &cos_phase);
  ramps[idx] = make_cuFloatComplex(cos_phase, sin_phase);
}

DevPtrSTF<cuFloatComplex> make_ramps(const ShortTimeFresnelPhaseSettings &s, cudaStream_t stream) {
  const size_t count = s.ny_win * s.nx_win * s.win_h * s.win_w;
  auto         ramps = curaii::make_unique_device_ptr<cuFloatComplex>(count, stream);

  const int block_size = 256;
  const int grid_size  = static_cast<int>((count + block_size - 1) / block_size);

  fill_stft_phase_ramps_kernel<<<grid_size, block_size, 0, stream>>>(
      ramps.get(), s.win_w, s.win_h, s.dx, s.dy, s.nx_win, s.ny_win, s.stride_x, s.stride_y,
      s.field_w, s.field_h, s.z, s.lambda);
  CUDA_CHECK(cudaGetLastError());

  return ramps;
}

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ShortTimeFresnelPhaseRampsFactory::infer] {}", msg);
    throw std::invalid_argument("ShortTimeFresnelPhaseRampsFactory: " + msg);
  }
}

} // namespace

ShortTimeFresnelPhaseRamps::ShortTimeFresnelPhaseRamps(
    const ShortTimeFresnelPhaseSettings &settings, DevPtrSTF<cuFloatComplex> &&d_ramps,
    size_t count, cudaStream_t stream)
    : settings_(settings), d_ramps_(std::move(d_ramps)), count_(count), stream_(stream) {}

holoflow::core::OpResult ShortTimeFresnelPhaseRamps::execute(holoflow::core::SyncCtx &ctx) {
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
  CUDA_CHECK(cudaMemcpyAsync(odata, d_ramps_.get(), count_ * sizeof(cuFloatComplex),
                             cudaMemcpyDeviceToDevice, stream_));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ShortTimeFresnelPhaseRampsFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                         const nlohmann::json                  &jsettings) const {
  const auto s = jsettings.get<ShortTimeFresnelPhaseSettings>();

  check(input_descs.empty(), "expected zero input tensors");
  check(s.win_w > 0, "win_w must be positive");
  check(s.win_h > 0, "win_h must be positive");
  check(s.nx_win > 0, "nx_win must be positive");
  check(s.ny_win > 0, "ny_win must be positive");
  check(s.stride_x > 0, "stride_x must be positive");
  check(s.stride_y > 0, "stride_y must be positive");
  check(s.field_w > 0, "field_w must be positive");
  check(s.field_h > 0, "field_h must be positive");
  check(s.dx > 0.0f, "dx must be positive");
  check(s.dy > 0.0f, "dy must be positive");
  check(s.lambda > 0.0f, "lambda must be positive");
  check(s.z != 0.0f, "z must be non-zero");

  holoflow::core::TDesc odesc({s.ny_win, s.nx_win, s.win_h, s.win_w}, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return {.input_descs   = {},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
ShortTimeFresnelPhaseRampsFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                          const nlohmann::json                  &jsettings,
                                          const holoflow::core::SyncCreateCtx   &ctx) const {
  infer(input_descs, jsettings);
  const auto s     = jsettings.get<ShortTimeFresnelPhaseSettings>();
  const auto count = s.ny_win * s.nx_win * s.win_h * s.win_w;
  return std::unique_ptr<holoflow::core::ISyncTask>(
      new ShortTimeFresnelPhaseRamps(s, make_ramps(s, ctx.stream), count, ctx.stream));
}

std::unique_ptr<holoflow::core::ISyncTask>
ShortTimeFresnelPhaseRampsFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                          std::span<const holoflow::core::TDesc>     input_descs,
                                          const nlohmann::json                      &jsettings,
                                          const holoflow::core::SyncCreateCtx       &ctx) const {
  infer(input_descs, jsettings);
  auto *old = dynamic_cast<ShortTimeFresnelPhaseRamps *>(old_task.get());
  if (old != nullptr) {
    const auto s = jsettings.get<ShortTimeFresnelPhaseSettings>();
    if (s == old->get_settings()) {
      old->update_stream(ctx.stream);
      return old_task;
    }
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
