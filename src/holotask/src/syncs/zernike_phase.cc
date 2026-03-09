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

#include "holotask/syncs/zernike_phase.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ZernikePhaseSettings &s) {
  j = nlohmann::json{{"indexes", s.indexes}, {"ny", s.ny}, {"nx", s.nx}};
}

void from_json(const nlohmann::json &j, ZernikePhaseSettings &s) {
  s.indexes.clear();
  if (j.contains("indexes"))
    j.at("indexes").get_to(s.indexes);
  else if (j.contains("indices"))
    j.at("indices").get_to(s.indexes);

  j.at("ny").get_to(s.ny);
  j.at("nx").get_to(s.nx);
}

namespace {

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikePhaseFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikePhaseFactory inference error: " + msg);
  }
}

} // namespace

ZernikePhase::ZernikePhase(const ZernikePhaseSettings &settings) : settings_(settings) {}

holoflow::core::OpResult ZernikePhase::execute(holoflow::core::SyncCtx &ctx) {
  // 1. Read input coefficients
  auto &input_view = ctx.inputs[0];
  auto *in_data    = reinterpret_cast<const float *>(input_view.data());

  // Parse coefficients based on requested indexes
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f;
  for (size_t i = 0; i < settings_.indexes.size(); ++i) {
    if (settings_.indexes[i] == 4)
      a4 = in_data[i];
    if (settings_.indexes[i] == 5)
      a5 = in_data[i];
    if (settings_.indexes[i] == 6)
      a6 = in_data[i];
  }

  logger()->info("Generating Zernike phase mask with coefficients: a4={}, a5={}, a6={}", a4, a5,
                 a6);
  // a4 *= 10.0f; // Scale coefficients for more visible phase (tune as needed)

  // 2. Setup output phase mask buffer
  auto &output_view = ctx.outputs[0];
  auto *out_data    = reinterpret_cast<float *>(output_view.data());

  int ny = settings_.ny;
  int nx = settings_.nx;

  // 3. Geometry Setup (Normalization by shorter dimension)
  float center_y    = (ny - 1) / 2.0f;
  float center_x    = (nx - 1) / 2.0f;
  float norm_radius = std::min(ny, nx) / 2.0f;

  float sqrt3 = std::sqrt(3.0f);
  float sqrt6 = std::sqrt(6.0f);

  // 4. Generate the Phase Mask
  for (int y_idx = 0; y_idx < ny; ++y_idx) {
    for (int x_idx = 0; x_idx < nx; ++x_idx) {
      // Normalized coordinates
      float y  = (y_idx - center_y) / norm_radius;
      float x  = (x_idx - center_x) / norm_radius;
      float r2 = x * x + y * y;

      // Evaluate Zernike Modes (Noll 4, 5, 6)
      float z4 = sqrt3 * (2.0f * r2 - 1.0f);
      float z5 = sqrt6 * (2.0f * x * y);
      float z6 = sqrt6 * (x * x - y * y);

      // Sum weighted phase.
      // Note: If you want this mask to CORRECT the aberration, you might want
      // to store the negative of this sum. Here we generate the exact measured phase.
      float phase = a4 * z4 + a5 * z5 + a6 * z6;

      // Write to contiguous 1D array representing 2D tensor
      out_data[y_idx * nx + x_idx] = phase;
    }
  }

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ZernikePhaseFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikePhaseSettings>();

  check(input_descs.size() == 1, "ZernikePhase task must have exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Input coefficients must be in host memory");
  check(idesc.dtype == holoflow::core::DType::F32, "Input coefficients dtype must be F32");
  check(idesc.rank() == 1, "Input coefficients rank must be 1");

  check(!settings.indexes.empty(), "indexes must not be empty");
  for (int idx : settings.indexes) {
    check(idx == 4 || idx == 5 || idx == 6, "Only zernike noll indexes 4, 5, 6 are supported");
  }

  check(settings.ny > 0 && settings.nx > 0, "Resolution ny and nx must be greater than 0");

  auto uniq = settings.indexes;
  std::sort(uniq.begin(), uniq.end());
  check(std::adjacent_find(uniq.begin(), uniq.end()) == uniq.end(), "indexes must be unique");

  check(idesc.shape[0] == settings.indexes.size(),
        "Input coefficient count must match settings.indexes size");

  // Output descriptor is now a 2D F32 tensor mapping to [ny, nx]
  holoflow::core::TDesc odesc({static_cast<size_t>(settings.ny), static_cast<size_t>(settings.nx)},
                              holoflow::core::DType::F32, holoflow::core::MemLoc::Host);

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
ZernikePhaseFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings,
                            const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;
  infer(input_descs, jsettings);
  const auto settings = jsettings.get<ZernikePhaseSettings>();
  return std::unique_ptr<holoflow::core::ISyncTask>(new ZernikePhase(settings));
}

} // namespace holotask::syncs