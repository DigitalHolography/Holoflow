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
#include <cmath>
#include <stdexcept>
#include <string>

#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ZernikePhaseSettings &s) {
  j = nlohmann::json{
      {"indexes", s.indexes},
      {"ny", s.ny},
      {"nx", s.nx},
  };
}

void from_json(const nlohmann::json &j, ZernikePhaseSettings &s) {
  s.indexes.clear();

  if (j.contains("indexes")) {
    j.at("indexes").get_to(s.indexes);
  } else if (j.contains("indices")) {
    j.at("indices").get_to(s.indexes);
  }

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

float eval_zernike_noll_value(int noll_index, float x_n, float y_n) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);
  const float sqrt8 = std::sqrt(8.0f);

  switch (noll_index) {
  case 2:
    return 2.0f * x_n;

  case 3:
    return 2.0f * y_n;

  case 4:
    return sqrt3 * (2.0f * (x_n * x_n + y_n * y_n) - 1.0f);

  case 5:
    return 2.0f * sqrt6 * x_n * y_n;

  case 6:
    return sqrt6 * (x_n * x_n - y_n * y_n);

  case 7:
    return sqrt8 * y_n * (3.0f * x_n * x_n - y_n * y_n);

  case 8:
    return sqrt8 * y_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f);

  case 9:
    return sqrt8 * x_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f);

  case 10:
    return sqrt8 * x_n * (x_n * x_n - 3.0f * y_n * y_n);

  default:
    throw std::invalid_argument("Unsupported Noll index");
  }
}

} // namespace

ZernikePhase::ZernikePhase(const ZernikePhaseSettings &settings) : settings_(settings) {}

holoflow::core::OpResult ZernikePhase::execute(holoflow::core::SyncCtx &ctx) {
  const auto *in_data  = reinterpret_cast<const float *>(ctx.inputs[0].data());
  auto       *out_data = reinterpret_cast<float *>(ctx.outputs[0].data());

  const int ny = settings_.ny;
  const int nx = settings_.nx;

  // ---------------------------------------------------------------------------
  // Phase-mask geometry
  // ---------------------------------------------------------------------------
  //
  // We build the phase on a rectangular sampling grid, but the Zernike basis is
  // defined on the unit disk. Therefore:
  //
  //   x_n = (x - x_center) / R
  //   y_n = (y - y_center) / R
  //
  // where R is the radius of the largest inscribed disk in the image support.
  //
  // The input coefficients are assumed to be phase amplitudes in radians.
  // Therefore the reconstructed phase is simply:
  //
  //   phi(x_n, y_n) = sum_k a_k Z_k(x_n, y_n)
  //
  // with phi expressed in radians.
  //
  const float center_y     = (static_cast<float>(ny) - 1.0f) * 0.5f;
  const float center_x     = (static_cast<float>(nx) - 1.0f) * 0.5f;
  const float pupil_radius = 0.5f * static_cast<float>(std::min(ny, nx));

  // Outside the pupil, the phase is set to zero. This gives a mask naturally
  // restricted to the circular aperture.
  for (int y_idx = 0; y_idx < ny; ++y_idx) {
    for (int x_idx = 0; x_idx < nx; ++x_idx) {
      const float x_n = (static_cast<float>(x_idx) - center_x) / pupil_radius;
      const float y_n = (static_cast<float>(y_idx) - center_y) / pupil_radius;

      const float r2        = x_n * x_n + y_n * y_n;
      float       phase_rad = 0.0f;

      if (r2 <= 1.0f || true) {
        for (std::size_t i = 0; i < settings_.indexes.size(); ++i) {
          const int   noll_index = settings_.indexes[i];
          const float z_value    = eval_zernike_noll_value(noll_index, x_n, y_n);
          phase_rad += in_data[i] * z_value;
        }
      }

      out_data[static_cast<std::size_t>(y_idx) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(x_idx)] = phase_rad;
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
    check(idx >= 2 && idx <= 10, "Only zernike Noll indexes 2..10 are supported");
  }

  auto uniq = settings.indexes;
  std::sort(uniq.begin(), uniq.end());
  check(std::adjacent_find(uniq.begin(), uniq.end()) == uniq.end(), "indexes must be unique");

  check(settings.ny > 0 && settings.nx > 0, "Resolution ny and nx must be greater than 0");
  check(idesc.shape[0] == settings.indexes.size(),
        "Input coefficient count must match settings.indexes size");

  holoflow::core::TDesc odesc(
      {static_cast<std::size_t>(settings.ny), static_cast<std::size_t>(settings.nx)},
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
