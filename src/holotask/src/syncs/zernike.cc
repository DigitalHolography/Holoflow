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

#include "holotask/syncs/zernike.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <spdlog/fmt/ranges.h>
#include <stdexcept>
#include <vector>
#include <array>
#include <chrono>

#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ZernikeSettings &s) {
  j = nlohmann::json{{"indexes", s.indexes}};
}

void from_json(const nlohmann::json &j, ZernikeSettings &s) {
  s.indexes.clear();
  if (j.contains("indexes")) {
    j.at("indexes").get_to(s.indexes);
  } else if (j.contains("indices")) {
    j.at("indices").get_to(s.indexes);
  }
}

namespace {

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikeFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikeFactory inference error: " + msg);
  }
}

struct Shift {
    float dx;
    float dy;
};

std::vector<Shift> recover_shifts(const holoflow::core::TView& view) {
    const auto& desc = view.desc;
    uint8_t* data = reinterpret_cast<uint8_t*>(view.storage->ptr + desc.offset);

    size_t nb_sub_y = desc.shape[1]; 
    size_t nb_sub_x = desc.shape[2]; 
    size_t win_h    = desc.shape[3]; 
    size_t win_w    = desc.shape[4]; 

    float center_y = (win_h - 1) / 2.0f;
    float center_x = (win_w - 1) / 2.0f;

    std::vector<Shift> shifts;
    shifts.reserve(nb_sub_y * nb_sub_x);

    for (size_t sy = 0; sy < nb_sub_y; ++sy) {
        for (size_t sx = 0; sx < nb_sub_x; ++sx) {
            uint8_t* subap_ptr = data + (sy * desc.strides[1]) + (sx * desc.strides[2]);

            int max_val = -1;
            int peak_y = 0, peak_x = 0;

            for (int y = 0; y < (int)win_h; ++y) {
                for (int x = 0; x < (int)win_w; ++x) {
                    uint8_t val = subap_ptr[y * desc.strides[3] + x];
                    if (val > max_val) {
                        max_val = val;
                        peak_y = y;
                        peak_x = x;
                    }
                }
            }

            float sum_val = 0.0f, sum_y = 0.0f, sum_x = 0.0f;
            int radius = 2; 

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int iy = peak_y + dy;
                    int ix = peak_x + dx;
                    if (iy >= 0 && iy < (int)win_h && ix >= 0 && ix < (int)win_w) {
                        float val = static_cast<float>(subap_ptr[iy * desc.strides[3] + ix]);
                        sum_val += val;
                        sum_y += val * iy;
                        sum_x += val * ix;
                    }
                }
            }

            float final_y = (sum_val > 0) ? (sum_y / sum_val) : (float)peak_y;
            float final_x = (sum_val > 0) ? (sum_x / sum_val) : (float)peak_x;
            shifts.push_back({final_x - center_x, final_y - center_y});
        }
    }
    return shifts;
}

// Solves 3x3 linear system M*x = b
std::array<float, 3> solve3x3(const float M[3][3], const float b[3]) {
    float det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if (std::abs(det) < 1e-9f) return {0, 0, 0};

    float invDet = 1.0f / det;
    float inv[3][3];
    inv[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet;
    inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * invDet;
    inv[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet;
    inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * invDet;
    inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * invDet;

    std::array<float, 3> x_out = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            x_out[i] += inv[i][j] * b[j];
    return x_out;
}

} // namespace


Zernike::Zernike(const ZernikeSettings &settings) : settings_(settings) {}

holoflow::core::OpResult Zernike::execute(holoflow::core::SyncCtx &ctx) {

  // do once per second or so, not critical to be super fast
  static auto last_log_time = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  if (now - last_log_time > std::chrono::seconds(1)) {
      logger()->info("Zernike task executing...");
      last_log_time = now;
  }
  else {
    return holoflow::core::OpResult::Ok; // Skip execution to reduce log spam
  }

  logger()->info("Executing Zernike task with indexes: {}", settings_.indexes);

  
  auto shifts = recover_shifts(ctx.inputs[0]);
  const auto& desc = ctx.inputs[0].desc;
  size_t nb_sub_y = desc.shape[1];
  size_t nb_sub_x = desc.shape[2];

  float GtG[3][3] = {0};
  float Gts[3] = {0};
  float sqrt3 = std::sqrt(3.0f);
  float sqrt6 = std::sqrt(6.0f);

  for (size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (size_t sx = 0; sx < nb_sub_x; ++sx) {
      // Normalize coords to [-1, 1] for a 5x5 grid
      float x = (static_cast<float>(sx) - 2.0f) / 2.0f;
      float y = (static_cast<float>(sy) - 2.0f) / 2.0f;

      // Derivatives: index 0->a4, 1->a5, 2->a6
      float g_dx[3] = { 4.0f * sqrt3 * x, sqrt6 * 2.0f * y,  sqrt6 * 2.0f * x };
      float g_dy[3] = { 4.0f * sqrt3 * y, sqrt6 * 2.0f * x, -sqrt6 * 2.0f * y };

      const auto& s = shifts[sy * nb_sub_x + sx];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          GtG[i][j] += g_dx[i] * g_dx[j] + g_dy[i] * g_dy[j];
        }
        Gts[i] += g_dx[i] * s.dx + g_dy[i] * s.dy;
      }
    }
  }

  auto coefs = solve3x3(GtG, Gts);
  
  // Map coefficients to requested output indexes
  float* out_ptr = reinterpret_cast<float*>(ctx.outputs[0].data());
  for (size_t i = 0; i < settings_.indexes.size(); ++i) {
      int idx = settings_.indexes[i];
      float val = coefs[idx - 4]; // a4 is index 0 in our solver
      out_ptr[i] = val;
      logger()->info("Zernike Coefficient a{}: {:.4f}", idx, val);
  }

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ZernikeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikeSettings>();

  check(input_descs.size() == 1, "Zernike task must have exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Input memory location must be Host");
  check(idesc.dtype == holoflow::core::DType::U8, "Input dtype must be U8");
  check(idesc.rank() == 5, "Input rank must be 5");

  check(!settings.indexes.empty(), "indexes must not be empty");
  for (int idx : settings.indexes) {
    check(idx == 4 || idx == 5 || idx == 6, "Only zernike indexes 4, 5, 6 are supported");
  }
  auto uniq = settings.indexes;
  std::sort(uniq.begin(), uniq.end());
  check(std::adjacent_find(uniq.begin(), uniq.end()) == uniq.end(), "indexes must be unique");

  holoflow::core::TDesc odesc({settings.indexes.size()}, holoflow::core::DType::F32,
                              holoflow::core::MemLoc::Host);

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
ZernikeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;
  infer(input_descs, jsettings);
  const auto settings = jsettings.get<ZernikeSettings>();
  return std::unique_ptr<holoflow::core::ISyncTask>(new Zernike(settings));
}

} // namespace holotask::syncs