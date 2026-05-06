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

#include "holotask/syncs/zernike_defocus_z_prop.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "bug.hh"
#include "cuda_runtime_api.h"
#include "logger.hh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ZernikeDefocusZPropSettings &s) {
  j = nlohmann::json{
      {"indexes", s.indexes},
      {"lambda", s.lambda},
      {"z_curr", s.z_curr},
      {"pupil_radius", s.pupil_radius},
  };
}

void from_json(const nlohmann::json &j, ZernikeDefocusZPropSettings &s) {
  s.indexes.clear();

  if (j.contains("indexes")) {
    j.at("indexes").get_to(s.indexes);
  } else if (j.contains("indices")) {
    j.at("indices").get_to(s.indexes);
  }

  j.at("lambda").get_to(s.lambda);
  j.at("z_curr").get_to(s.z_curr);
  j.at("pupil_radius").get_to(s.pupil_radius);
}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

constexpr int   kDefocusNollIndex = 4;
constexpr float kInverseZEpsilon  = 1e-18f;

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikeDefocusZPropFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikeDefocusZPropFactory inference error: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.rank(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc;
}

std::size_t defocus_position(const std::vector<int> &indexes) {
  const auto it = std::find(indexes.begin(), indexes.end(), kDefocusNollIndex);
  return static_cast<std::size_t>(std::distance(indexes.begin(), it));
}

struct ZPropEstimate {
  double delta_inv_z = 0.0;
  double z_new       = 0.0;
  double delta_z_mm  = 0.0;
};

ZPropEstimate estimate_z_prop_from_a4(float a4_rad, const ZernikeDefocusZPropSettings &settings) {
  const double pupil_radius_sq = static_cast<double>(settings.pupil_radius) * settings.pupil_radius;
  const double delta_inv_z     = (2.0 * std::sqrt(3.0) * settings.lambda * a4_rad) /
                                 (static_cast<double>(M_PI) * pupil_radius_sq);
  const double inv_z_new       = (1.0 / settings.z_curr) - delta_inv_z;
  const double z_new           = 1.0 / inv_z_new;

  return {
      .delta_inv_z = delta_inv_z,
      .z_new       = z_new,
      .delta_z_mm  = 1000.0 * (z_new - settings.z_curr),
  };
}

// -------------------------------------------------------------------------------------------------
// ZernikeDefocusZProp task implementation
// -------------------------------------------------------------------------------------------------

class ZernikeDefocusZProp : public holoflow::core::ISyncTask {
public:
  ZernikeDefocusZProp(ZernikeDefocusZPropSettings settings, holoflow::core::TDesc idesc,
                      cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto       &input       = ctx.inputs[0];
    const auto  a4_position = defocus_position(settings_.indexes);
    const auto *src         = reinterpret_cast<const float *>(input.data()) + a4_position;

    float a4_rad = 0.0f;
    switch (input.desc.mem_loc) {
    case holoflow::core::MemLoc::Host:
      a4_rad = *src;
      break;
    case holoflow::core::MemLoc::Device:
      CUDA_CHECK(cudaMemcpyAsync(&a4_rad, src, sizeof(float), cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      break;
    default:
      throw std::logic_error("Unsupported memory location for Zernike defocus z_prop estimate");
    }

    if (!std::isfinite(a4_rad)) {
      logger()->info("[ZernikeDefocusZPropTask] a4={:.4e} rad; estimated z_prop is undefined",
                     a4_rad);
      return holoflow::core::OpResult::Ok;
    }

    const auto estimate = estimate_z_prop_from_a4(a4_rad, settings_);
    if (!std::isfinite(estimate.z_new) || std::abs(1.0 / estimate.z_new) <= kInverseZEpsilon) {
      logger()->info("[ZernikeDefocusZPropTask] a4={:.4e} rad; estimated z_prop is undefined",
                     a4_rad);
      return holoflow::core::OpResult::Ok;
    }

    logger()->info("[ZernikeDefocusZPropTask] a4={:.4e} rad, delta_inv_z={:.4e} 1/m, "
                   "z_curr={:.4e} m, z_new={:.4e} m, delta_z={:.4e} mm "
                   "(pupil_radius={:.4e} m)",
                   a4_rad, estimate.delta_inv_z, settings_.z_curr, estimate.z_new,
                   estimate.delta_z_mm, settings_.pupil_radius);
    return holoflow::core::OpResult::Ok;
  }

  const ZernikeDefocusZPropSettings &settings() const { return settings_; }
  const holoflow::core::TDesc       &idesc() const { return idesc_; }
  void                               update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ZernikeDefocusZPropSettings settings_;
  holoflow::core::TDesc       idesc_;
  cudaStream_t                stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// ZernikeDefocusZPropFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ZernikeDefocusZPropFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                  const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikeDefocusZPropSettings>();

  check(settings.lambda > 0.0f, "lambda must be positive");
  check(settings.z_curr != 0.0f, "z_curr must be non-zero");
  check(settings.pupil_radius > 0.0f, "pupil_radius must be positive");
  check(!settings.indexes.empty(), "indexes must not be empty");
  check(std::find(settings.indexes.begin(), settings.indexes.end(), kDefocusNollIndex) !=
            settings.indexes.end(),
        "indexes must contain Z4 defocus");

  auto unique_indexes = settings.indexes;
  std::sort(unique_indexes.begin(), unique_indexes.end());
  check(std::adjacent_find(unique_indexes.begin(), unique_indexes.end()) == unique_indexes.end(),
        "indexes must be unique");

  check(input_descs.size() == 1, "ZernikeDefocusZProp task must have exactly one input");

  const auto &idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::F32, "Input coefficients dtype must be F32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Host ||
            idesc.mem_loc == holoflow::core::MemLoc::Device,
        "Input coefficients must be in Host or Device memory");
  check(idesc.rank() == 1, "Input coefficients rank must be 1");
  check(is_c_contiguous(idesc), "Input coefficients must be C-contiguous");
  check(idesc.shape[0] == settings.indexes.size(),
        "Input coefficient count must match configured indexes size");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeDefocusZPropFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json                  &jsettings,
                                   const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);

  auto settings = jsettings.get<ZernikeDefocusZPropSettings>();
  return std::make_unique<ZernikeDefocusZProp>(std::move(settings), input_descs[0], ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeDefocusZPropFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                   std::span<const holoflow::core::TDesc>     input_descs,
                                   const nlohmann::json                      &jsettings,
                                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old = dynamic_cast<ZernikeDefocusZProp *>(old_task.get());
  if (old == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  auto settings = jsettings.get<ZernikeDefocusZPropSettings>();
  if (settings == old->settings() && same_desc(input_descs[0], old->idesc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
