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
#include <utility>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

namespace {

constexpr int kMaxSupportedModes = 9; // Noll indices 2..10

} // namespace

void to_json(nlohmann::json &j, const ZernikePhaseSettings &s) {
  j = nlohmann::json{
      {"indexes", s.indexes},
      {"ny", s.ny},
      {"nx", s.nx},
      {"output", s.output},
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

  if (j.contains("output")) {
    j.at("output").get_to(s.output);
  } else if (j.contains("device")) {
    j.at("device").get_to(s.output);
  } else {
    s.output = holoflow::core::MemLoc::Host;
  }
}

namespace {

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikePhaseFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikePhaseFactory inference error: " + msg);
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

struct ZernikePhaseKernelSettings {
  int indexes[kMaxSupportedModes]{};
  int n_modes = 0;
  int ny      = 0;
  int nx      = 0;
};

ZernikePhaseKernelSettings make_kernel_settings(const ZernikePhaseSettings &settings) {
  ZernikePhaseKernelSettings kernel_settings{
      .n_modes = static_cast<int>(settings.indexes.size()),
      .ny      = settings.ny,
      .nx      = settings.nx,
  };

  for (std::size_t i = 0; i < settings.indexes.size(); ++i) {
    kernel_settings.indexes[i] = settings.indexes[i];
  }

  return kernel_settings;
}

__device__ float eval_zernike_noll_value_device(int noll_index, float x_n, float y_n) {
  constexpr float sqrt3 = 1.7320508075688772f;
  constexpr float sqrt6 = 2.4494897427831781f;
  constexpr float sqrt8 = 2.8284271247461903f;

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
    return 0.0f;
  }
}

__global__ void zernike_phase_kernel(const float *coefficients, float *phase,
                                     ZernikePhaseKernelSettings settings) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n   = settings.ny * settings.nx;
  if (idx >= n) {
    return;
  }

  const int y_idx = idx / settings.nx;
  const int x_idx = idx % settings.nx;

  const int   min_size     = settings.ny < settings.nx ? settings.ny : settings.nx;
  const float center_y     = (static_cast<float>(settings.ny) - 1.0f) * 0.5f;
  const float center_x     = (static_cast<float>(settings.nx) - 1.0f) * 0.5f;
  const float pupil_radius = 0.5f * static_cast<float>(min_size);
  const float x_n          = (static_cast<float>(x_idx) - center_x) / pupil_radius;
  const float y_n          = (static_cast<float>(y_idx) - center_y) / pupil_radius;

  const float r2        = x_n * x_n + y_n * y_n;
  float       phase_rad = 0.0f;

  if (r2 <= 1.0f || true) {
    for (int i = 0; i < settings.n_modes; ++i) {
      const int   noll_index = settings.indexes[i];
      const float z_value    = eval_zernike_noll_value_device(noll_index, x_n, y_n);
      phase_rad += coefficients[i] * z_value;
    }
  }

  phase[idx] = phase_rad;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// ZernikePhase task implementation
// -------------------------------------------------------------------------------------------------

class ZernikePhase : public holoflow::core::ISyncTask {
public:
  ZernikePhase(ZernikePhaseSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), kernel_settings_(make_kernel_settings(settings_)),
        stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto *in_data  = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto       *out_data = reinterpret_cast<float *>(ctx.outputs[0].data());

    if (settings_.output == holoflow::core::MemLoc::Device) {
      const int     n          = settings_.ny * settings_.nx;
      constexpr int block_size = 256;
      const int     grid_size  = (n + block_size - 1) / block_size;
      zernike_phase_kernel<<<grid_size, block_size, 0, stream_>>>(in_data, out_data,
                                                                  kernel_settings_);
      CUDA_CHECK(cudaGetLastError());
      return holoflow::core::OpResult::Ok;
    }

    const int ny = settings_.ny;
    const int nx = settings_.nx;

    const float center_y     = (static_cast<float>(ny) - 1.0f) * 0.5f;
    const float center_x     = (static_cast<float>(nx) - 1.0f) * 0.5f;
    const float pupil_radius = 0.5f * static_cast<float>(std::min(ny, nx));

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

  void                        update_stream(cudaStream_t stream) { stream_ = stream; }
  const ZernikePhaseSettings &settings() const { return settings_; }

private:
  ZernikePhaseSettings       settings_;
  ZernikePhaseKernelSettings kernel_settings_;
  cudaStream_t               stream_;
};

// -------------------------------------------------------------------------------------------------
// ZernikePhaseFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ZernikePhaseFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikePhaseSettings>();

  check(input_descs.size() == 1, "ZernikePhase task must have exactly one input");

  const auto &idesc = input_descs[0];
  check(settings.output == holoflow::core::MemLoc::Host ||
            settings.output == holoflow::core::MemLoc::Device,
        "Output memory location must be Host or Device");
  check(idesc.mem_loc == settings.output,
        "Input coefficients must be in the output memory location");
  check(idesc.dtype == holoflow::core::DType::F32, "Input coefficients dtype must be F32");
  check(idesc.rank() == 1, "Input coefficients rank must be 1");
  check(is_c_contiguous(idesc), "Input coefficients must be C-contiguous");

  check(!settings.indexes.empty(), "indexes must not be empty");
  check(settings.indexes.size() <= kMaxSupportedModes, "Too many requested Zernike modes");
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
      holoflow::core::DType::F32, settings.output);

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
  (void)infer(input_descs, jsettings);

  const auto settings = jsettings.get<ZernikePhaseSettings>();
  return std::make_unique<ZernikePhase>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikePhaseFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                            std::span<const holoflow::core::TDesc>     input_descs,
                            const nlohmann::json                      &jsettings,
                            const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_zernike_phase = dynamic_cast<ZernikePhase *>(old_task.get());
  if (old_zernike_phase == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<ZernikePhaseSettings>();
  if (settings == old_zernike_phase->settings()) {
    old_zernike_phase->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
