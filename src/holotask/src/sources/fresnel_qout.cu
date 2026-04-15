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

#include "holotask/sources/fresnel_qout.hh"

#include <cuComplex.h>
#include <math_constants.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::sources {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void to_json(nlohmann::json &j, const FresnelQoutSettings &fqs) {
  j = nlohmann::json{
      {"lambda", fqs.lambda}, {"dx", fqs.dx}, {"dy", fqs.dy}, {"nx", fqs.nx}, {"ny", fqs.ny},
  };
}

void from_json(const nlohmann::json &j, FresnelQoutSettings &fqs) {
  j.at("lambda").get_to(fqs.lambda);
  j.at("dx").get_to(fqs.dx);
  j.at("dy").get_to(fqs.dy);
  j.at("nx").get_to(fqs.nx);
  j.at("ny").get_to(fqs.ny);
}

namespace {

class FresnelQout : public holoflow::core::ISyncTask {
public:
  FresnelQout(FresnelQoutSettings settings, holoflow::core::TDesc idesc, DevPtr<float> &&d_r2,
              cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), d_r2_(std::move(d_r2)),
        stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  const FresnelQoutSettings   &settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  FresnelQoutSettings   settings_;
  holoflow::core::TDesc idesc_;
  DevPtr<float>         d_r2_;
  cudaStream_t          stream_;
};

__global__ void quadratic_phase_shift_kernel(const float *r2, const float *z, cuFloatComplex *qout,
                                             int width, int height, float lambda);
__global__ void quadratic_r2_kernel(float *r2, int width, int height, float dx, float dy);
} // namespace

holoflow::core::OpResult FresnelQout::execute(holoflow::core::SyncCtx &ctx) {
  auto *z_ptr = reinterpret_cast<const float *>(ctx.inputs[0].data());
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
  int   nx    = static_cast<int>(settings_.nx);
  int   ny    = static_cast<int>(settings_.ny);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
  quadratic_phase_shift_kernel<<<grid_size, block_size, 0, stream_>>>(d_r2_.get(), z_ptr, odata,
                                                                       nx, ny, settings_.lambda);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

namespace {

__global__ void quadratic_phase_shift_kernel(const float *r2, const float *z, cuFloatComplex *qout,
                                             int width, int height, float lambda) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  const int   idx   = row * width + col;
  const float z_val = z[0];
  if (z_val == 0.0f) {
    qout[idx] = make_cuComplex(0.0f, 0.0f);
    return;
  }

  const float k           = 2.0f * CUDART_PI_F / lambda;
  const float total_phase = k * z_val + CUDART_PI_F / (lambda * z_val) * r2[idx];
  const float amplitude   = 1.0f / (lambda * z_val);
  float       sin_phase   = 0.0f;
  float       cos_phase   = 0.0f;
  sincosf(total_phase, &sin_phase, &cos_phase);

  qout[idx] = make_cuComplex(amplitude * sin_phase, -amplitude * cos_phase);
}

__global__ void quadratic_r2_kernel(float *r2, int width, int height, float dx, float dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size     = width > height ? width : height;
  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * dx;
  float y = ((row + offset_y) - size / 2.0f) * dy;

  r2[row * width + col] = x * x + y * y;
}

DevPtr<float> make_quadratic_r2(const FresnelQoutSettings &settings, cudaStream_t stream) {
  auto nx   = static_cast<int>(settings.nx);
  auto ny   = static_cast<int>(settings.ny);
  auto d_r2 = curaii::make_unique_device_ptr<float>(settings.nx * settings.ny, stream);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
  quadratic_r2_kernel<<<grid_size, block_size, 0, stream>>>(d_r2.get(), nx, ny, settings.dx,
                                                             settings.dy);
  return d_r2;
}

} // namespace

holoflow::core::InferResult
FresnelQoutFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[FresnelQoutFactory::infer] error: {}", msg);
      throw std::invalid_argument("FresnelQoutFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<FresnelQoutSettings>();

  check(input_descs.size() == 1, "expected exactly one input tensor (z)");
  const auto &z_desc = input_descs[0];

  check(z_desc.mem_loc == holoflow::core::MemLoc::Device, "z tensor must be on device");
  check(z_desc.dtype == holoflow::core::DType::F32, "z tensor must be F32");
  check(z_desc.num_elements() == 1, "z tensor must be a scalar");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.nx > 0, "nx must be positive");
  check(settings.ny > 0, "ny must be positive");

  holoflow::core::TDesc odesc({settings.ny, settings.nx}, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {z_desc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelQoutFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings,
                           const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  auto settings = jsettings.get<FresnelQoutSettings>();

  auto d_r2 = make_quadratic_r2(settings, ctx.stream);
  return std::make_unique<FresnelQout>(settings, input_descs[0], std::move(d_r2), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelQoutFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                           std::span<const holoflow::core::TDesc>     input_descs,
                           const nlohmann::json                      &jsettings,
                           const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_fresnel = dynamic_cast<FresnelQout *>(old_task.get());
  if (old_fresnel == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto  new_settings = jsettings.get<FresnelQoutSettings>();
  const auto &new_idesc    = input_descs[0];
  const auto &old_idesc    = old_fresnel->idesc();

  const bool can_reuse =
      (new_settings == old_fresnel->settings()) && (new_idesc.shape == old_idesc.shape) &&
      (new_idesc.strides == old_idesc.strides) && (new_idesc.dtype == old_idesc.dtype) &&
      (new_idesc.mem_loc == old_idesc.mem_loc) && (new_idesc.offset == old_idesc.offset);
  if (can_reuse) {
    old_fresnel->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources
