// Copyright 2025 Digital Holography Foundation
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

#include "holotask/sources/fresnel_qin.hh"

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holotask::sources {

void to_json(nlohmann::json &j, const FresnelQinSettings &fqs) {
  j = nlohmann::json{
      {"lambda", fqs.lambda}, {"dx", fqs.dx}, {"dy", fqs.dy}, {"nx", fqs.nx}, {"ny", fqs.ny},
  };
}

void from_json(const nlohmann::json &j, FresnelQinSettings &fqs) {
  j.at("lambda").get_to(fqs.lambda);
  j.at("dx").get_to(fqs.dx);
  j.at("dy").get_to(fqs.dy);
  j.at("nx").get_to(fqs.nx);
  j.at("ny").get_to(fqs.ny);
}

FresnelQin::FresnelQin(const FresnelQinSettings &settings, const holoflow::core::TDesc &idesc, 
                       DevPtr<float> &&d_r2, cudaStream_t stream)
    : settings_(settings), idesc_(idesc), d_r2_(std::move(d_r2)), stream_(stream) {}

namespace {
__global__ void quadratic_lens_kernel(const float *r2, const float *z, cuFloatComplex *lens,
                                      int width, int height, float lambda);
}

holoflow::core::OpResult FresnelQin::execute(holoflow::core::SyncCtx &ctx) {
  auto *z_ptr = reinterpret_cast<const float *>(ctx.inputs[0].data());
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
  int   nx    = static_cast<int>(settings_.nx);
  int   ny    = static_cast<int>(settings_.ny);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
  quadratic_lens_kernel<<<grid_size, block_size, 0, stream_>>>(d_r2_.get(), z_ptr, odata, nx, ny,
                                                               settings_.lambda);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

namespace {

__global__ void quadratic_lens_kernel(const float *r2, const float *z, cuFloatComplex *lens,
                                      int width, int height, float lambda) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  const int   idx   = row * width + col;
  const float z_val = z[0];
  if (z_val == 0.0f) {
    lens[idx] = make_cuComplex(0.0f, 0.0f);
    return;
  }

  float phase     = CUDART_PI_F / (lambda * z_val) * r2[idx];
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[idx] = make_cuComplex(cos_phase, sin_phase);
}

__global__ void quadratic_r2_kernel(float *r2, int width, int height, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size = width > height ? width : height;

  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * pixel_size;
  float y = ((row + offset_y) - size / 2.0f) * pixel_size;

  r2[row * width + col] = x * x + y * y;
}

DevPtr<float> make_quadratic_r2(const FresnelQinSettings &settings, cudaStream_t stream) {
  auto nx   = static_cast<int>(settings.nx);
  auto ny   = static_cast<int>(settings.ny);
  auto d_r2 = curaii::make_unique_device_ptr<float>(settings.nx * settings.ny, stream);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
  quadratic_r2_kernel<<<grid_size, block_size, 0, stream>>>(d_r2.get(), nx, ny, settings.dx);

  CUDA_CHECK(cudaGetLastError());
  return d_r2;
}

} // namespace

holoflow::core::InferResult
FresnelQinFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                   &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[FresnelQinFactory::infer] error: {}", msg);
      throw std::invalid_argument("FresnelQinFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<FresnelQinSettings>();

  check(input_descs.size() == 1, "expected exactly one input tensor (z)");
  const auto &z_desc = input_descs[0];

  check(z_desc.mem_loc == holoflow::core::MemLoc::Device, "z tensor must be on device");
  check(z_desc.dtype == holoflow::core::DType::F32, "z tensor must be F32");
  check(z_desc.num_elements() == 1, "z tensor must be a scalar");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");
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
FresnelQinFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings,
                          const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<FresnelQinSettings>();

  auto  d_r2 = make_quadratic_r2(settings, ctx.stream);
  auto *task = new FresnelQin(settings, input_descs[0], std::move(d_r2), ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelQinFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                          std::span<const holoflow::core::TDesc>     input_descs,
                          const nlohmann::json                       &jsettings,
                          const holoflow::core::SyncCreateCtx        &ctx) const {

  auto* old_fresnel = dynamic_cast<FresnelQin*>(old_task.get());
  if (old_fresnel != nullptr && input_descs.size() == 1) {
    
    const auto new_settings = jsettings.get<FresnelQinSettings>();
    const auto& new_idesc   = input_descs[0];
    const auto& old_idesc   = old_fresnel->get_idesc();

    bool can_reuse = (new_settings == old_fresnel->get_settings()) &&
                     (new_idesc.shape == old_idesc.shape) &&
                     (new_idesc.strides == old_idesc.strides) &&
                     (new_idesc.dtype == old_idesc.dtype) &&
                     (new_idesc.mem_loc == old_idesc.mem_loc);

    if (can_reuse) {
      old_fresnel->update_stream(ctx.stream);
      return old_task; 
    }
  }

  // Fallback: Structural change detected or invalid old task.
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources