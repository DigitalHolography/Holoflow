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

#include "holotask/syncs/angular_spectrum.hh"

#include <cstdlib>
#include <ranges>
#include <string>
#include <vector>

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const AngularSpectrumSettings::Filter &f) {
  j = {
      {"r_inner", f.r_inner},
      {"r_outer", f.r_outer},
      {"s_inner", f.s_inner},
      {"s_outer", f.s_outer},
  };
}

void from_json(const nlohmann::json &j, AngularSpectrumSettings::Filter &f) {
  j.at("r_inner").get_to(f.r_inner);
  j.at("r_outer").get_to(f.r_outer);
  j.at("s_inner").get_to(f.s_inner);
  j.at("s_outer").get_to(f.s_outer);
}

void to_json(nlohmann::json &j, const AngularSpectrumSettings &as) {
  j = {
      {"lambda", as.lambda},
      {"dx", as.dx},
      {"dy", as.dy},
      {"z", as.z},
  };
  if (as.filter.has_value()) {
    j["filter"] = as.filter.value();
  }
}

void from_json(const nlohmann::json &j, AngularSpectrumSettings &as) {
  j.at("lambda").get_to(as.lambda);
  j.at("dx").get_to(as.dx);
  j.at("dy").get_to(as.dy);
  j.at("z").get_to(as.z);
  if (j.contains("filter")) {
    as.filter = AngularSpectrumSettings::Filter{};
    j.at("filter").get_to(as.filter.value());
  } else {
    as.filter = std::nullopt;
  }
}

// -------------------------------------------------------------------------------------------------
// Private implementation types
// -------------------------------------------------------------------------------------------------

namespace {

// Minimized CallerInfo struct. Dimensions are injected via macros.
struct ApplyLensCallerInfo {
  cuFloatComplex *lens;
};

// -------------------------------------------------------------------------------------------------
// Validation
// -------------------------------------------------------------------------------------------------

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[AngularSpectrumFactory::infer] error: {}", msg);
    throw std::invalid_argument("AngularSpectrumFactory inference error: " + msg);
  }
}

// -------------------------------------------------------------------------------------------------
// NVRTC / JIT-callback compilation
// -------------------------------------------------------------------------------------------------

std::string get_compute_arch() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<std::string> get_nvrtc_args() {
  auto CUDA_PATH = std::getenv("CUDA_PATH");
  HOLOVIBES_CHECK(CUDA_PATH != nullptr, "CUDA_PATH environment variable not set");

  return {
      "-I" + std::string{CUDA_PATH} + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };
}

std::vector<char> compile_source_to_lto(const std::string &source, const std::string &name) {
  auto                 args_string = get_nvrtc_args();
  curaii::NvrtcProgram prog(source.c_str(), name.c_str(), 0, nullptr, nullptr);

  std::vector<char *> args;
  std::ranges::transform(args_string, std::back_inserter(args),
                         [](const std::string &s) { return const_cast<char *>(s.c_str()); });

  try {
    NVRTC_CHECK(nvrtcCompileProgram(prog.get(), static_cast<int>(args.size()), args.data()));
    size_t code_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog.get(), &code_size));
    std::vector<char> lto(code_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog.get(), lto.data()));
    return lto;
  } catch (const curaii::NvrtcError &e) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog.get(), &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog.get(), log.data()));
    logger()->error("[AngularSpectrum] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

std::vector<char> apply_lens_lto(int width, int height, int batch) {
  // Inject compile-time constants for the NVRTC compiler
  std::string src = "#define WIDTH " + std::to_string(width) + "\n" + "#define HEIGHT " +
                    std::to_string(height) + "\n" + "#define BATCH " + std::to_string(batch) + "\n";

  // Common header — struct definition shared with the host side
  src += R"(
#include <cuComplex.h>

struct ApplyLensCallerInfo {
  cuFloatComplex *lens;
};

__device__ cuFloatComplex apply_lens_callback(
    void *data, size_t offset, void *callerInfo, void *sharedPtr) {
  auto  *info     = (ApplyLensCallerInfo *)callerInfo;
  size_t lens_idx = offset % (WIDTH * HEIGHT);
  auto   val      = ((cuFloatComplex *)data)[offset];

  return cuCmulf(val, info->lens[lens_idx]);
}
)";

  return compile_source_to_lto(src, "apply_lens_callback.cu");
}

// -------------------------------------------------------------------------------------------------
// Device kernels
// -------------------------------------------------------------------------------------------------

__global__ void spectral_lens_kernel(cuFloatComplex *lens, int width, int height, float lambda,
                                     float z, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  float du = 1.0f / (width * pixel_size);
  float dv = 1.0f / (height * pixel_size);
  float u  = (col - width / 2) * du;
  float v  = (row - height / 2) * dv;

  float tmp = 1.0f - (lambda * lambda) * (u * u + v * v);
  tmp       = fmaxf(tmp, 0.0f);

  float phase             = 2.0f * CUDART_PI_F * z / lambda * sqrtf(tmp);
  lens[row * width + col] = make_cuComplex(cosf(phase), sinf(phase));
}

__global__ void apply_filter_2d_kernel(cuFloatComplex *filter, const uint32_t width,
                                       const uint32_t height, const uint32_t r_inner,
                                       const uint32_t r_outer, const uint32_t smooth_inner,
                                       const uint32_t smooth_outer) {
  const uint32_t x   = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y   = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t idx = y * width + x;

  if (x >= width || y >= height)
    return;

  // Center the coordinates
  const float r_x  = static_cast<float>(x) - static_cast<float>(width) / 2.0f;
  const float r_y  = static_cast<float>(y) - static_cast<float>(height) / 2.0f;
  const float dist = hypotf(r_x, r_y);

  // Transition boundaries
  const float inner_start = static_cast<float>(r_inner) - static_cast<float>(smooth_inner);
  const float inner_end   = static_cast<float>(r_inner) + static_cast<float>(smooth_inner);
  const float outer_start = static_cast<float>(r_outer) - static_cast<float>(smooth_outer);
  const float outer_end   = static_cast<float>(r_outer) + static_cast<float>(smooth_outer);

  // Define named zones
  const bool in_inner_hole       = (dist < inner_start);
  const bool in_inner_transition = (dist >= inner_start && dist < inner_end);
  const bool in_plateau          = (dist >= inner_end && dist < outer_start);
  const bool in_outer_transition = (dist >= outer_start && dist < outer_end);

  float val = 0.0f;

  if (in_inner_hole) {
    // Before inner radius
    val = 0.0f;
  } else if (in_inner_transition) {
    // Fade in
    val = 0.5f * (1.0f + sinf(CUDART_PI_F / (2.0f * smooth_inner) * (dist - inner_start)));
  } else if (in_plateau) {
    // Select region
    val = 1.0f;
  } else if (in_outer_transition) {
    // Fade out
    val = 0.5f * (1.0f + sinf(CUDART_PI_F / (2.0f * smooth_outer) * (outer_end - dist)));
  } else {
    // Beyond outer radius
    val = 0.0f;
  }

  // Squaring the real value into the complex filter
  // Note: cuCmulf(val, val) results in (val*val - 0*0) + i(val*0 + 0*val)
  filter[idx] = make_cuComplex(val * val, 0.0f);
}

__global__ void swap_corners_kernel(cuFloatComplex *in, cuFloatComplex *out, int width, int height,
                                    int batch) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  int width_half  = width / 2;
  int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch)
    return;

  int             batch_offset = z * width * height;
  cuFloatComplex *in_frame     = in + batch_offset;
  cuFloatComplex *out_frame    = out + batch_offset;

  // Swap top-left with bottom-right
  int top_left_idx     = x + y * width;
  int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  cuFloatComplex tmp          = in_frame[top_left_idx];
  out_frame[top_left_idx]     = in_frame[bottom_right_idx];
  out_frame[bottom_right_idx] = tmp;

  // Swap top-right with bottom-left
  int top_right_idx   = (x + width_half) + y * width;
  int bottom_left_idx = x + (y + height_half) * width;

  tmp                        = in_frame[top_right_idx];
  out_frame[top_right_idx]   = in_frame[bottom_left_idx];
  out_frame[bottom_left_idx] = tmp;
}

DevPtr<cuFloatComplex> make_spectral_lens(int width, int height,
                                          const AngularSpectrumSettings &settings) {
  auto d_lens = curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(width) * height);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  spectral_lens_kernel<<<grid, block>>>(d_lens.get(), width, height, settings.lambda, settings.z,
                                        settings.dx);

  if (settings.filter.has_value()) {
    const auto &f = settings.filter.value();
    apply_filter_2d_kernel<<<grid, block>>>(
        d_lens.get(), static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        static_cast<uint32_t>(f.r_inner), static_cast<uint32_t>(f.r_outer),
        static_cast<uint32_t>(f.s_inner), static_cast<uint32_t>(f.s_outer));
  }

  swap_corners_kernel<<<grid, block>>>(d_lens.get(), d_lens.get(), width, height, 1);
  CUDA_CHECK(cudaGetLastError());
  return d_lens;
}

// -------------------------------------------------------------------------------------------------
// AngularSpectrum task implementation (private to this translation unit)
// -------------------------------------------------------------------------------------------------

class AngularSpectrum : public holoflow::core::ISyncTask {
public:
  // -- Configuration ------------------------------------------------------------------------------
  AngularSpectrumSettings settings;
  holoflow::core::TDesc   idesc;
  curaii::CufftHandle     fwd_plan;
  curaii::CufftHandle     inv_plan;

  // -- Device resources ---------------------------------------------------------------------------
  DevPtr<cuFloatComplex> d_lens;
  DevPtr<void>           d_caller_info;
  std::vector<char>      lto;

  // -- ISyncTask interface ------------------------------------------------------------------------
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data());
    auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
    CUFFT_CHECK(cufftXtExec(fwd_plan.get(), idata, idata, CUFFT_FORWARD));
    CUFFT_CHECK(cufftXtExec(inv_plan.get(), idata, odata, CUFFT_INVERSE));
    return holoflow::core::OpResult::Ok;
  }

  // -- Update utilities ---------------------------------------------------------------------------
  void update_stream(cudaStream_t stream) {
    CUFFT_CHECK(cufftSetStream(fwd_plan.get(), stream));
    CUFFT_CHECK(cufftSetStream(inv_plan.get(), stream));
  }
};

} // namespace

// -------------------------------------------------------------------------------------------------
// AngularSpectrumFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
AngularSpectrumFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings) const {
  auto settings = jsettings.get<AngularSpectrumSettings>();

  // clang-format off
  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() == 3, "input must be a 3D tensor");
  check(idesc.dtype == holoflow::core::DType::CF32, "input must be complex float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");
  // clang-format on

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AngularSpectrumFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  auto        infer    = this->infer(input_descs, jsettings);
  auto        settings = jsettings.get<AngularSpectrumSettings>();
  const auto &idesc    = input_descs[0];

  const int B = static_cast<int>(idesc.shape[0]);
  const int H = static_cast<int>(idesc.shape[1]);
  const int W = static_cast<int>(idesc.shape[2]);

  // -- JIT callback -------------------------------------------------------------------------------
  auto lto    = apply_lens_lto(W, H, B);
  auto d_lens = make_spectral_lens(W, H, settings);

  ApplyLensCallerInfo info{
      .lens = d_lens.get(),
  };
  auto d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(1);
  auto e = cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream);
  CUDA_CHECK(e);

  // -- cuFFT plans --------------------------------------------------------------------------------
  int           rank          = 2;
  long long int n[2]          = {H, W};
  long long int inembed[2]    = {H, W};
  int           istride       = 1;
  int           idist         = H * W;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {H, W};
  int           ostride       = 1;
  int           odist         = H * W;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = B;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle fwd_plan;
  curaii::CufftHandle inv_plan;
  CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "apply_lens_callback", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Construct task directly
  auto task           = std::make_unique<AngularSpectrum>();
  task->settings      = settings;
  task->idesc         = idesc;
  task->fwd_plan      = std::move(fwd_plan);
  task->inv_plan      = std::move(inv_plan);
  task->d_lens        = std::move(d_lens);
  task->d_caller_info = std::move(d_info);
  task->lto           = std::move(lto);

  return task;
}

std::unique_ptr<holoflow::core::ISyncTask>
AngularSpectrumFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                               std::span<const holoflow::core::TDesc>     input_descs,
                               const nlohmann::json                      &jsettings,
                               const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old_angular = dynamic_cast<AngularSpectrum *>(old_task.get());
  if (old_angular == nullptr || input_descs.size() != 1) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_angular->idesc;
  auto        settings  = jsettings.get<AngularSpectrumSettings>();

  bool same_settings = settings == old_angular->settings;
  bool same_shape    = (new_idesc.shape == old_idesc.shape);
  bool same_strides  = (new_idesc.strides == old_idesc.strides);
  bool same_dtype    = (new_idesc.dtype == old_idesc.dtype);
  bool same_mem_loc  = (new_idesc.mem_loc == old_idesc.mem_loc);
  bool can_reuse     = same_settings && same_shape && same_strides && same_dtype && same_mem_loc;

  if (can_reuse) {
    old_angular->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
