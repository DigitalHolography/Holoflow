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

void to_json(nlohmann::json &j, const AngularSpectrumSettings::Padding &p) {
  j = {
      {"width", p.width},
      {"height", p.height},
  };
}

void from_json(const nlohmann::json &j, AngularSpectrumSettings::Padding &p) {
  j.at("width").get_to(p.width);
  j.at("height").get_to(p.height);
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
  if (as.padding.has_value()) {
    j["padding"] = as.padding.value();
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
  if (j.contains("padding")) {
    as.padding = AngularSpectrumSettings::Padding{};
    j.at("padding").get_to(as.padding.value());
  } else {
    as.padding = std::nullopt;
  }
}

// -------------------------------------------------------------------------------------------------
// Private implementation types
// -------------------------------------------------------------------------------------------------

namespace {

// Minimized CallerInfo struct. Dimensions are injected via macros.
struct ApplyTransferFunctionCallerInfo {
  cuFloatComplex *transfer_function;
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

std::vector<char> load_input_lto(bool is_real, int input_width, int input_height, int output_width,
                                 int output_height) {
  std::string src = "#define ANGULAR_SPECTRUM_INPUT_IS_REAL " + std::to_string(is_real ? 1 : 0) +
                    "\n" + "#define INPUT_WIDTH " + std::to_string(input_width) + "ull\n" +
                    "#define INPUT_HEIGHT " + std::to_string(input_height) + "ull\n" +
                    "#define OUTPUT_WIDTH " + std::to_string(output_width) + "ull\n" +
                    "#define OUTPUT_HEIGHT " + std::to_string(output_height) + "ull\n" +
                    "#define INPUT_PLANE_SIZE (INPUT_WIDTH * INPUT_HEIGHT)\n" +
                    "#define OUTPUT_PLANE_SIZE (OUTPUT_WIDTH * OUTPUT_HEIGHT)\n";

  src += R"(
#include <cuComplex.h>

__device__ cuFloatComplex load_angular_spectrum_input_callback(
    void *data, unsigned long long offset, void *callerInfo, void *sharedPtr) {
  const unsigned long long batch = offset / OUTPUT_PLANE_SIZE;
  const unsigned long long plane_offset = offset % OUTPUT_PLANE_SIZE;
  const unsigned long long output_x = plane_offset % OUTPUT_WIDTH;
  const unsigned long long output_y = plane_offset / OUTPUT_WIDTH;
  const unsigned long long offset_x = (OUTPUT_WIDTH - INPUT_WIDTH) / 2;
  const unsigned long long offset_y = (OUTPUT_HEIGHT - INPUT_HEIGHT) / 2;

  if (output_x < offset_x || output_x >= offset_x + INPUT_WIDTH ||
      output_y < offset_y || output_y >= offset_y + INPUT_HEIGHT) {
    return make_cuComplex(0.0f, 0.0f);
  }

  const unsigned long long input_x = output_x - offset_x;
  const unsigned long long input_y = output_y - offset_y;
  const unsigned long long input_offset =
      batch * INPUT_PLANE_SIZE + input_y * INPUT_WIDTH + input_x;
#if ANGULAR_SPECTRUM_INPUT_IS_REAL
  return make_cuComplex(((float *)data)[input_offset], 0.0f);
#else
  return ((cuFloatComplex *)data)[input_offset];
#endif
}
)";

  return compile_source_to_lto(src, "load_angular_spectrum_input_callback.cu");
}

std::vector<char> apply_transfer_function_lto(int width, int height) {
  std::string src = "#define WIDTH " + std::to_string(width) + "ull\n" + "#define HEIGHT " +
                    std::to_string(height) + "ull\n" +
                    "#define TRANSFER_FUNCTION_SIZE (WIDTH * HEIGHT)\n";

  // Common header — struct definition shared with the host side
  src += R"(
#include <cuComplex.h>

struct ApplyTransferFunctionCallerInfo {
  cuFloatComplex *transfer_function;
};

__device__ cuFloatComplex apply_transfer_function_callback(
    void *data, unsigned long long offset, void *callerInfo, void *sharedPtr) {
  auto *info = (ApplyTransferFunctionCallerInfo *)callerInfo;
  const unsigned long long transfer_idx = offset % TRANSFER_FUNCTION_SIZE;
  const auto val = ((cuFloatComplex *)data)[offset];

  return cuCmulf(val, info->transfer_function[transfer_idx]);
}
)";

  return compile_source_to_lto(src, "apply_transfer_function_callback.cu");
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

__global__ void multiply_filter_2d_kernel(cuFloatComplex *transfer_function, const uint32_t width,
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

  const float weight = val * val;
  transfer_function[idx].x *= weight;
  transfer_function[idx].y *= weight;
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

DevPtr<cuFloatComplex> make_transfer_function(int width, int height,
                                              const AngularSpectrumSettings &settings) {
  auto d_transfer_function =
      curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(width) * height);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  spectral_lens_kernel<<<grid, block>>>(d_transfer_function.get(), width, height, settings.lambda,
                                        settings.z, settings.dx);

  if (settings.filter.has_value()) {
    const auto &f = settings.filter.value();
    multiply_filter_2d_kernel<<<grid, block>>>(
        d_transfer_function.get(), static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        static_cast<uint32_t>(f.r_inner), static_cast<uint32_t>(f.r_outer),
        static_cast<uint32_t>(f.s_inner), static_cast<uint32_t>(f.s_outer));
  }

  swap_corners_kernel<<<grid, block>>>(d_transfer_function.get(), d_transfer_function.get(), width,
                                       height, 1);
  CUDA_CHECK(cudaGetLastError());
  return d_transfer_function;
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
  DevPtr<cuFloatComplex> d_transfer_function;
  DevPtr<void>           d_caller_info;
  std::vector<char>      load_lto;
  std::vector<char>      transfer_function_lto;

  // -- ISyncTask interface ------------------------------------------------------------------------
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());

    if (settings.padding.has_value()) {
      if (idesc.dtype == holoflow::core::DType::F32) {
        auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data());
        CUFFT_CHECK(cufftXtExec(fwd_plan.get(), idata, odata, CUFFT_FORWARD));
      } else {
        auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data());
        CUFFT_CHECK(cufftXtExec(fwd_plan.get(), idata, odata, CUFFT_FORWARD));
      }
      CUFFT_CHECK(cufftXtExec(inv_plan.get(), odata, odata, CUFFT_INVERSE));
    } else if (idesc.dtype == holoflow::core::DType::F32) {
      auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data());
      CUFFT_CHECK(cufftXtExec(fwd_plan.get(), idata, odata, CUFFT_FORWARD));
      CUFFT_CHECK(cufftXtExec(inv_plan.get(), odata, odata, CUFFT_INVERSE));
    } else {
      auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data());
      CUFFT_CHECK(cufftXtExec(fwd_plan.get(), idata, idata, CUFFT_FORWARD));
      CUFFT_CHECK(cufftXtExec(inv_plan.get(), idata, odata, CUFFT_INVERSE));
    }

    return holoflow::core::OpResult::Ok;
  }

  // -- Update utilities ---------------------------------------------------------------------------
  void update_stream(cudaStream_t new_stream) {
    CUFFT_CHECK(cufftSetStream(fwd_plan.get(), new_stream));
    CUFFT_CHECK(cufftSetStream(inv_plan.get(), new_stream));
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
  check(idesc.rank() >= 2, "input must be a tensor of rank 2 or higher");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32, "input must be complex float32 or real float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");
  if (settings.padding.has_value()) {
    const auto &padding = settings.padding.value();
    const auto  height  = static_cast<int64_t>(idesc.shape[idesc.rank() - 2]);
    const auto  width   = static_cast<int64_t>(idesc.shape[idesc.rank() - 1]);
    check(padding.width > 0, "padding width must be positive");
    check(padding.height > 0, "padding height must be positive");
    check(padding.width >= width, "padding width must be at least the input width");
    check(padding.height >= height, "padding height must be at least the input height");
    check((padding.width - width) % 2 == 0, "horizontal padding must be even");
    check((padding.height - height) % 2 == 0, "vertical padding must be even");
  }
  // clang-format on

  auto output_shape = idesc.shape;
  if (settings.padding.has_value()) {
    output_shape[output_shape.size() - 2] = static_cast<size_t>(settings.padding->height);
    output_shape[output_shape.size() - 1] = static_cast<size_t>(settings.padding->width);
  }
  holoflow::core::TDesc odesc(output_shape, holoflow::core::DType::CF32, idesc.mem_loc);
  std::vector<holoflow::core::InPlace> in_place;
  if (idesc.dtype == holoflow::core::DType::CF32 && !settings.padding.has_value()) {
    in_place.push_back({0, 0});
  }

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = std::move(in_place),
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AngularSpectrumFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  auto        infer       = this->infer(input_descs, jsettings);
  auto        settings    = jsettings.get<AngularSpectrumSettings>();
  const auto &idesc       = input_descs[0];
  const auto  tensor_rank = idesc.rank();
  const bool  is_real     = idesc.dtype == holoflow::core::DType::F32;
  (void)infer;

  int B = 1;
  for (size_t i = 0; i + 2 < tensor_rank; ++i) {
    B *= static_cast<int>(idesc.shape[i]);
  }
  const int H  = static_cast<int>(idesc.shape[tensor_rank - 2]);
  const int W  = static_cast<int>(idesc.shape[tensor_rank - 1]);
  const int OH = settings.padding.has_value() ? settings.padding->height : H;
  const int OW = settings.padding.has_value() ? settings.padding->width : W;

  // -- JIT callback -------------------------------------------------------------------------------
  const bool needs_input_callback = is_real || settings.padding.has_value();
  auto       load_lto =
      needs_input_callback ? load_input_lto(is_real, W, H, OW, OH) : std::vector<char>{};
  auto transfer_function_lto = apply_transfer_function_lto(OW, OH);
  auto d_transfer_function   = make_transfer_function(OW, OH, settings);

  ApplyTransferFunctionCallerInfo info{
      .transfer_function = d_transfer_function.get(),
  };
  auto d_info = curaii::make_unique_device_ptr<ApplyTransferFunctionCallerInfo>(1);
  auto e = cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream);
  CUDA_CHECK(e);

  // -- cuFFT plans --------------------------------------------------------------------------------
  int           rank          = 2;
  long long int n[2]          = {OH, OW};
  long long int inembed[2]    = {OH, OW};
  int           istride       = 1;
  int           idist         = OH * OW;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {OH, OW};
  int           ostride       = 1;
  int           odist         = OH * OW;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = B;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle fwd_plan;
  curaii::CufftHandle inv_plan;
  CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  if (needs_input_callback) {
    CUFFT_CHECK(cufftXtSetJITCallback(fwd_plan.get(), "load_angular_spectrum_input_callback",
                                      load_lto.data(), load_lto.size(), CUFFT_CB_LD_COMPLEX,
                                      nullptr));
  }
  CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "apply_transfer_function_callback",
                                    transfer_function_lto.data(), transfer_function_lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Construct task directly
  auto task                   = std::make_unique<AngularSpectrum>();
  task->settings              = settings;
  task->idesc                 = idesc;
  task->fwd_plan              = std::move(fwd_plan);
  task->inv_plan              = std::move(inv_plan);
  task->d_transfer_function   = std::move(d_transfer_function);
  task->d_caller_info         = std::move(d_info);
  task->load_lto              = std::move(load_lto);
  task->transfer_function_lto = std::move(transfer_function_lto);

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
