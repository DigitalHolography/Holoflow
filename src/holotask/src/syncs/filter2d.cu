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

#include "holotask/syncs/filter2d.hh"

#include <cstdlib>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
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

void to_json(nlohmann::json &j, const Filter2DSettings &f) {
  j = nlohmann::json{
      {"r_inner", f.r_inner},
      {"r_outer", f.r_outer},
      {"s_inner", f.s_inner},
      {"s_outer", f.s_outer},
  };
}

void from_json(const nlohmann::json &j, Filter2DSettings &f) {
  j.at("r_inner").get_to(f.r_inner);
  j.at("r_outer").get_to(f.r_outer);
  j.at("s_inner").get_to(f.s_inner);
  j.at("s_outer").get_to(f.s_outer);
}

namespace {

struct ApplyFilterCallerInfo {
  cuFloatComplex *filter;
};

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[Filter2DFactory::infer] error: {}", msg);
    throw std::invalid_argument("Filter2DFactory inference error: " + msg);
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
    logger()->error("[Filter2D] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

std::vector<char> apply_filter_lto(int filter_width, int height) {
  std::string src = "#define FILTER_WIDTH " + std::to_string(filter_width) + "ull\n" +
                    "#define HEIGHT " + std::to_string(height) + "ull\n" +
                    "#define FILTER_SIZE (FILTER_WIDTH * HEIGHT)\n";

  src += R"(
#include <cuComplex.h>

struct ApplyFilterCallerInfo {
  cuFloatComplex *filter;
};

__device__ cuFloatComplex apply_filter_callback(
    void *data, size_t offset, void *callerInfo, void *sharedPtr) {
  auto  *info       = (ApplyFilterCallerInfo *)callerInfo;
  size_t filter_idx = offset % FILTER_SIZE;
  auto   val        = ((cuFloatComplex *)data)[offset];

  return cuCmulf(val, info->filter[filter_idx]);
}
)";

  return compile_source_to_lto(src, "apply_filter_callback.cu");
}

// -------------------------------------------------------------------------------------------------
// Device kernels
// -------------------------------------------------------------------------------------------------

__global__ void apply_filter_2d_kernel(cuFloatComplex *filter, const uint32_t width,
                                       const uint32_t height, const uint32_t r_inner,
                                       const uint32_t r_outer, const uint32_t smooth_inner,
                                       const uint32_t smooth_outer) {
  const uint32_t x   = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y   = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t idx = y * width + x;
  if (x >= width || y >= height) {
    return;
  }

  const float r_x  = static_cast<float>(x) - static_cast<float>(width) / 2;
  const float r_y  = static_cast<float>(y) - static_cast<float>(height) / 2;
  const float dist = hypotf(r_x, r_y);

  const float f_r_inner      = static_cast<float>(r_inner);
  const float f_r_outer      = static_cast<float>(r_outer);
  const float f_smooth_inner = static_cast<float>(smooth_inner);
  const float f_smooth_outer = static_cast<float>(smooth_outer);

  float a = 0.0f;
  if (dist < f_r_outer) {
    a = 1.0f;
  } else if (dist < f_r_outer + f_smooth_outer) {
    a = cosf((dist - f_r_outer) / f_smooth_outer * CUDART_PI_F / 2);
  }

  float b = 0.0f;
  if (dist < f_r_inner) {
    b = 1.0f;
  } else if (dist < f_r_inner + f_smooth_inner) {
    b = cosf((dist - f_r_inner) / f_smooth_inner * CUDART_PI_F / 2);
  }

  filter[idx].x = a * (1 - b);
  filter[idx].y = a * (1 - b);
}

__global__ void apply_filter_2d_r2c_kernel(cuFloatComplex *filter, const uint32_t width,
                                           const uint32_t height, const uint32_t filter_width,
                                           const uint32_t r_inner, const uint32_t r_outer,
                                           const uint32_t smooth_inner,
                                           const uint32_t smooth_outer) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= filter_width || y >= height) {
    return;
  }

  const int freq_y =
      y <= height / 2 ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(height);
  const float r_x  = static_cast<float>(x);
  const float r_y  = static_cast<float>(freq_y);
  const float dist = hypotf(r_x, r_y);

  const float f_r_inner      = static_cast<float>(r_inner);
  const float f_r_outer      = static_cast<float>(r_outer);
  const float f_smooth_inner = static_cast<float>(smooth_inner);
  const float f_smooth_outer = static_cast<float>(smooth_outer);

  float a = 0.0f;
  if (dist < f_r_outer) {
    a = 1.0f;
  } else if (f_smooth_outer > 0.0f && dist < f_r_outer + f_smooth_outer) {
    a = cosf((dist - f_r_outer) / f_smooth_outer * CUDART_PI_F / 2);
  }

  float b = 0.0f;
  if (dist < f_r_inner) {
    b = 1.0f;
  } else if (f_smooth_inner > 0.0f && dist < f_r_inner + f_smooth_inner) {
    b = cosf((dist - f_r_inner) / f_smooth_inner * CUDART_PI_F / 2);
  }

  filter[y * filter_width + x] = make_cuComplex(a * (1 - b), 0.0f);
}

__global__ void swap_corners_kernel(cuFloatComplex *in, cuFloatComplex *out, int width, int height,
                                    int batch_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  const int width_half  = width / 2;
  const int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch_size) {
    return;
  }

  const int       batch_offset = z * width * height;
  cuFloatComplex *in_frame     = in + batch_offset;
  cuFloatComplex *out_frame    = out + batch_offset;

  const int top_left_idx     = x + y * width;
  const int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  cuFloatComplex tmp          = in_frame[top_left_idx];
  out_frame[top_left_idx]     = in_frame[bottom_right_idx];
  out_frame[bottom_right_idx] = tmp;

  const int top_right_idx   = (x + width_half) + y * width;
  const int bottom_left_idx = x + (y + height_half) * width;

  tmp                        = in_frame[top_right_idx];
  out_frame[top_right_idx]   = in_frame[bottom_left_idx];
  out_frame[bottom_left_idx] = tmp;
}

DevPtr<cuFloatComplex> make_filter(int width, int height, const Filter2DSettings &settings,
                                   bool r2c) {
  const int filter_width = r2c ? width / 2 + 1 : width;
  auto      d_filter =
      curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(filter_width) * height);

  dim3 block_size(16, 16);
  dim3 grid_size((filter_width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  CUDA_CHECK(cudaMemset(d_filter.get(), 0,
                        static_cast<size_t>(filter_width) * height * sizeof(cuFloatComplex)));

  if (r2c) {
    apply_filter_2d_r2c_kernel<<<grid_size, block_size>>>(
        d_filter.get(), width, height, filter_width, settings.r_inner, settings.r_outer,
        settings.s_inner, settings.s_outer);
  } else {
    apply_filter_2d_kernel<<<grid_size, block_size>>>(d_filter.get(), width, height,
                                                      settings.r_inner, settings.r_outer,
                                                      settings.s_inner, settings.s_outer);
    swap_corners_kernel<<<grid_size, block_size>>>(d_filter.get(), d_filter.get(), width, height,
                                                   1);
  }

  CUDA_CHECK(cudaPeekAtLastError());
  return d_filter;
}

// -------------------------------------------------------------------------------------------------
// Filter2D task implementation
// -------------------------------------------------------------------------------------------------

class Filter2D : public holoflow::core::ISyncTask {
public:
  Filter2D(Filter2DSettings settings, holoflow::core::TDesc idesc, curaii::CufftHandle &&fwd_plan,
           curaii::CufftHandle &&inv_plan, DevPtr<cuFloatComplex> &&d_filter,
           DevPtr<cuFloatComplex> &&d_spectrum, DevPtr<void> &&d_caller_info,
           std::vector<char> &&filter_lto)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), fwd_plan_(std::move(fwd_plan)),
        inv_plan_(std::move(inv_plan)), d_filter_(std::move(d_filter)),
        d_spectrum_(std::move(d_spectrum)), d_caller_info_(std::move(d_caller_info)),
        filter_lto_(std::move(filter_lto)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (idesc_.dtype == holoflow::core::DType::F32) {
      auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data());
      auto *odata = reinterpret_cast<float *>(ctx.outputs[0].data());
      CUFFT_CHECK(cufftXtExec(fwd_plan_.get(), idata, d_spectrum_.get(), CUFFT_FORWARD));
      CUFFT_CHECK(cufftXtExec(inv_plan_.get(), d_spectrum_.get(), odata, CUFFT_INVERSE));
    } else {
      auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
      auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data());
      CUFFT_CHECK(cufftXtExec(fwd_plan_.get(), idata, idata, CUFFT_FORWARD));
      CUFFT_CHECK(cufftXtExec(inv_plan_.get(), idata, odata, CUFFT_INVERSE));
    }

    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) {
    CUFFT_CHECK(cufftSetStream(fwd_plan_.get(), stream));
    CUFFT_CHECK(cufftSetStream(inv_plan_.get(), stream));
  }

  const Filter2DSettings      &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  Filter2DSettings       settings_;
  holoflow::core::TDesc  idesc_;
  curaii::CufftHandle    fwd_plan_;
  curaii::CufftHandle    inv_plan_;
  DevPtr<cuFloatComplex> d_filter_;
  DevPtr<cuFloatComplex> d_spectrum_;
  DevPtr<void>           d_caller_info_;
  std::vector<char>      filter_lto_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Filter2DFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
Filter2DFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<Filter2DSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() >= 2, "input must be a tensor of rank 2 or higher");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32,
        "input must be complex float32 or real float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(settings.r_inner >= 0, "r_inner must be non-negative");
  check(settings.r_outer >= 0, "r_outer must be non-negative");
  check(settings.s_inner >= 0, "s_inner must be non-negative");
  check(settings.s_outer >= 0, "s_outer must be non-negative");

  holoflow::core::TDesc                odesc = idesc;
  std::vector<holoflow::core::InPlace> in_place;
  if (idesc.dtype == holoflow::core::DType::CF32) {
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
Filter2DFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)this->infer(input_descs, jsettings);
  const auto  settings    = jsettings.get<Filter2DSettings>();
  const auto &idesc       = input_descs[0];
  const auto  tensor_rank = idesc.rank();
  const bool  is_real     = idesc.dtype == holoflow::core::DType::F32;

  int batch_size = 1;
  for (size_t i = 0; i + 2 < tensor_rank; ++i) {
    batch_size *= static_cast<int>(idesc.shape[i]);
  }
  const int height = static_cast<int>(idesc.shape[tensor_rank - 2]);
  const int width  = static_cast<int>(idesc.shape[tensor_rank - 1]);

  const int              filter_width = is_real ? width / 2 + 1 : width;
  auto                   d_filter     = make_filter(width, height, settings, is_real);
  DevPtr<cuFloatComplex> d_spectrum;
  if (is_real) {
    d_spectrum = curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(batch_size) *
                                                                height * filter_width);
  }

  ApplyFilterCallerInfo info{
      .filter = d_filter.get(),
  };
  auto d_info = curaii::make_unique_device_ptr<ApplyFilterCallerInfo>(1);
  CUDA_CHECK(
      cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream));

  auto filter_lto = apply_filter_lto(filter_width, height);

  constexpr int rank              = 2;
  long long int n[2]              = {height, width};
  long long int inembed[2]        = {height, width};
  long long int onembed[2]        = {height, width};
  constexpr int istride           = 1;
  const int     idist             = height * width;
  constexpr int ostride           = 1;
  const int     odist             = height * width;
  const int     spectrum_dist     = height * filter_width;
  long long int spectrum_embed[2] = {height, filter_width};
  const int     batch             = batch_size;
  size_t        work_size         = 0;

  curaii::CufftHandle fwd_plan;
  curaii::CufftHandle inv_plan;
  CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "apply_filter_callback", filter_lto.data(),
                                    filter_lto.size(), CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  if (is_real) {
    CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, CUDA_R_32F,
                                    spectrum_embed, ostride, spectrum_dist, CUDA_C_32F, batch,
                                    &work_size, CUDA_C_32F));

    CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, spectrum_embed, istride, spectrum_dist,
                                    CUDA_C_32F, onembed, ostride, odist, CUDA_R_32F, batch,
                                    &work_size, CUDA_C_32F));
  } else {
    CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, CUDA_C_32F,
                                    onembed, ostride, odist, CUDA_C_32F, batch, &work_size,
                                    CUDA_C_32F));

    CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, inembed, istride, idist, CUDA_C_32F,
                                    onembed, ostride, odist, CUDA_C_32F, batch, &work_size,
                                    CUDA_C_32F));
  }

  return std::make_unique<Filter2D>(settings, idesc, std::move(fwd_plan), std::move(inv_plan),
                                    std::move(d_filter), std::move(d_spectrum), std::move(d_info),
                                    std::move(filter_lto));
}

std::unique_ptr<holoflow::core::ISyncTask>
Filter2DFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)this->infer(input_descs, jsettings);

  auto *old_filter = dynamic_cast<Filter2D *>(old_task.get());
  if (old_filter == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_filter->idesc();
  const auto  settings  = jsettings.get<Filter2DSettings>();
  const bool can_reuse = settings == old_filter->settings() && new_idesc.shape == old_idesc.shape &&
                         new_idesc.strides == old_idesc.strides &&
                         new_idesc.dtype == old_idesc.dtype &&
                         new_idesc.mem_loc == old_idesc.mem_loc;

  if (can_reuse) {
    old_filter->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
