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

#include "holotask/syncs/flatfield.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "logger.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const FlatfieldSettings &s) {
  j = nlohmann::json{{"sigma_y", s.sigma_y}, {"sigma_x", s.sigma_x}};
}

void from_json(const nlohmann::json &j, FlatfieldSettings &s) {
  j.at("sigma_y").get_to(s.sigma_y);
  j.at("sigma_x").get_to(s.sigma_x);
}

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[FlatfieldFactory::infer] error: {}", msg);
    throw std::invalid_argument("FlatfieldFactory inference error: " + msg);
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

std::vector<float> make_gaussian_kernel(float sigma) {
  const int radius = static_cast<int>(std::ceil(3.0f * sigma));
  check(radius > 0, "sigma is too small to form a Gaussian kernel");

  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
  const float        inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);
  float              sum            = 0.0f;

  for (int i = -radius; i <= radius; ++i) {
    const float value                       = std::exp(-static_cast<float>(i * i) * inv_two_sigma2);
    kernel[static_cast<size_t>(i + radius)] = value;
    sum += value;
  }

  for (auto &value : kernel) {
    value /= sum;
  }

  return kernel;
}

std::string get_compute_arch() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<std::string> get_nvrtc_args() {
  auto CUDA_PATH = std::getenv("CUDA_PATH");
  if (CUDA_PATH == nullptr) {
    throw std::runtime_error("CUDA_PATH environment variable not set");
  }

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
    logger()->error("[Flatfield] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

std::vector<char> highpass_callback_lto(int width, int height, float sigma_y, float sigma_x) {
  const int   filter_width = width / 2 + 1;
  const float scale        = 1.0f / static_cast<float>(width * height);

  std::string src = "#define WIDTH " + std::to_string(width) + "ull\n" + "#define HEIGHT " +
                    std::to_string(height) + "ull\n" + "#define FILTER_WIDTH " +
                    std::to_string(filter_width) + "ull\n" + "#define SIGMA_Y " +
                    std::to_string(sigma_y) + "f\n" + "#define SIGMA_X " + std::to_string(sigma_x) +
                    "f\n" + "#define SCALE " + std::to_string(scale) + "f\n";

  src += R"(
#include <cuComplex.h>

__device__ cuFloatComplex flatfield_highpass_callback(
    void *data, size_t offset, void *callerInfo, void *sharedPtr) {
  const size_t local = offset % (HEIGHT * FILTER_WIDTH);
  const size_t y     = local / FILTER_WIDTH;
  const size_t x     = local - y * FILTER_WIDTH;

  const float fx = static_cast<float>(x) / static_cast<float>(WIDTH);
  const float fy_idx =
      y <= HEIGHT / 2 ? static_cast<float>(y)
                      : static_cast<float>(static_cast<long long>(y) -
                                           static_cast<long long>(HEIGHT));
  const float fy = fy_idx / static_cast<float>(HEIGHT);

  constexpr float two_pi_squared = 19.739208802178716f;
  const float lowpass =
      expf(-two_pi_squared * (SIGMA_X * SIGMA_X * fx * fx + SIGMA_Y * SIGMA_Y * fy * fy));
  const float gain = (1.0f - lowpass) * SCALE;
  const auto  val  = reinterpret_cast<cuFloatComplex *>(data)[offset];

  return make_cuFloatComplex(val.x * gain, val.y * gain);
}
)";

  return compile_source_to_lto(src, "flatfield_highpass_callback.cu");
}

bool should_use_fft_flatfield(int height, int width, size_t kernel_y_size, size_t kernel_x_size) {
  // The spatial path preserves the exact clamp-at-edge behavior for small kernels. Large kernels
  // use the analytic Fourier-domain Gaussian high-pass to avoid O(pixels * kernel radius) work.
  constexpr size_t kMinFftKernelSize = 65;
  return height > 1 && width > 1 && std::max(kernel_y_size, kernel_x_size) >= kMinFftKernelSize;
}

__global__ void gaussian_horizontal_kernel(const float *__restrict__ input,
                                           float *__restrict__ temp,
                                           const float *__restrict__ kernel, size_t total,
                                           int height, int width, int radius) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int    plane_size = height * width;
  const int    local      = static_cast<int>(idx % static_cast<size_t>(plane_size));
  const int    y          = local / width;
  const int    x          = local - y * width;
  const size_t base       = idx - static_cast<size_t>(local);

  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const int xk = x + k;
    const int xx = xk < 0 ? 0 : (xk >= width ? width - 1 : xk);
    sum += input[base + y * width + xx] * kernel[k + radius];
  }

  temp[idx] = sum;
}

// Applies the vertical pass on image axis -2 after the horizontal pass on axis -1. All leading
// axes are treated as independent planes, matching scipy's sigma [0, ..., sigma_y, sigma_x].
__global__ void gaussian_subtract_vertical_kernel(const float *__restrict__ input,
                                                  const float *__restrict__ temp,
                                                  float *__restrict__ output,
                                                  const float *__restrict__ kernel, size_t total,
                                                  int height, int width, int radius) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int    plane_size = height * width;
  const int    local      = static_cast<int>(idx % static_cast<size_t>(plane_size));
  const int    y          = local / width;
  const int    x          = local - y * width;
  const size_t base       = idx - static_cast<size_t>(local);

  float background = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const int yk = y + k;
    const int yy = yk < 0 ? 0 : (yk >= height ? height - 1 : yk);
    background += temp[base + yy * width + x] * kernel[k + radius];
  }

  output[idx] = input[idx] - background;
}

class Flatfield : public holoflow::core::ISyncTask {
public:
  Flatfield(FlatfieldSettings settings, holoflow::core::TDesc idesc, size_t total, int height,
            int width, bool use_fft, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), total_(total), height_(height),
        width_(width), use_fft_(use_fft), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data());
    auto *odata = reinterpret_cast<float *>(ctx.outputs[0].data());

    if (use_fft_) {
      CUFFT_CHECK(cufftXtExec(fwd_plan_->get(), idata, d_spectrum_.get(), CUFFT_FORWARD));
      CUFFT_CHECK(cufftXtExec(inv_plan_->get(), d_spectrum_.get(), odata, CUFFT_INVERSE));
      return holoflow::core::OpResult::Ok;
    }

    constexpr int block = 256;
    const int     grid  = static_cast<int>((total_ + block - 1) / block);
    gaussian_horizontal_kernel<<<grid, block, 0, stream_>>>(idata, d_temp_.get(), d_kernel_x_.get(),
                                                            total_, height_, width_, radius_x_);
    gaussian_subtract_vertical_kernel<<<grid, block, 0, stream_>>>(
        idata, d_temp_.get(), odata, d_kernel_y_.get(), total_, height_, width_, radius_y_);

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) {
    stream_ = stream;
    if (use_fft_) {
      CUFFT_CHECK(cufftSetStream(fwd_plan_->get(), stream_));
      CUFFT_CHECK(cufftSetStream(inv_plan_->get(), stream_));
    }
  }

  void set_spatial_data(int radius_y, int radius_x, DevPtr<float> &&d_kernel_y,
                        DevPtr<float> &&d_kernel_x, DevPtr<float> &&d_temp) {
    radius_y_   = radius_y;
    radius_x_   = radius_x;
    d_kernel_y_ = std::move(d_kernel_y);
    d_kernel_x_ = std::move(d_kernel_x);
    d_temp_     = std::move(d_temp);
  }

  void set_fft_data(curaii::CufftHandle &&fwd_plan, curaii::CufftHandle &&inv_plan,
                    DevPtr<cuFloatComplex> &&d_spectrum, std::vector<char> &&highpass_lto) {
    fwd_plan_.emplace(std::move(fwd_plan));
    inv_plan_.emplace(std::move(inv_plan));
    d_spectrum_   = std::move(d_spectrum);
    highpass_lto_ = std::move(highpass_lto);
  }

  const FlatfieldSettings     &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  FlatfieldSettings                  settings_;
  holoflow::core::TDesc              idesc_;
  size_t                             total_;
  int                                height_;
  int                                width_;
  bool                               use_fft_;
  int                                radius_y_ = 0;
  int                                radius_x_ = 0;
  DevPtr<float>                      d_kernel_y_;
  DevPtr<float>                      d_kernel_x_;
  DevPtr<float>                      d_temp_;
  std::optional<curaii::CufftHandle> fwd_plan_;
  std::optional<curaii::CufftHandle> inv_plan_;
  DevPtr<cuFloatComplex>             d_spectrum_;
  std::vector<char>                  highpass_lto_;
  cudaStream_t                       stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// FlatfieldFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
FlatfieldFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<FlatfieldSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() >= 2, "input must be a tensor of rank 2 or higher");
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.num_elements() > 0, "input must not be empty");
  check(settings.sigma_y > 0.0f, "sigma_y must be positive");
  check(settings.sigma_x > 0.0f, "sigma_x must be positive");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FlatfieldFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)this->infer(input_descs, jsettings);
  const auto  settings = jsettings.get<FlatfieldSettings>();
  const auto &idesc    = input_descs[0];
  const auto  rank     = idesc.rank();

  const int  height   = static_cast<int>(idesc.shape[rank - 2]);
  const int  width    = static_cast<int>(idesc.shape[rank - 1]);
  const auto kernel_y = make_gaussian_kernel(settings.sigma_y);
  const auto kernel_x = make_gaussian_kernel(settings.sigma_x);
  const int  radius_y = static_cast<int>(kernel_y.size() / 2);
  const int  radius_x = static_cast<int>(kernel_x.size() / 2);

  int batch = 1;
  for (size_t i = 0; i + 2 < rank; ++i) {
    batch *= static_cast<int>(idesc.shape[i]);
  }

  const bool use_fft = should_use_fft_flatfield(height, width, kernel_y.size(), kernel_x.size());
  auto task = std::make_unique<Flatfield>(settings, idesc, idesc.num_elements(), height, width,
                                          use_fft, ctx.stream);

  if (use_fft) {
    const int filter_width = width / 2 + 1;
    auto highpass_lto = highpass_callback_lto(width, height, settings.sigma_y, settings.sigma_x);
    auto d_spectrum   = curaii::make_unique_device_ptr<cuFloatComplex>(
        static_cast<size_t>(batch) * static_cast<size_t>(height) *
        static_cast<size_t>(filter_width));

    constexpr int rank_2d           = 2;
    long long int n[2]              = {height, width};
    long long int inembed[2]        = {height, width};
    long long int spectrum_embed[2] = {height, filter_width};
    constexpr int istride           = 1;
    const int     idist             = height * width;
    constexpr int ostride           = 1;
    const int     spectrum_dist     = height * filter_width;
    const int     batch_count       = batch;
    size_t        work_size         = 0;

    curaii::CufftHandle fwd_plan;
    curaii::CufftHandle inv_plan;
    CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
    CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

    CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "flatfield_highpass_callback",
                                      highpass_lto.data(), highpass_lto.size(), CUFFT_CB_LD_COMPLEX,
                                      nullptr));

    CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank_2d, n, inembed, istride, idist, CUDA_R_32F,
                                    spectrum_embed, ostride, spectrum_dist, CUDA_C_32F, batch_count,
                                    &work_size, CUDA_C_32F));
    CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank_2d, n, spectrum_embed, istride,
                                    spectrum_dist, CUDA_C_32F, inembed, ostride, idist, CUDA_R_32F,
                                    batch_count, &work_size, CUDA_C_32F));

    task->set_fft_data(std::move(fwd_plan), std::move(inv_plan), std::move(d_spectrum),
                       std::move(highpass_lto));
    return task;
  }

  auto d_kernel_y = curaii::make_unique_device_ptr<float>(kernel_y.size());
  CUDA_CHECK(cudaMemcpyAsync(d_kernel_y.get(), kernel_y.data(), kernel_y.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  auto d_kernel_x = curaii::make_unique_device_ptr<float>(kernel_x.size());
  CUDA_CHECK(cudaMemcpyAsync(d_kernel_x.get(), kernel_x.data(), kernel_x.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  auto d_temp = curaii::make_unique_device_ptr<float>(idesc.num_elements());

  task->set_spatial_data(radius_y, radius_x, std::move(d_kernel_y), std::move(d_kernel_x),
                         std::move(d_temp));
  return task;
}

std::unique_ptr<holoflow::core::ISyncTask>
FlatfieldFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                         std::span<const holoflow::core::TDesc>     input_descs,
                         const nlohmann::json                      &jsettings,
                         const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)this->infer(input_descs, jsettings);

  auto *old_flatfield = dynamic_cast<Flatfield *>(old_task.get());
  if (old_flatfield == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_flatfield->idesc();
  const auto  settings  = jsettings.get<FlatfieldSettings>();
  const bool  can_reuse =
      settings == old_flatfield->settings() && new_idesc.shape == old_idesc.shape &&
      new_idesc.strides == old_idesc.strides && new_idesc.dtype == old_idesc.dtype &&
      new_idesc.mem_loc == old_idesc.mem_loc;

  if (can_reuse) {
    old_flatfield->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
