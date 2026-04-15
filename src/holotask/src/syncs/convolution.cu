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

#include "holotask/syncs/convolution.hh"

#include <filesystem>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "logger.hh"

namespace holotask::syncs {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ConvolutionSettings &s) {
  j = nlohmann::json{{"kernel_file", s.kernel_file}, {"divide", s.divide}};
}

void from_json(const nlohmann::json &j, ConvolutionSettings &s) {
  j.at("kernel_file").get_to(s.kernel_file);
  j.at("divide").get_to(s.divide);
}

namespace {

struct ComplexMultiplyCallerInfo {
  int           size;
  cufftComplex *freq_kernel;
  float         scale;
};

struct FFTConvolutionData {
  unsigned int padded_width;
  unsigned int padded_height;

  DevPtr<cufftComplex>              d_padded_input;
  DevPtr<cufftComplex>              d_padded_kernel;
  DevPtr<cufftComplex>              d_freq_input;
  DevPtr<cufftComplex>              d_freq_kernel;
  DevPtr<ComplexMultiplyCallerInfo> d_callback_info;
  curaii::CufftHandle               inv_plan;
  curaii::CufftHandle               fft_plan;
};

struct ParsedKernel {
  int                width;
  int                height;
  std::vector<float> values;
};

void check(bool condition, const std::string &message) {
  if (!condition) {
    logger()->error("[ConvolutionFactory::infer] error: {}", message);
    throw std::invalid_argument("ConvolutionFactory inference error: " + message);
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

ParsedKernel load_kernel(const std::string &kernel_file) {
  std::ifstream file(kernel_file);
  check(file.is_open(), "Given kernel file does not exist");

  nlohmann::json j;
  file >> j;

  check(j.contains("kernel"), "JSON must contain field 'kernel'");
  check(j["kernel"].is_array(), "Kernel must be an array");

  const auto &kernel_json = j.at("kernel");
  check(!kernel_json.empty(), "Kernel must not be empty");
  check(kernel_json[0].is_array(), "Kernel rows must be arrays");
  check(!kernel_json[0].empty(), "Kernel must have at least one column");

  const int kernel_height = static_cast<int>(kernel_json.size());
  const int kernel_width  = static_cast<int>(kernel_json[0].size());

  std::vector<float> values;
  values.reserve(static_cast<size_t>(kernel_width * kernel_height));
  for (const auto &row : kernel_json) {
    check(row.is_array(), "Kernel rows must be arrays");
    check(static_cast<int>(row.size()) == kernel_width,
          "All rows of the kernel must have the same width");
    for (const auto &val : row) {
      values.push_back(val.get<float>());
    }
  }

  return ParsedKernel{
      .width  = kernel_width,
      .height = kernel_height,
      .values = std::move(values),
  };
}

// -------------------------------------------------------------------------------------------------
// Helper functions for NVRTC compilation
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
  std::string              cuda_path{CUDA_PATH};
  std::vector<std::string> args{
      "-I" + cuda_path + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };
  return args;
}

std::vector<char> compile_source_to_lto(const std::string &source, const std::string &name) {
  auto                 args_string = get_nvrtc_args();
  curaii::NvrtcProgram prog(source.c_str(), name.c_str(), 0, nullptr, nullptr);
  std::vector<char *>  args;
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
    logger()->error("[Convolution] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

std::vector<char> complex_multiply_callback_lto() {
  std::string callback_source = R"(
  #include <cufft.h>
  
  struct ComplexMultiplyCallerInfo {
    int             size;
    cufftComplex   *freq_kernel;
    float           scale;
  };
  
  __device__ cufftComplex complex_multiply_callback(
      void *data, size_t offset, void *callerInfo, void *sharedPtr) {
    auto *info = (ComplexMultiplyCallerInfo *)callerInfo;
    auto *freq_input = (cufftComplex *)data;
    auto *freq_kernel = info->freq_kernel;
    float scale = info->scale;
    
    cufftComplex a = freq_input[offset];
    cufftComplex b = freq_kernel[offset];
    
    cufftComplex result;
    result.x = (a.x * b.x - a.y * b.y) * scale;
    result.y = (a.x * b.y + a.y * b.x) * scale;
    
    return result;
  }
  )";
  return compile_source_to_lto(callback_source, "complex_multiply_callback.cu");
}

__global__ void pad_kernel(const float *__restrict__ kernel, cufftComplex *__restrict__ padded,
                           const int kernel_width, const int kernel_height, const int padded_width,
                           const int padded_height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= padded_width || y >= padded_height)
    return;

  const int idx = y * padded_width + x;

  const int x_offset = (padded_width - kernel_width) / 2;
  const int y_offset = (padded_height - kernel_height) / 2;

  if (x >= x_offset && x < x_offset + kernel_width && y >= y_offset &&
      y < y_offset + kernel_height) {
    const int kx  = x - x_offset;
    const int ky  = y - y_offset;
    padded[idx].x = kernel[ky * kernel_width + kx];
    padded[idx].y = 0.0f;
  } else {
    padded[idx].x = 0.0f;
    padded[idx].y = 0.0f;
  }
}

__global__ void extract_real_kernel(const cufftComplex *__restrict__ input,
                                    float *__restrict__ output, const int output_width,
                                    const int output_height, const int padded_width) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= output_width || y >= output_height)
    return;

  const int shift_x = (x + output_width / 2) % output_width;
  const int shift_y = (y + output_height / 2) % output_height;

  const int src_idx = shift_y * padded_width + shift_x;
  const int dst_idx = y * output_width + x;

  float value     = input[src_idx].x;
  output[dst_idx] = value;
}

__global__ void divide_kernel(const float *a, const float *b, float *c, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  c[idx] = a[idx] / b[idx];
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Convolution task implementation
// -------------------------------------------------------------------------------------------------

class Convolution : public holoflow::core::ISyncTask {
public:
  Convolution(ConvolutionSettings settings, holoflow::core::TDesc input_desc,
              holoflow::core::TDesc output_desc, cudaStream_t stream, DevPtr<float> &&d_kernel,
              int kernel_width, int kernel_height,
              std::filesystem::file_time_type kernel_last_write_time)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)),
        output_desc_(std::move(output_desc)), stream_(stream), d_kernel_(std::move(d_kernel)),
        kernel_width_(kernel_width), kernel_height_(kernel_height),
        kernel_last_write_time_(kernel_last_write_time) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
      return holoflow::core::OpResult::Cancelled;
    }

    if (ctx.inputs.empty() || ctx.outputs.empty()) {
      return holoflow::core::OpResult::NotReady;
    }

    auto &input_view  = ctx.inputs[0];
    auto &output_view = ctx.outputs[0];

    float *input_data  = reinterpret_cast<float *>(input_view.data());
    float *output_data = reinterpret_cast<float *>(output_view.data());

    const auto input_width  = static_cast<unsigned int>(input_desc_.shape.back());
    const auto input_height = static_cast<unsigned int>(input_desc_.shape[input_desc_.rank() - 2]);
    const auto output_width = static_cast<unsigned int>(output_desc_.shape.back());
    const auto output_height =
        static_cast<unsigned int>(output_desc_.shape[output_desc_.rank() - 2]);

    dim3 block_dim(16, 16);

    dim3 grid_dim_pad((fft_data_->padded_width + block_dim.x - 1) / block_dim.x,
                      (fft_data_->padded_height + block_dim.y - 1) / block_dim.y);

    pad_kernel<<<grid_dim_pad, block_dim, 0, stream_>>>(
        input_data, fft_data_->d_padded_input.get(), input_width, input_height,
        fft_data_->padded_width, fft_data_->padded_height);

    CUDA_CHECK(cudaGetLastError());

    const int freq_size = static_cast<int>(fft_data_->padded_width * fft_data_->padded_height);
    const float scale   = 1.0f / static_cast<float>(fft_data_->padded_width * fft_data_->padded_height);

    const ComplexMultiplyCallerInfo info{freq_size, fft_data_->d_freq_kernel.get(), scale};
    CUDA_CHECK(cudaMemcpy(fft_data_->d_callback_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

    cufftExecC2C(fft_data_->fft_plan.get(), fft_data_->d_padded_input.get(),
                 fft_data_->d_freq_input.get(), CUFFT_FORWARD);
    cufftExecC2C(fft_data_->inv_plan.get(), fft_data_->d_freq_input.get(),
                 fft_data_->d_padded_input.get(), CUFFT_INVERSE);

    dim3 grid_dim_extract((output_width + block_dim.x - 1) / block_dim.x,
                          (output_height + block_dim.y - 1) / block_dim.y);

    extract_real_kernel<<<grid_dim_extract, block_dim, 0, stream_>>>(
        fft_data_->d_padded_input.get(), output_data, output_width, output_height,
        fft_data_->padded_width);

    if (settings_.divide) {
      const int num_elements = static_cast<int>(output_width * output_height);
      constexpr int threads  = 256;
      const int blocks       = (num_elements + threads - 1) / threads;

      divide_kernel<<<blocks, threads, 0, stream_>>>(input_data, output_data, output_data,
                                                     num_elements);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return holoflow::core::OpResult::Ok;
  }

  void create_fft_data() {
    const auto input_width  = static_cast<unsigned int>(input_desc_.shape.back());
    const auto input_height = static_cast<unsigned int>(input_desc_.shape[input_desc_.rank() - 2]);

    fft_data_ = std::make_unique<FFTConvolutionData>();
    fft_data_->padded_width  = input_width;
    fft_data_->padded_height = input_height;

    const size_t padded_size = static_cast<size_t>(fft_data_->padded_width) * fft_data_->padded_height;
    fft_data_->d_padded_input  = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_padded_kernel = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_freq_input    = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_freq_kernel   = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_callback_info = curaii::make_unique_device_ptr<ComplexMultiplyCallerInfo>(1);

    dim3 block_dim(16, 16);
    dim3 grid_dim((fft_data_->padded_width + block_dim.x - 1) / block_dim.x,
                  (fft_data_->padded_height + block_dim.y - 1) / block_dim.y);

    pad_kernel<<<grid_dim, block_dim, 0, stream_>>>(
        d_kernel_.get(), fft_data_->d_padded_kernel.get(), kernel_width_, kernel_height_,
        fft_data_->padded_width, fft_data_->padded_height);

    CUDA_CHECK(cudaGetLastError());
  }

  void update_stream(cudaStream_t stream) {
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(fft_data_->fft_plan.get(), stream_));
    CUFFT_CHECK(cufftSetStream(fft_data_->inv_plan.get(), stream_));
  }

  const ConvolutionSettings &settings() const { return settings_; }
  const holoflow::core::TDesc &input_desc() const { return input_desc_; }
  const std::filesystem::file_time_type &kernel_last_write_time() const {
    return kernel_last_write_time_;
  }
  std::vector<char> &lto() { return lto_; }
  FFTConvolutionData &fft_data() { return *fft_data_; }
  cudaStream_t stream() const { return stream_; }

private:
  ConvolutionSettings                settings_;
  holoflow::core::TDesc              input_desc_;
  holoflow::core::TDesc              output_desc_;
  cudaStream_t                       stream_;
  DevPtr<float>                      d_kernel_;
  int                                kernel_width_;
  int                                kernel_height_;
  std::vector<char>                  lto_;
  std::filesystem::file_time_type    kernel_last_write_time_;
  std::unique_ptr<FFTConvolutionData> fft_data_;
};

// -------------------------------------------------------------------------------------------------
// ConvolutionFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ConvolutionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  check(!input_descs.empty(), "No input descriptors provided");
  check(input_descs.size() == 1, "Convolution expects exactly one input");

  const auto &input_desc = input_descs[0];
  const auto  settings   = jsettings.get<ConvolutionSettings>();

  check(input_desc.rank() == 2, "Input must be a rank-2 tensor");
  check(input_desc.dtype == holoflow::core::DType::F32, "Input must be F32 type");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "Input must be in device memory");
  check(is_c_contiguous(input_desc), "Input tensor must be C-contiguous");

  (void)load_kernel(settings.kernel_file);

  holoflow::core::TDesc output_desc = input_desc;

  return holoflow::core::InferResult{
      .input_descs   = {input_desc},
      .output_descs  = {output_desc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ConvolutionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings,
                           const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res       = infer(input_descs, jsettings);
  const auto settings        = jsettings.get<ConvolutionSettings>();
  const auto &input_desc     = input_descs[0];
  const auto parsed_kernel   = load_kernel(settings.kernel_file);
  const auto last_write_time = std::filesystem::last_write_time(settings.kernel_file);

  auto d_kernel = curaii::make_unique_device_ptr<float>(parsed_kernel.values.size());
  CUDA_CHECK(cudaMemcpyAsync(d_kernel.get(), parsed_kernel.values.data(),
                             parsed_kernel.values.size() * sizeof(float), cudaMemcpyHostToDevice,
                             ctx.stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  auto task = std::make_unique<Convolution>(
      settings, input_desc, infer_res.output_descs[0], ctx.stream, std::move(d_kernel),
      parsed_kernel.width, parsed_kernel.height, last_write_time);

  task->create_fft_data();
  task->lto() = complex_multiply_callback_lto();

  constexpr int rank = 2;
  long long int n[2]       = {static_cast<long long>(task->fft_data().padded_height),
                              static_cast<long long>(task->fft_data().padded_width)};
  long long int inembed[2] = {static_cast<long long>(task->fft_data().padded_height),
                              static_cast<long long>(task->fft_data().padded_width)};
  long long int onembed[2] = {static_cast<long long>(task->fft_data().padded_height),
                              static_cast<long long>(task->fft_data().padded_width)};
  constexpr int istride       = 1;
  const int     idist         = static_cast<int>(task->fft_data().padded_height * task->fft_data().padded_width);
  constexpr int ostride       = 1;
  const int     odist         = idist;
  constexpr int batch         = 1;
  size_t        work_size     = 0;
  auto          inputtype     = CUDA_C_32F;
  auto          outputtype    = CUDA_C_32F;
  auto          executiontype = CUDA_C_32F;

  CUFFT_CHECK(cufftSetStream(task->fft_data().fft_plan.get(), task->stream()));
  CUFFT_CHECK(cufftSetStream(task->fft_data().inv_plan.get(), task->stream()));

  auto *d_info_ptr = reinterpret_cast<void *>(task->fft_data().d_callback_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(task->fft_data().inv_plan.get(), "complex_multiply_callback",
                                    task->lto().data(), task->lto().size(), CUFFT_CB_LD_COMPLEX,
                                    &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(task->fft_data().fft_plan.get(), rank, n, inembed, istride, idist,
                                  inputtype, onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));
  CUFFT_CHECK(cufftXtMakePlanMany(task->fft_data().inv_plan.get(), rank, n, inembed, istride, idist,
                                  inputtype, onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  cufftExecC2C(task->fft_data().fft_plan.get(), task->fft_data().d_padded_kernel.get(),
               task->fft_data().d_freq_kernel.get(), CUFFT_FORWARD);
  CUDA_CHECK(cudaStreamSynchronize(task->stream()));

  return task;
}

std::unique_ptr<holoflow::core::ISyncTask>
ConvolutionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                           std::span<const holoflow::core::TDesc>     input_descs,
                           const nlohmann::json                      &jsettings,
                           const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_conv = dynamic_cast<Convolution *>(old_task.get());
  if (old_conv == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto  settings           = jsettings.get<ConvolutionSettings>();
  const auto &new_input_desc     = input_descs[0];
  const auto &old_input_desc     = old_conv->input_desc();
  const auto  current_write_time = std::filesystem::last_write_time(settings.kernel_file);
  const bool  kernel_changed     = current_write_time != old_conv->kernel_last_write_time();
  const bool  can_reuse          = !kernel_changed &&
                          settings == old_conv->settings() &&
                          new_input_desc.shape == old_input_desc.shape &&
                          new_input_desc.strides == old_input_desc.strides &&
                          new_input_desc.dtype == old_input_desc.dtype &&
                          new_input_desc.mem_loc == old_input_desc.mem_loc;

  if (can_reuse) {
    old_conv->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
