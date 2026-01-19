#include "holotask/syncs/convolution.hh"

#include <fstream>

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ConvolutionSettings &s) {
  j = nlohmann::json{{"kernel_file", s.kernel_file}, {"divide", s.divide}};
}

void from_json(const nlohmann::json &j, ConvolutionSettings &s) {
  j.at("kernel_file").get_to(s.kernel_file);
  j.at("divide").get_to(s.divide);
}

namespace {

// Helper functions for NVRTC compilation
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

Convolution::Convolution(ConvolutionSettings settings, const holoflow::core::TDesc &input_desc,
                         const holoflow::core::TDesc &output_desc, cudaStream_t stream,
                         DevPtr<float> &&d_kernel, const int kernel_width, const int kernel_height,
                         std::filesystem::file_time_type kernel_last_write_time)
    : settings_(std::move(settings)), input_desc_(input_desc), output_desc_(output_desc),
      stream_(stream), d_kernel_(std::move(d_kernel)), kernel_width_(kernel_width),
      kernel_height_(kernel_height), kernel_last_write_time_(kernel_last_write_time) {}

holoflow::core::OpResult Convolution::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
    return holoflow::core::OpResult::Cancelled;
  }

  if (ctx.inputs.empty() || ctx.outputs.empty()) {
    return holoflow::core::OpResult::NotReady;
  }

  auto &input_view  = ctx.inputs[0];
  auto       &output_view = ctx.outputs[0];

  float *input_data  = reinterpret_cast<float *>(input_view.data());
  float       *output_data = reinterpret_cast<float *>(output_view.data());

  auto input_width = static_cast<unsigned int>(input_desc_.shape.back());
  auto input_height =
      input_desc_.shape.size() > 1
          ? static_cast<unsigned int>(input_desc_.shape[input_desc_.shape.size() - 2])
          : 1;
  auto output_width = static_cast<unsigned int>(output_desc_.shape.back());
  auto output_height =
      output_desc_.shape.size() > 1
          ? static_cast<unsigned int>(output_desc_.shape[output_desc_.shape.size() - 2])
          : 1;

  dim3 block_dim(16, 16);

  dim3 grid_dim_pad((fft_data_->padded_width + block_dim.x - 1) / block_dim.x,
                    (fft_data_->padded_height + block_dim.y - 1) / block_dim.y);

  pad_kernel<<<grid_dim_pad, block_dim, 0, stream_>>>(
      input_data, fft_data_->d_padded_input.get(), input_width, input_height,
      fft_data_->padded_width, fft_data_->padded_height);

  CUDA_CHECK(cudaGetLastError());

  const int freq_size = fft_data_->padded_width * fft_data_->padded_height;
  float     scale     = 1.0f / (fft_data_->padded_width * fft_data_->padded_height);

  ComplexMultiplyCallerInfo info{freq_size, fft_data_->d_freq_kernel.get(), scale};
  CUDA_CHECK(
      cudaMemcpy(fft_data_->d_callback_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

  // Forward FFT
  cufftExecC2C(fft_data_->fft_plan.get(), fft_data_->d_padded_input.get(),
               fft_data_->d_freq_input.get(), CUFFT_FORWARD);

  // Inverse FFT with callback
  cufftExecC2C(fft_data_->inv_plan.get(), fft_data_->d_freq_input.get(),
               fft_data_->d_padded_input.get(), CUFFT_INVERSE);

  dim3 grid_dim_extract((output_width + block_dim.x - 1) / block_dim.x,
                        (output_height + block_dim.y - 1) / block_dim.y);

  extract_real_kernel<<<grid_dim_extract, block_dim, 0, stream_>>>(
      fft_data_->d_padded_input.get(), output_data, output_width, output_height,
      fft_data_->padded_width);

  if (settings_.divide) {
    int num_elements = output_width * output_height;
    int threads      = 256;
    int blocks       = (num_elements + threads - 1) / threads;

    divide_kernel<<<blocks, threads, 0, stream_>>>(input_data, output_data, output_data,
                                                   num_elements);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ConvolutionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  const auto check = [](bool condition, const char *message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };

  check(!input_descs.empty(), "No input descriptors provided");
  check(input_descs.size() == 1, "Convolution expects exactly one input");

  const auto &input_desc = input_descs[0];
  auto        settings   = jsettings.get<ConvolutionSettings>();

  check(input_desc.rank() >= 2, "Input must have at least 2 dimensions");
  check(input_desc.dtype == holoflow::core::DType::F32, "Input must be F32 type");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "Input must be in device memory");

  std::ifstream file(settings.kernel_file);
  check(file.is_open(), "Given kernel file does not exist");

  nlohmann::json j;
  file >> j;

  check(j.contains("kernel"), "JSON must contain field 'kernel'");
  check(j["kernel"].is_array(), "Kernel must be an array");
  auto kernel_json = j.at("kernel");

  auto kernel_height = kernel_json.size();
  auto kernel_width  = kernel_json[0].size();

  check(kernel_height > 0, "Kernel must not be empty");
  check(kernel_width > 0, "Kernel must have at least one column");

  for (const auto &row : kernel_json) {
    check(row.size() == kernel_width, "All rows of the kernel must have the same width");
  }

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
  auto result   = infer(input_descs, jsettings);
  auto settings = jsettings.get<ConvolutionSettings>();

  const auto &input_desc = input_descs[0];

  std::ifstream file(settings.kernel_file);
  if (!file.is_open()) {
    throw std::runtime_error("Could not load :" + settings.kernel_file);
  }

  auto last_write_time = std::filesystem::last_write_time(settings.kernel_file);

  nlohmann::json j;
  file >> j;

  auto kernel_json = j.at("kernel");

  auto kernel_height = kernel_json.size();
  auto kernel_width  = kernel_json[0].size();

  std::vector<float> squashed_kernel;

  for (const auto &row : kernel_json) {
    for (const auto &val : row) {
      squashed_kernel.push_back(val.get<float>());
    }
  }

  auto d_kernel = curaii::make_unique_device_ptr<float>(squashed_kernel.size());

  CUDA_CHECK(cudaMemcpyAsync(d_kernel.get(), squashed_kernel.data(),
                             squashed_kernel.size() * sizeof(float), cudaMemcpyHostToDevice,
                             ctx.stream));

  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  auto task = new Convolution(settings, input_desc, result.output_descs[0], ctx.stream,
                              std::move(d_kernel), static_cast<int>(kernel_width),
                              static_cast<int>(kernel_height), last_write_time);

  task->create_fft_data();

  // Compile callback LTO
  auto lto   = complex_multiply_callback_lto();
  task->lto_ = lto;

  // Create FFT plans using XtMakePlanMany
  int           rank          = 2;
  long long int n[2]          = {static_cast<long long>(task->fft_data_->padded_height),
                                 static_cast<long long>(task->fft_data_->padded_width)};
  long long int inembed[2]    = {static_cast<long long>(task->fft_data_->padded_height),
                                 static_cast<long long>(task->fft_data_->padded_width)};
  int           istride       = 1;
  int           idist         = task->fft_data_->padded_height * task->fft_data_->padded_width;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {static_cast<long long>(task->fft_data_->padded_height),
                                 static_cast<long long>(task->fft_data_->padded_width)};
  int           ostride       = 1;
  int           odist         = task->fft_data_->padded_height * task->fft_data_->padded_width;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = 1;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  CUFFT_CHECK(cufftSetStream(task->fft_data_->fft_plan.get(), task->stream_));
  CUFFT_CHECK(cufftSetStream(task->fft_data_->inv_plan.get(), task->stream_));

  auto *d_info_ptr = reinterpret_cast<void *>(task->fft_data_->d_callback_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(task->fft_data_->inv_plan.get(), "complex_multiply_callback",
                                    lto.data(), lto.size(), CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(task->fft_data_->fft_plan.get(), rank, n, inembed, istride, idist,
                                  inputtype, onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));
  CUFFT_CHECK(cufftXtMakePlanMany(task->fft_data_->inv_plan.get(), rank, n, inembed, istride, idist,
                                  inputtype, onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  cufftExecC2C(task->fft_data_->fft_plan.get(), task->fft_data_->d_padded_kernel.get(),
               task->fft_data_->d_freq_kernel.get(), CUFFT_FORWARD);

  CUDA_CHECK(cudaStreamSynchronize(task->stream_));

  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

void Convolution::create_fft_data() {
  auto input_width = static_cast<unsigned int>(input_desc_.shape.back());
  auto input_height =
      input_desc_.shape.size() > 1
          ? static_cast<unsigned int>(input_desc_.shape[input_desc_.shape.size() - 2])
          : 1;

  fft_data_ = std::make_unique<FFTConvolutionData>();

  fft_data_->padded_width  = input_width;
  fft_data_->padded_height = input_height;

  size_t padded_size         = fft_data_->padded_width * fft_data_->padded_height;
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

std::unique_ptr<holoflow::core::ISyncTask>
ConvolutionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                           std::span<const holoflow::core::TDesc>     input_descs,
                           const nlohmann::json                      &jsettings,
                           const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old_conv = dynamic_cast<Convolution *>(old_task.get());
  HOLOVIBES_CHECK(old_conv != nullptr, "old_task is not an Convolution");

  infer(input_descs, jsettings);
  auto settings = jsettings.get<ConvolutionSettings>();

  auto current_write_time = std::filesystem::last_write_time(settings.kernel_file);
  bool kernel_changed     = (current_write_time != old_conv->kernel_last_write_time_);

  if (!kernel_changed) {
    old_conv->settings_ = settings;
    old_conv->stream_   = ctx.stream;
    old_task.release();
    CUFFT_CHECK(cufftSetStream(old_conv->fft_data_->fft_plan.get(), old_conv->stream_));
    CUFFT_CHECK(cufftSetStream(old_conv->fft_data_->inv_plan.get(), old_conv->stream_));
    return std::unique_ptr<holoflow::core::ISyncTask>(old_conv);
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs