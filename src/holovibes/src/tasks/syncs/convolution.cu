#include "convolution.hh"

#include <fstream>

#include "bug.hh"

// chqnge pqd kernel

namespace holovibes::tasks::syncs {

void to_json(nlohmann::json &j, const ConvolutionSettings &s) {
  j = nlohmann::json{{"kernel_file", s.kernel_file}, {"divide", s.divide}};
}

void from_json(const nlohmann::json &j, ConvolutionSettings &s) {
  j.at("kernel_file").get_to(s.kernel_file);
  j.at("divide").get_to(s.divide);
}

namespace {

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

__global__ void complex_multiply_kernel(cufftComplex *__restrict__ a,
                                        const cufftComplex *__restrict__ b, const int size,
                                        const float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
    return;

  const float real_a = a[idx].x;
  const float imag_a = a[idx].y;
  const float real_b = b[idx].x;
  const float imag_b = b[idx].y;

  a[idx].x = (real_a * real_b - imag_a * imag_b) * scale;
  a[idx].y = (real_a * imag_b + imag_a * real_b) * scale;
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
                         DevPtr<float> &&d_kernel, const int kernel_width, const int kernel_height)
    : settings_(std::move(settings)), input_desc_(input_desc), output_desc_(output_desc),
      stream_(stream), d_kernel_(std::move(d_kernel)), kernel_width_(kernel_width),
      kernel_height_(kernel_height) {}

holoflow::core::OpResult Convolution::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
    return holoflow::core::OpResult::Cancelled;
  }

  if (ctx.inputs.empty() || ctx.outputs.empty()) {
    return holoflow::core::OpResult::NotReady;
  }

  const auto &input_view  = ctx.inputs[0];
  auto       &output_view = ctx.outputs[0];

  const float *input_data  = reinterpret_cast<const float *>(input_view.data);
  float       *output_data = reinterpret_cast<float *>(output_view.data);

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

  // Forward FFT
  cufftExecC2C(fft_data_->fft_plan, fft_data_->d_padded_input.get(), fft_data_->d_freq_input.get(),
               CUFFT_FORWARD);

  const int freq_size = fft_data_->padded_width * fft_data_->padded_height;
  int       threads   = 256;
  int       blocks    = (freq_size + threads - 1) / threads;
  float     scale     = 1.0f / (fft_data_->padded_width * fft_data_->padded_height);

  complex_multiply_kernel<<<blocks, threads, 0, stream_>>>(
      fft_data_->d_freq_input.get(), fft_data_->d_freq_kernel.get(), freq_size, scale);

  CUDA_CHECK(cudaGetLastError());

  // Inverse FFT
  cufftExecC2C(fft_data_->fft_plan, fft_data_->d_freq_input.get(), fft_data_->d_padded_input.get(),
               CUFFT_INVERSE);

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

  auto task =
      new Convolution(settings, input_desc, result.output_descs[0], ctx.stream, std::move(d_kernel),
                      static_cast<int>(kernel_width), static_cast<int>(kernel_height));

  task->create_fft_data();
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

void Convolution::create_fft_data() {
  auto input_width = static_cast<unsigned int>(input_desc_.shape.back());
  auto input_height =
      input_desc_.shape.size() > 1
          ? static_cast<unsigned int>(input_desc_.shape[input_desc_.shape.size() - 2])
          : 1;

  if (!fft_data_) {
    fft_data_ = std::make_unique<FFTConvolutionData>();

    fft_data_->padded_width  = input_width;
    fft_data_->padded_height = input_height;

    size_t padded_size         = fft_data_->padded_width * fft_data_->padded_height;
    fft_data_->d_padded_input  = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_padded_kernel = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_freq_input    = curaii::make_unique_device_ptr<cufftComplex>(padded_size);
    fft_data_->d_freq_kernel   = curaii::make_unique_device_ptr<cufftComplex>(padded_size);

    int n[2] = {static_cast<int>(fft_data_->padded_height),
                static_cast<int>(fft_data_->padded_width)};

    cufftResult result;
    result = cufftPlanMany(&fft_data_->fft_plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1);
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Failed to create FFT plan");
    }

    result = cufftSetStream(fft_data_->fft_plan, stream_);
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Failed to set FFT stream");
    }

    // Precompute kernel FFT
    dim3 block_dim(16, 16);
    dim3 grid_dim((fft_data_->padded_width + block_dim.x - 1) / block_dim.x,
                  (fft_data_->padded_height + block_dim.y - 1) / block_dim.y);

    pad_kernel<<<grid_dim, block_dim, 0, stream_>>>(
        d_kernel_.get(), fft_data_->d_padded_kernel.get(), kernel_width_, kernel_height_,
        fft_data_->padded_width, fft_data_->padded_height);

    CUDA_CHECK(cudaGetLastError());

    cufftExecC2C(fft_data_->fft_plan, fft_data_->d_padded_kernel.get(),
                 fft_data_->d_freq_kernel.get(), CUFFT_FORWARD);

    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
}

} // namespace holovibes::tasks::syncs