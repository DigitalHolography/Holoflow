#include "bug.hh"
#include "tasks/convolution.hh"

#include <fstream>

namespace holovibes::tasks {

void to_json(nlohmann::json &j, const ConvolutionSettings &s) {
  j = nlohmann::json{
                     {"kernel_file", s.kernel_file}
};
}

void from_json(const nlohmann::json &j, ConvolutionSettings &s) {
  j.at("kernel_file").get_to(s.kernel_file);
}

namespace {

__global__ void convolution_2d_kernel(const float *__restrict__ input, float *__restrict__ output,
                                      const float *__restrict__ kernel, const int input_width,
                                      const int input_height, const int output_width,
                                      const int output_height, const int kernel_width,
                                      const int kernel_height, const int kernel_radius_x,
                                      const int kernel_radius_y) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= output_width || y >= output_height)
    return;

  const int input_x = x + kernel_radius_x;
  const int input_y = y + kernel_radius_y;

  float sum = 0.0f;

  for (int ky = 0; ky < kernel_height; ++ky) {
    for (int kx = 0; kx < kernel_width; ++kx) {
      const int sample_x = input_x + kx - kernel_radius_x;
      const int sample_y = input_y + ky - kernel_radius_y;

      const int clamped_x = max(0, min(sample_x, input_width - 1));
      const int clamped_y = max(0, min(sample_y, input_height - 1));

      const float input_val  = input[clamped_y * input_width + clamped_x];
      const float kernel_val = kernel[ky * kernel_width + kx];

      sum += input_val * kernel_val;
    }
  }

  output[y * output_width + x] = sum;
}
} // namespace

Convolution::Convolution(ConvolutionSettings settings, const holoflow::core::TDesc &input_desc,
                         const holoflow::core::TDesc &output_desc, cudaStream_t stream,
                         DevPtr<float> &&d_kernel,
                         size_t kernel_width, size_t kernel_height, size_t kernel_radius_x,
                         size_t kernel_radius_y)
    : settings_(std::move(settings)), input_desc_(input_desc), output_desc_(output_desc),
      stream_(stream), d_kernel_(std::move(d_kernel)),
      kernel_width_(kernel_width), kernel_height_(kernel_height), kernel_radius_x_(kernel_radius_x),
      kernel_radius_y_(kernel_radius_y) {}

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

  auto input_width = input_desc_.shape.back();
  auto input_height =
      input_desc_.shape.size() > 1 ? input_desc_.shape[input_desc_.shape.size() - 2] : 1;
  auto output_width = output_desc_.shape.back();
  auto output_height =
      output_desc_.shape.size() > 1 ? output_desc_.shape[output_desc_.shape.size() - 2] : 1;

  auto width  = input_desc_.shape.back();
  auto height = input_desc_.shape.size() > 1 ? input_desc_.shape[input_desc_.shape.size() - 2] : 1;

  dim3 block_dim(16, 16);
  dim3 grid_dim(static_cast<unsigned int>((width + block_dim.x - 1) / block_dim.x),
                static_cast<unsigned int>((height + block_dim.y - 1) / block_dim.y));

  convolution_2d_kernel<<<grid_dim, block_dim, 0, stream_>>>(
      input_data, output_data, d_kernel_.get(), static_cast<int>(input_width),
      static_cast<int>(input_height), static_cast<int>(output_width),
      static_cast<int>(output_height), static_cast<int>(kernel_width_),
      static_cast<int>(kernel_height_), static_cast<int>(kernel_radius_x_),
      static_cast<int>(kernel_radius_y_));

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
  // check(!settings.kernel.empty(), "Kernel must not be empty");
  // check(settings.kernel_shape.size() >= 2, "Kernel shape must have at least 2 dimensions");

  // const size_t kernel_elements = settings.kernel_shape[0] * settings.kernel_shape[1];
  // check(settings.kernel.size() == kernel_elements, "Kernel size must match kernel shape");

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
    auto kernel_width = kernel_json[0].size();

    std::vector<float> squashed_kernel;

    for (const auto& row : kernel_json) {
        for (const auto& val : row) {
            squashed_kernel.push_back(val.get<float>());
        }
    }

  auto kernel_radius_x = kernel_width / 2;
  auto kernel_radius_y = kernel_height / 2;

  auto d_kernel = curaii::make_unique_device_ptr<float>(squashed_kernel.size());

  CUDA_CHECK(cudaMemcpyAsync(d_kernel.get(), squashed_kernel.data(),
                             squashed_kernel.size() * sizeof(float), cudaMemcpyHostToDevice,
                             ctx.stream));

  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(new Convolution(
      settings, input_desc, result.output_descs[0], ctx.stream, std::move(d_kernel), kernel_width, kernel_height, kernel_radius_x, kernel_radius_y));
}

} // namespace holovibes::tasks