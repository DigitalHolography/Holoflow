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

#include "holotask/syncs/spatial_contrast.hh"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SpatialContrastSettings &s) {
  j = nlohmann::json{{"window_size", s.window_size},
                     {"temporal_window_size", s.temporal_window_size},
                     {"method", s.method == SpatialContrastMethod::FFT ? "fft" : "kernel"}};
}

void from_json(const nlohmann::json &j, SpatialContrastSettings &s) {
  s.window_size = j.value("window_size", 7);
  s.temporal_window_size = j.value("temporal_window_size", 1);
  const auto method = j.value("method", std::string{"kernel"});
  if (method == "kernel")
    s.method = SpatialContrastMethod::Kernel;
  else if (method == "fft")
    s.method = SpatialContrastMethod::FFT;
  else
    throw std::invalid_argument("SpatialContrast: method must be 'kernel' or 'fft'");
}

namespace {

constexpr int kMinWindowSize = 3;
constexpr int kMaxWindowSize = 255;

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::invalid_argument("SpatialContrast: " + message);
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
    if (desc.shape[i] != 0 && expected > std::numeric_limits<size_t>::max() / desc.shape[i]) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

size_t num_elements(std::span<const size_t> shape) {
  size_t count = 1;
  for (const auto dim : shape) {
    if (dim == 0) {
      return 0;
    }
    check(count <= std::numeric_limits<size_t>::max() / dim, "input size overflows");
    count *= dim;
  }
  return count;
}

template <typename InT>
__global__ void horizontal_sums_kernel(const InT *__restrict__ input, double2 *__restrict__ sums,
                                       int width, int height, int window_size,
                                       int64_t total_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }

  const int64_t plane_size   = static_cast<int64_t>(width) * height;
  const int64_t plane_offset = (idx / plane_size) * plane_size;
  const int64_t pixel_idx    = idx - plane_offset;
  const int     y            = static_cast<int>(pixel_idx / width);
  const int     x            = static_cast<int>(pixel_idx % width);
  const int     radius       = window_size / 2;
  const int     x_begin      = x > radius ? x - radius : 0;
  const int     x_end        = (x + radius < width) ? x + radius : width - 1;

  double sum         = 0.0;
  double squared_sum = 0.0;
  const int64_t row_offset = plane_offset + static_cast<int64_t>(y) * width;
  for (int sample_x = x_begin; sample_x <= x_end; ++sample_x) {
    const double value = static_cast<double>(input[row_offset + sample_x]);
    sum += value;
    squared_sum += value * value;
  }

  sums[idx] = make_double2(sum, squared_sum);
}

template <typename InT>
__global__ void temporal_contrast_kernel(const InT *__restrict__ input, float *__restrict__ output,
                                         int width, int height, int temporal_window_size,
                                         int64_t output_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= output_elements)
    return;

  const int64_t plane_size = static_cast<int64_t>(width) * height;
  double        sum         = 0.0;
  double        squared_sum = 0.0;
  for (int time = 0; time < temporal_window_size; ++time) {
    const double value =
        static_cast<double>(input[static_cast<int64_t>(time) * plane_size + idx]);
    sum += value;
    squared_sum += value * value;
  }

  const double mean     = sum / static_cast<double>(temporal_window_size);
  const double variance =
      fmax(squared_sum / static_cast<double>(temporal_window_size) - mean * mean, 0.0);
  output[idx] = mean > 0.0 ? static_cast<float>(sqrt(variance) / mean) : 0.0F;
}

__global__ void contrast_from_sums_kernel(const double2 *__restrict__ horizontal_sums,
                                          float *__restrict__ output, int width, int height,
                                          int window_size, int temporal_window_size,
                                          int64_t output_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= output_elements) {
    return;
  }

  const int64_t plane_size   = static_cast<int64_t>(width) * height;
  const int64_t plane_offset = (idx / plane_size) * plane_size;
  const int64_t pixel_idx    = idx - plane_offset;
  const int     y            = static_cast<int>(pixel_idx / width);
  const int     x            = static_cast<int>(pixel_idx % width);
  const int     radius       = window_size / 2;
  const int     y_begin      = y > radius ? y - radius : 0;
  const int     y_end        = (y + radius < height) ? y + radius : height - 1;
  const int     x_begin      = x > radius ? x - radius : 0;
  const int     x_end        = (x + radius < width) ? x + radius : width - 1;

  double sum         = 0.0;
  double squared_sum = 0.0;
  for (int time = 0; time < temporal_window_size; ++time) {
    const int64_t time_offset = static_cast<int64_t>(time) * plane_size;
    for (int sample_y = y_begin; sample_y <= y_end; ++sample_y) {
      const double2 row_sum =
          horizontal_sums[plane_offset + time_offset +
                          static_cast<int64_t>(sample_y) * width + x];
      sum += row_sum.x;
      squared_sum += row_sum.y;
    }
  }

  const int64_t sample_count = static_cast<int64_t>(temporal_window_size) *
                               (x_end - x_begin + 1) * (y_end - y_begin + 1);
  const double mean         = sum / static_cast<double>(sample_count);
  const double variance =
      fmax(squared_sum / static_cast<double>(sample_count) - mean * mean, 0.0);
  output[idx] = mean > 0.0 ? static_cast<float>(sqrt(variance) / mean) : 0.0F;
}

__global__ void fill_box_kernel(cufftComplex *__restrict__ box, int fft_width, int fft_height,
                                int window_size, int64_t fft_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= fft_elements)
    return;

  const int x = static_cast<int>(idx % fft_width);
  const int y = static_cast<int>(idx / fft_width);
  box[idx] = make_cuFloatComplex(x < window_size && y < window_size ? 1.0F : 0.0F, 0.0F);
}

template <typename InT>
__global__ void prepare_fft_input_kernel(const InT *__restrict__ input,
                                         cufftComplex *__restrict__ fft_data, int width,
                                         int height, int fft_width, int fft_height, int radius,
                                         int64_t fft_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= fft_elements)
    return;

  const int64_t fft_plane_size = static_cast<int64_t>(fft_width) * fft_height;
  const int64_t batch          = idx / fft_plane_size;
  const int64_t plane          = batch / 2;
  const bool    squared        = (batch & 1) != 0;
  const int64_t pixel          = idx - batch * fft_plane_size;
  const int     x              = static_cast<int>(pixel % fft_width);
  const int     y              = static_cast<int>(pixel / fft_width);

  float value = 0.0F;
  if (x >= radius && x < radius + width && y >= radius && y < radius + height) {
    const int64_t source_idx = plane * static_cast<int64_t>(width) * height +
                               static_cast<int64_t>(y - radius) * width + (x - radius);
    value = static_cast<float>(input[source_idx]);
    if (squared)
      value *= value;
  }
  fft_data[idx] = make_cuFloatComplex(value, 0.0F);
}

__global__ void multiply_fft_kernel(cufftComplex *__restrict__ values,
                                    const cufftComplex *__restrict__ box,
                                    int64_t fft_plane_size, int64_t total_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const cufftComplex value = values[idx];
  const cufftComplex kernel = box[idx % fft_plane_size];
  values[idx] = make_cuFloatComplex(value.x * kernel.x - value.y * kernel.y,
                                    value.x * kernel.y + value.y * kernel.x);
}

__global__ void contrast_from_fft_kernel(const cufftComplex *__restrict__ fft_data,
                                         float *__restrict__ output, int width, int height,
                                         int fft_width, int fft_height, int window_size,
                                         int temporal_window_size, int64_t input_planes,
                                         int64_t output_elements) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= output_elements)
    return;

  const int64_t plane_size = static_cast<int64_t>(width) * height;
  const int64_t output_plane = idx / plane_size;
  const int64_t pixel_idx = idx - output_plane * plane_size;
  const int y = static_cast<int>(pixel_idx / width);
  const int x = static_cast<int>(pixel_idx % width);
  const int crop_offset = window_size - 1;
  const int64_t fft_plane_size = static_cast<int64_t>(fft_width) * fft_height;
  const int64_t first_input_plane = output_plane * temporal_window_size;
  const int radius = window_size / 2;
  const int x_begin = x > radius ? x - radius : 0;
  const int x_end = x + radius < width ? x + radius : width - 1;
  const int y_begin = y > radius ? y - radius : 0;
  const int y_end = y + radius < height ? y + radius : height - 1;
  const int64_t sample_count = static_cast<int64_t>(temporal_window_size) *
                               (x_end - x_begin + 1) * (y_end - y_begin + 1);
  const int64_t crop_idx = static_cast<int64_t>(y + crop_offset) * fft_width + (x + crop_offset);

  float sum = 0.0F;
  float squared_sum = 0.0F;
  for (int time = 0; time < temporal_window_size; ++time) {
    const int64_t input_plane = first_input_plane + time;
    if (input_plane >= input_planes)
      break;
    const auto sum_value = fft_data[(input_plane * 2) * fft_plane_size + crop_idx];
    const auto squared_value = fft_data[(input_plane * 2 + 1) * fft_plane_size + crop_idx];
    sum += sum_value.x;
    squared_sum += squared_value.x;
  }

  const float fft_scale = 1.0F / static_cast<float>(fft_plane_size);
  const float mean = sum * fft_scale / static_cast<float>(sample_count);
  const float variance = fmaxf(squared_sum * fft_scale / static_cast<float>(sample_count) -
                                   mean * mean,
                               0.0F);
  output[idx] = mean > 0.0F ? sqrtf(variance) / mean : 0.0F;
}

int next_power_of_two(int value) {
  if (value <= 0 || value > (1 << 30))
    throw std::invalid_argument("SpatialContrast: FFT dimensions exceed supported size");
  int result = 1;
  while (result < value)
    result <<= 1;
  return result;
}

class SpatialContrast : public holoflow::core::ISyncTask {
public:
  SpatialContrast(SpatialContrastSettings settings, holoflow::core::TDesc idesc, int width,
                  int height, int temporal_window_size, int64_t total_elements,
                  int64_t output_elements, int64_t input_planes, int fft_width, int fft_height,
                  DevPtr<double2> &&horizontal_sums, DevPtr<cufftComplex> &&fft_data,
                  DevPtr<cufftComplex> &&box_spectrum, curaii::CufftHandle &&fft_plan,
                  curaii::CufftHandle &&box_plan, cudaStream_t stream)
      : settings_(settings), idesc_(std::move(idesc)), width_(width), height_(height),
        temporal_window_size_(temporal_window_size), total_elements_(total_elements),
        output_elements_(output_elements), input_planes_(input_planes), fft_width_(fft_width),
        fft_height_(fft_height), horizontal_sums_(std::move(horizontal_sums)),
        fft_data_(std::move(fft_data)), box_spectrum_(std::move(box_spectrum)),
        fft_plan_(std::move(fft_plan)), box_plan_(std::move(box_plan)), stream_(stream) {}

  bool supports_cuda_graph() const noexcept override {
    return settings_.method == SpatialContrastMethod::Kernel || settings_.window_size == 1;
  }

  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs, &cancelled, nullptr, nullptr};
    (void)execute(execution);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    constexpr int block_size = 256;
    const auto    grid_size  = static_cast<unsigned int>((total_elements_ + block_size - 1) /
                                                         block_size);
    const auto output_grid_size = static_cast<unsigned int>(
        (output_elements_ + block_size - 1) / block_size);
    const auto   *input      = ctx.inputs[0].data();
    auto         *output     = reinterpret_cast<float *>(ctx.outputs[0].data());

    if (settings_.window_size == 1) {
      const auto enqueue_temporal = [&](const auto *typed_input) {
        temporal_contrast_kernel<<<output_grid_size, block_size, 0, stream_>>>(
            typed_input, output, width_, height_, temporal_window_size_, output_elements_);
      };
      switch (idesc_.dtype) {
      case holoflow::core::DType::U8:
        enqueue_temporal(reinterpret_cast<const uint8_t *>(input));
        break;
      case holoflow::core::DType::U16:
        enqueue_temporal(reinterpret_cast<const uint16_t *>(input));
        break;
      case holoflow::core::DType::F32:
        enqueue_temporal(reinterpret_cast<const float *>(input));
        break;
      default:
        logger()->error("[SpatialContrast::execute] unsupported dtype");
        std::abort();
      }
      CUDA_CHECK(cudaGetLastError());
      return holoflow::core::OpResult::Ok;
    }

    if (settings_.method == SpatialContrastMethod::FFT && settings_.window_size > 1) {
      const int64_t fft_plane_size = static_cast<int64_t>(fft_width_) * fft_height_;
      const int64_t fft_elements = input_planes_ * 2 * fft_plane_size;
      const auto fft_grid_size = static_cast<unsigned int>((fft_elements + block_size - 1) /
                                                           block_size);
      const auto enqueue_fft = [&](const auto *typed_input) {
        prepare_fft_input_kernel<<<fft_grid_size, block_size, 0, stream_>>>(
            typed_input, fft_data_.get(), width_, height_, fft_width_, fft_height_,
            settings_.window_size / 2, fft_elements);
        CUFFT_CHECK(cufftExecC2C(fft_plan_.get(), fft_data_.get(), fft_data_.get(), CUFFT_FORWARD));
        multiply_fft_kernel<<<fft_grid_size, block_size, 0, stream_>>>(
            fft_data_.get(), box_spectrum_.get(), fft_plane_size, fft_elements);
        CUFFT_CHECK(cufftExecC2C(fft_plan_.get(), fft_data_.get(), fft_data_.get(), CUFFT_INVERSE));
        contrast_from_fft_kernel<<<output_grid_size, block_size, 0, stream_>>>(
            fft_data_.get(), output, width_, height_, fft_width_, fft_height_, settings_.window_size,
            temporal_window_size_, input_planes_, output_elements_);
      };

      switch (idesc_.dtype) {
      case holoflow::core::DType::U8:
        enqueue_fft(reinterpret_cast<const uint8_t *>(input));
        break;
      case holoflow::core::DType::U16:
        enqueue_fft(reinterpret_cast<const uint16_t *>(input));
        break;
      case holoflow::core::DType::F32:
        enqueue_fft(reinterpret_cast<const float *>(input));
        break;
      default:
        logger()->error("[SpatialContrast::execute] unsupported dtype");
        std::abort();
      }
      CUDA_CHECK(cudaGetLastError());
      return holoflow::core::OpResult::Ok;
    }

    const auto enqueue = [&](const auto *typed_input) {
      horizontal_sums_kernel<<<grid_size, block_size, 0, stream_>>>(
          typed_input, horizontal_sums_.get(), width_, height_, settings_.window_size,
          total_elements_);
      contrast_from_sums_kernel<<<output_grid_size, block_size, 0, stream_>>>(
          horizontal_sums_.get(), output, width_, height_, settings_.window_size,
          temporal_window_size_, output_elements_);
    };

    switch (idesc_.dtype) {
    case holoflow::core::DType::U8:
      enqueue(reinterpret_cast<const uint8_t *>(input));
      break;
    case holoflow::core::DType::U16:
      enqueue(reinterpret_cast<const uint16_t *>(input));
      break;
    case holoflow::core::DType::F32:
      enqueue(reinterpret_cast<const float *>(input));
      break;
    default:
      logger()->error("[SpatialContrast::execute] unsupported dtype");
      std::abort();
    }

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) {
    stream_ = stream;
    if (settings_.method == SpatialContrastMethod::FFT && settings_.window_size > 1) {
      CUFFT_CHECK(cufftSetStream(fft_plan_.get(), stream_));
      CUFFT_CHECK(cufftSetStream(box_plan_.get(), stream_));
    }
  }

  const SpatialContrastSettings &settings() const { return settings_; }
  const holoflow::core::TDesc  &idesc() const { return idesc_; }

private:
  SpatialContrastSettings settings_;
  holoflow::core::TDesc  idesc_;
  int                    width_;
  int                    height_;
  int                    temporal_window_size_;
  int64_t                total_elements_;
  int64_t                output_elements_;
  int64_t                input_planes_;
  int                    fft_width_;
  int                    fft_height_;
  DevPtr<double2>        horizontal_sums_;
  DevPtr<cufftComplex>   fft_data_;
  DevPtr<cufftComplex>   box_spectrum_;
  curaii::CufftHandle    fft_plan_;
  curaii::CufftHandle    box_plan_;
  cudaStream_t           stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// SpatialContrastFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
SpatialContrastFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<SpatialContrastSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() >= 2, "input must have rank 2 or higher");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32,
        "input must have dtype U8, U16, or F32");
  check((settings.window_size == 1 && settings.temporal_window_size > 1) ||
            (settings.window_size >= kMinWindowSize && settings.window_size <= kMaxWindowSize &&
             settings.window_size % 2 == 1),
        "window_size must be 1 for temporal-only contrast or an odd integer between 3 and 255");
  check(settings.temporal_window_size == 1 || settings.temporal_window_size >= 3,
        "temporal_window_size must be 1 or at least 3");
  check(settings.method == SpatialContrastMethod::Kernel ||
            settings.method == SpatialContrastMethod::FFT,
        "method must be Kernel or FFT");
  if (settings.temporal_window_size > 1) {
    check(idesc.rank() == 3 && idesc.shape[0] ==
                                  static_cast<size_t>(settings.temporal_window_size),
          "3D input batch must match temporal_window_size");
  }

  const auto height = idesc.shape[idesc.rank() - 2];
  const auto width  = idesc.shape[idesc.rank() - 1];
  check(height > 0 && height <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "image height must fit in a positive int");
  check(width > 0 && width <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "image width must fit in a positive int");

  const auto total_elements = num_elements(idesc.shape);
  check(total_elements > 0 &&
            total_elements <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "input must have a positive element count that fits in int64");

  auto oshape = idesc.shape;
  if (settings.temporal_window_size > 1)
    oshape[0] = 1;
  holoflow::core::TDesc odesc(oshape, holoflow::core::DType::F32, idesc.mem_loc);
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SpatialContrastFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto  settings = jsettings.get<SpatialContrastSettings>();
  const auto &idesc    = input_descs[0];
  const auto  height   = static_cast<int>(idesc.shape[idesc.rank() - 2]);
  const auto  width    = static_cast<int>(idesc.shape[idesc.rank() - 1]);
  const auto  count    = num_elements(idesc.shape);
  const auto  temporal_window_size = settings.temporal_window_size;
  const auto  plane_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  const auto  input_planes = count / plane_size;
  const auto  output_count = temporal_window_size > 1
                                 ? count / static_cast<size_t>(temporal_window_size)
                                 : count;
  DevPtr<double2>      sums;
  DevPtr<cufftComplex> fft_data;
  DevPtr<cufftComplex> box_spectrum;
  curaii::CufftHandle  fft_plan;
  curaii::CufftHandle  box_plan;
  int                  fft_width = 0;
  int                  fft_height = 0;

  if (settings.window_size == 1) {
    // Temporal-only contrast is fused into one kernel and needs no intermediate buffer.
  } else if (settings.method == SpatialContrastMethod::Kernel) {
    sums = curaii::make_unique_device_ptr<double2>(count);
  } else {
    const int64_t fft_min_width = static_cast<int64_t>(width) +
                                  2LL * settings.window_size - 2;
    const int64_t fft_min_height = static_cast<int64_t>(height) +
                                   2LL * settings.window_size - 2;
    check(fft_min_width <= std::numeric_limits<int>::max() &&
              fft_min_height <= std::numeric_limits<int>::max(),
          "FFT dimensions exceed supported size");
    fft_width = next_power_of_two(static_cast<int>(fft_min_width));
    fft_height = next_power_of_two(static_cast<int>(fft_min_height));
    const auto fft_plane_size = static_cast<size_t>(fft_width) * fft_height;
    check(fft_plane_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
          "FFT plane is too large for cuFFT");
    check(input_planes <= static_cast<size_t>(std::numeric_limits<int>::max() / 2),
          "FFT batch count is too large for cuFFT");
    check(input_planes <= std::numeric_limits<size_t>::max() / 2 / fft_plane_size,
          "FFT workspace size overflows");
    const auto fft_elements = input_planes * 2 * fft_plane_size;
    fft_data = curaii::make_unique_device_ptr<cufftComplex>(fft_elements);
    box_spectrum = curaii::make_unique_device_ptr<cufftComplex>(fft_plane_size);

    int dims[2] = {fft_height, fft_width};
    int embed[2] = {fft_height, fft_width};
    const int distance = static_cast<int>(fft_plane_size);
    size_t fft_work_size = 0;
    size_t box_work_size = 0;
    CUFFT_CHECK(cufftMakePlanMany(fft_plan.get(), 2, dims, embed, 1, distance, embed, 1, distance,
                                  CUFFT_C2C, static_cast<int>(2 * input_planes), &fft_work_size));
    CUFFT_CHECK(cufftMakePlan2d(box_plan.get(), fft_height, fft_width, CUFFT_C2C, &box_work_size));
    CUFFT_CHECK(cufftSetStream(fft_plan.get(), ctx.stream));
    CUFFT_CHECK(cufftSetStream(box_plan.get(), ctx.stream));
    const auto box_grid = static_cast<unsigned int>((fft_plane_size + 255) / 256);
    fill_box_kernel<<<box_grid, 256, 0, ctx.stream>>>(box_spectrum.get(), fft_width, fft_height,
                                                     settings.window_size, fft_plane_size);
    CUFFT_CHECK(cufftExecC2C(box_plan.get(), box_spectrum.get(), box_spectrum.get(), CUFFT_FORWARD));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  }

  return std::make_unique<SpatialContrast>(settings, idesc, width, height,
                                           temporal_window_size, static_cast<int64_t>(count),
                                           static_cast<int64_t>(output_count),
                                           static_cast<int64_t>(input_planes), fft_width,
                                           fft_height, std::move(sums), std::move(fft_data),
                                           std::move(box_spectrum), std::move(fft_plan),
                                           std::move(box_plan), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
SpatialContrastFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                               std::span<const holoflow::core::TDesc>     input_descs,
                               const nlohmann::json                      &jsettings,
                               const holoflow::core::SyncCreateCtx       &ctx) const {
  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  (void)infer(input_descs, jsettings);

  auto *old_contrast = dynamic_cast<SpatialContrast *>(old_task.get());
  if (old_contrast != nullptr && input_descs.size() == 1) {
    const auto  new_settings = jsettings.get<SpatialContrastSettings>();
    const auto &new_idesc    = input_descs[0];
    const auto &old_idesc    = old_contrast->idesc();
    const bool  can_reuse = new_settings == old_contrast->settings() &&
                            new_idesc.shape == old_idesc.shape &&
                            new_idesc.strides == old_idesc.strides &&
                            new_idesc.dtype == old_idesc.dtype &&
                            new_idesc.mem_loc == old_idesc.mem_loc;
    if (can_reuse) {
      old_contrast->update_stream(ctx.stream);
      return old_task;
    }
  }

  ctx.invalidate_execution();
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
