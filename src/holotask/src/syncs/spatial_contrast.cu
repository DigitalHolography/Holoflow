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
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SpatialContrastSettings &s) {
  j = nlohmann::json{{"window_size", s.window_size}};
}

void from_json(const nlohmann::json &j, SpatialContrastSettings &s) {
  s.window_size = j.value("window_size", 7);
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

__global__ void contrast_from_sums_kernel(const double2 *__restrict__ horizontal_sums,
                                          float *__restrict__ output, int width, int height,
                                          int window_size, int64_t total_elements) {
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
  const int     y_begin      = y > radius ? y - radius : 0;
  const int     y_end        = (y + radius < height) ? y + radius : height - 1;
  const int     x_begin      = x > radius ? x - radius : 0;
  const int     x_end        = (x + radius < width) ? x + radius : width - 1;

  double sum         = 0.0;
  double squared_sum = 0.0;
  for (int sample_y = y_begin; sample_y <= y_end; ++sample_y) {
    const double2 row_sum =
        horizontal_sums[plane_offset + static_cast<int64_t>(sample_y) * width + x];
    sum += row_sum.x;
    squared_sum += row_sum.y;
  }

  const int    sample_count = (x_end - x_begin + 1) * (y_end - y_begin + 1);
  const double mean         = sum / static_cast<double>(sample_count);
  const double variance =
      fmax(squared_sum / static_cast<double>(sample_count) - mean * mean, 0.0);
  output[idx] = mean > 0.0 ? static_cast<float>(sqrt(variance) / mean) : 0.0F;
}

class SpatialContrast : public holoflow::core::ISyncTask {
public:
  SpatialContrast(SpatialContrastSettings settings, holoflow::core::TDesc idesc, int width,
                  int height, int64_t total_elements, DevPtr<double2> &&horizontal_sums,
                  cudaStream_t stream)
      : settings_(settings), idesc_(std::move(idesc)), width_(width), height_(height),
        total_elements_(total_elements), horizontal_sums_(std::move(horizontal_sums)),
        stream_(stream) {}

  bool supports_cuda_graph() const noexcept override { return true; }

  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs, &cancelled, nullptr, nullptr};
    (void)execute(execution);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    constexpr int block_size = 256;
    const auto    grid_size  = static_cast<unsigned int>((total_elements_ + block_size - 1) /
                                                   block_size);
    const auto   *input      = ctx.inputs[0].data();
    auto         *output     = reinterpret_cast<float *>(ctx.outputs[0].data());

    const auto enqueue = [&](const auto *typed_input) {
      horizontal_sums_kernel<<<grid_size, block_size, 0, stream_>>>(
          typed_input, horizontal_sums_.get(), width_, height_, settings_.window_size,
          total_elements_);
      contrast_from_sums_kernel<<<grid_size, block_size, 0, stream_>>>(
          horizontal_sums_.get(), output, width_, height_, settings_.window_size, total_elements_);
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

  void update_stream(cudaStream_t stream) { stream_ = stream; }

  const SpatialContrastSettings &settings() const { return settings_; }
  const holoflow::core::TDesc  &idesc() const { return idesc_; }

private:
  SpatialContrastSettings settings_;
  holoflow::core::TDesc  idesc_;
  int                    width_;
  int                    height_;
  int64_t                total_elements_;
  DevPtr<double2>        horizontal_sums_;
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
  check(settings.window_size >= kMinWindowSize && settings.window_size <= kMaxWindowSize &&
            settings.window_size % 2 == 1,
        "window_size must be an odd integer between 3 and 255");

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

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::F32, idesc.mem_loc);
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
  auto        sums     = curaii::make_unique_device_ptr<double2>(count);

  return std::make_unique<SpatialContrast>(settings, idesc, width, height,
                                           static_cast<int64_t>(count), std::move(sums), ctx.stream);
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
