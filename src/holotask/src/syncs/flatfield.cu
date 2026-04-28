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

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
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
            int width, int radius_y, int radius_x, DevPtr<float> &&d_kernel_y,
            DevPtr<float> &&d_kernel_x, DevPtr<float> &&d_temp, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), total_(total), height_(height),
        width_(width), radius_y_(radius_y), radius_x_(radius_x), d_kernel_y_(std::move(d_kernel_y)),
        d_kernel_x_(std::move(d_kernel_x)), d_temp_(std::move(d_temp)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto *idata = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto       *odata = reinterpret_cast<float *>(ctx.outputs[0].data());

    constexpr int block = 256;
    const int     grid  = static_cast<int>((total_ + block - 1) / block);
    gaussian_horizontal_kernel<<<grid, block, 0, stream_>>>(idata, d_temp_.get(), d_kernel_x_.get(),
                                                            total_, height_, width_, radius_x_);
    gaussian_subtract_vertical_kernel<<<grid, block, 0, stream_>>>(
        idata, d_temp_.get(), odata, d_kernel_y_.get(), total_, height_, width_, radius_y_);

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) { stream_ = stream; }

  const FlatfieldSettings     &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  FlatfieldSettings     settings_;
  holoflow::core::TDesc idesc_;
  size_t                total_;
  int                   height_;
  int                   width_;
  int                   radius_y_;
  int                   radius_x_;
  DevPtr<float>         d_kernel_y_;
  DevPtr<float>         d_kernel_x_;
  DevPtr<float>         d_temp_;
  cudaStream_t          stream_;
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

  auto d_kernel_y = curaii::make_unique_device_ptr<float>(kernel_y.size());
  CUDA_CHECK(cudaMemcpyAsync(d_kernel_y.get(), kernel_y.data(), kernel_y.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  auto d_kernel_x = curaii::make_unique_device_ptr<float>(kernel_x.size());
  CUDA_CHECK(cudaMemcpyAsync(d_kernel_x.get(), kernel_x.data(), kernel_x.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  auto d_temp = curaii::make_unique_device_ptr<float>(idesc.num_elements());

  return std::make_unique<Flatfield>(settings, idesc, idesc.num_elements(), height, width, radius_y,
                                     radius_x, std::move(d_kernel_y), std::move(d_kernel_x),
                                     std::move(d_temp), ctx.stream);
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
