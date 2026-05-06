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

#include "holotask/syncs/pct_clip.hh"

#include <cub/cub.cuh>

#include <string>
#include <utility>

#include <math_constants.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const PctClipSettings::Ellipse &e) {
  j = nlohmann::json{
      {"cx", e.cx}, {"cy", e.cy}, {"rx", e.rx}, {"ry", e.ry}, {"angle", e.angle},
  };
}

void from_json(const nlohmann::json &j, PctClipSettings::Ellipse &e) {
  j.at("cx").get_to(e.cx);
  j.at("cy").get_to(e.cy);
  j.at("rx").get_to(e.rx);
  j.at("ry").get_to(e.ry);
  j.at("angle").get_to(e.angle);
}

void to_json(nlohmann::json &j, const PctClipSettings &s) {
  j = nlohmann::json{
      {"min_pct", s.min_pct},
      {"max_pct", s.max_pct},
      {"roi", s.roi},
  };
}

void from_json(const nlohmann::json &j, PctClipSettings &s) {
  j.at("min_pct").get_to(s.min_pct);
  j.at("max_pct").get_to(s.max_pct);
  j.at("roi").get_to(s.roi);
}

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[PctClipFactory::infer] error: {}", msg);
    throw std::invalid_argument("PctClipFactory inference error: " + msg);
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

__global__ void clip_kernel(float *odata, const float *idata, int n, const float *min_val,
                            const float *max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float val  = idata[idx];
  val        = fminf(fmaxf(val, *min_val), *max_val);
  odata[idx] = val;
}

__device__ bool in_ellipse(int x, int y, int width, int height, PctClipSettings::Ellipse roi) {
  if (width <= 0 || height <= 0 || roi.rx <= 0.f || roi.ry <= 0.f) {
    return false;
  }

  float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
  float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

  float dx = xn - roi.cx;
  float dy = yn - roi.cy;

  float th = roi.angle * (CUDART_PI_F / 180.0f);
  float c  = cosf(th);
  float s  = sinf(th);
  float xr = c * dx + s * dy;
  float yr = -s * dx + c * dy;

  return (xr * xr) / (roi.rx * roi.rx) + (yr * yr) / (roi.ry * roi.ry) <= 1.0f;
}

__global__ void roi_mask_kernel(uint8_t *mask, int *roi_count, int width, int height, int depth,
                                PctClipSettings::Ellipse roi) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= depth) {
    return;
  }

  int  idx      = z * width * height + y * width + x;
  bool selected = in_ellipse(x, y, width, height, roi);
  mask[idx]     = selected;
  if (selected) {
    atomicAdd(roi_count, 1);
  }
}

// -------------------------------------------------------------------------------------------------
// PctClip task implementation
// -------------------------------------------------------------------------------------------------

class PctClip : public holoflow::core::ISyncTask {
public:
  PctClip(PctClipSettings settings, holoflow::core::TDesc idesc, int roi_count, int min_idx,
          int max_idx, DevPtr<float> &&d_min, DevPtr<float> &&d_max, DevPtr<uint8_t> &&d_roi_mask,
          size_t sort_tmp_bytes, DevPtr<uint8_t> &&d_sort_tmp, size_t select_tmp_bytes,
          DevPtr<uint8_t> &&d_select_tmp, DevPtr<float> &&d_roi, DevPtr<int> &&d_roi_count,
          cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), roi_count_(roi_count),
        min_idx_(min_idx), max_idx_(max_idx), d_min_(std::move(d_min)), d_max_(std::move(d_max)),
        d_roi_mask_(std::move(d_roi_mask)), sort_tmp_bytes_(sort_tmp_bytes),
        d_sort_tmp_(std::move(d_sort_tmp)), select_tmp_bytes_(select_tmp_bytes),
        d_select_tmp_(std::move(d_select_tmp)), d_roi_(std::move(d_roi)),
        d_roi_count_(std::move(d_roi_count)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *idata = reinterpret_cast<float *>(ctx.inputs[0].data());
    auto *odata = reinterpret_cast<float *>(ctx.outputs[0].data());
    auto  count = static_cast<int>(ctx.inputs[0].desc.num_elements());

    CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp_.get(), select_tmp_bytes_, idata,
                                          d_roi_mask_.get(), d_roi_.get(), d_roi_count_.get(),
                                          count, stream_));

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_sort_tmp_.get(), sort_tmp_bytes_, d_roi_.get(),
                                              odata, roi_count_, 0, 32, stream_));

    float *d_min = odata + min_idx_;
    float *d_max = odata + max_idx_;

    CUDA_CHECK(
        cudaMemcpyAsync(d_min_.get(), d_min, sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(
        cudaMemcpyAsync(d_max_.get(), d_max, sizeof(float), cudaMemcpyDeviceToDevice, stream_));

    constexpr int block_size = 256;
    const int     grid_size  = (count + block_size - 1) / block_size;
    clip_kernel<<<grid_size, block_size, 0, stream_>>>(odata, idata, count, d_min_.get(),
                                                       d_max_.get());

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) { stream_ = stream; }

  const PctClipSettings       &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  PctClipSettings       settings_;
  holoflow::core::TDesc idesc_;
  int                   roi_count_;
  int                   min_idx_;
  int                   max_idx_;
  DevPtr<float>         d_min_;
  DevPtr<float>         d_max_;
  DevPtr<uint8_t>       d_roi_mask_;
  size_t                sort_tmp_bytes_;
  DevPtr<uint8_t>       d_sort_tmp_;
  size_t                select_tmp_bytes_;
  DevPtr<uint8_t>       d_select_tmp_;
  DevPtr<float>         d_roi_;
  DevPtr<int>           d_roi_count_;
  cudaStream_t          stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// PctClipFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
PctClipFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<PctClipSettings>();

  check(settings.min_pct >= 0.0f, "min_pct must be >= 0");
  check(settings.min_pct <= 100.0f, "min_pct must be <= 100");
  check(settings.max_pct >= 0.0f, "max_pct must be >= 0");
  check(settings.max_pct <= 100.0f, "max_pct must be <= 100");
  check(settings.min_pct < settings.max_pct, "min_pct must be < max_pct");
  check(settings.roi.rx > 0.0f, "roi.rx must be > 0");
  check(settings.roi.rx <= 1.0f, "roi.rx must be <= 1");
  check(settings.roi.ry > 0.0f, "roi.ry must be > 0");
  check(settings.roi.ry <= 1.0f, "roi.ry must be <= 1");
  check(settings.roi.cx >= 0.0f, "roi.cx must be >= 0");
  check(settings.roi.cx <= 1.0f, "roi.cx must be <= 1");
  check(settings.roi.cy >= 0.0f, "roi.cy must be >= 0");
  check(settings.roi.cy <= 1.0f, "roi.cy must be <= 1");
  check(input_descs.size() == 1, "expected exactly one input");

  const auto &idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(idesc.rank() == 3, "input must be 3D");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");

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
PctClipFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  using curaii::make_unique_device_ptr;

  (void)this->infer(input_descs, jsettings);
  const auto  settings = jsettings.get<PctClipSettings>();
  const auto &idesc    = input_descs[0];

  const int depth  = static_cast<int>(idesc.shape[0]);
  const int height = static_cast<int>(idesc.shape[1]);
  const int width  = static_cast<int>(idesc.shape[2]);
  const int count  = static_cast<int>(idesc.num_elements());

  auto d_min       = make_unique_device_ptr<float>(1);
  auto d_max       = make_unique_device_ptr<float>(1);
  auto d_roi_mask  = make_unique_device_ptr<uint8_t>(count);
  auto d_roi_count = make_unique_device_ptr<int>(1);

  dim3 block_size(16, 16, 1);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y,
                 (depth + block_size.z - 1) / block_size.z);
  CUDA_CHECK(cudaMemsetAsync(d_roi_count.get(), 0, sizeof(int), ctx.stream));
  roi_mask_kernel<<<grid_size, block_size, 0, ctx.stream>>>(d_roi_mask.get(), d_roi_count.get(),
                                                            width, height, depth, settings.roi);
  CUDA_CHECK(cudaGetLastError());

  int h_roi_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&h_roi_count, d_roi_count.get(), sizeof(int), cudaMemcpyDeviceToHost,
                             ctx.stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  HOLOVIBES_CHECK(h_roi_count > 0, "No pixels in ROI");

  const int min_idx = static_cast<int>(settings.min_pct / 100.0f * (h_roi_count - 1));
  const int max_idx = static_cast<int>(settings.max_pct / 100.0f * (h_roi_count - 1));

  size_t sort_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_tmp_bytes, static_cast<float *>(nullptr),
                                            static_cast<float *>(nullptr), h_roi_count, 0, 32,
                                            ctx.stream));
  auto d_sort_tmp = make_unique_device_ptr<uint8_t>(sort_tmp_bytes);

  size_t select_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Flagged(
      nullptr, select_tmp_bytes, static_cast<float *>(nullptr), static_cast<uint8_t *>(nullptr),
      static_cast<float *>(nullptr), static_cast<int *>(nullptr), count, ctx.stream));
  auto d_select_tmp = make_unique_device_ptr<uint8_t>(select_tmp_bytes);
  auto d_roi        = make_unique_device_ptr<float>(h_roi_count);

  return std::make_unique<PctClip>(settings, idesc, h_roi_count, min_idx, max_idx, std::move(d_min),
                                   std::move(d_max), std::move(d_roi_mask), sort_tmp_bytes,
                                   std::move(d_sort_tmp), select_tmp_bytes, std::move(d_select_tmp),
                                   std::move(d_roi), std::move(d_roi_count), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
PctClipFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)this->infer(input_descs, jsettings);

  auto *old_pct_clip = dynamic_cast<PctClip *>(old_task.get());
  if (old_pct_clip == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_pct_clip->idesc();
  const auto  settings  = jsettings.get<PctClipSettings>();
  const bool  can_reuse =
      settings == old_pct_clip->settings() && new_idesc.shape == old_idesc.shape &&
      new_idesc.strides == old_idesc.strides && new_idesc.dtype == old_idesc.dtype &&
      new_idesc.mem_loc == old_idesc.mem_loc;

  if (can_reuse) {
    old_pct_clip->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
