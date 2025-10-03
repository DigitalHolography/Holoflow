#include "pct_clip.hh"

#include <cub/cub.cuh>
#include <math_constants.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holovibes::tasks {

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

__global__ void clip_kernel(float *odata, const float *idata, int N, float *min_val,
                            float *max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  float min  = fmax(*min_val, 1.0f);
  float max  = fmin(*max_val, 99.0f);
  float val  = idata[idx];
  val        = fmin(fmax(val, min), max);
  odata[idx] = val;
}

} // namespace

PctClip::PctClip(const PctClipSettings &settings, DevPtr<float> &&d_min, DevPtr<float> &&d_max,
                 DevPtr<uint8_t> &&d_roi_mask, size_t sort_tmp_bytes, DevPtr<uint8_t> &&d_sort_tmp,
                 size_t select_tmp_bytes, DevPtr<uint8_t> &&d_select_tmp, DevPtr<float> &&d_roi,
                 DevPtr<int> &&d_roi_count, cudaStream_t stream)
    : settings_(settings), d_min_(std::move(d_min)), d_max_(std::move(d_max)),
      d_roi_mask_(std::move(d_roi_mask)), sort_tmp_bytes_(sort_tmp_bytes),
      d_sort_tmp_(std::move(d_sort_tmp)), select_tmp_bytes_(select_tmp_bytes),
      d_select_tmp_(std::move(d_select_tmp)), d_roi_(std::move(d_roi)),
      d_roi_count_(std::move(d_roi_count)), stream_(stream) {}

holoflow::core::OpResult PctClip::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  auto count          = idesc.num_elements();

  // Filter ROI pixels.
  CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp_.get(), select_tmp_bytes_,
                                        reinterpret_cast<float *>(idata), d_roi_mask_.get(),
                                        d_roi_.get(), d_roi_count_.get(), count, stream_));
  int  h_roi_count;
  auto kind = cudaMemcpyDeviceToHost;
  CUDA_CHECK(cudaMemcpyAsync(&h_roi_count, d_roi_count_.get(), sizeof(int), kind, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Sort ROI pixels (d_out is temporary storage here).
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_sort_tmp_.get(), sort_tmp_bytes_, d_roi_.get(),
                                            reinterpret_cast<float *>(odata), h_roi_count, 0, 32,
                                            stream_));

  // Compute percentiles.
  HOLOVIBES_CHECK(h_roi_count > 0, "No pixels in ROI");
  kind           = cudaMemcpyDeviceToDevice;
  int    min_idx = static_cast<int>(settings_.min_pct / 100.0f * (h_roi_count - 1));
  int    max_idx = static_cast<int>(settings_.max_pct / 100.0f * (h_roi_count - 1));
  float *d_min   = reinterpret_cast<float *>(odata) + min_idx;
  float *d_max   = reinterpret_cast<float *>(odata) + max_idx;
  CUDA_CHECK(cudaMemcpyAsync(d_min_.get(), d_min, sizeof(float), kind, stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_max_.get(), d_max, sizeof(float), kind, stream_));

  // Clip
  int block_size = 256;
  int grid_size  = (count + block_size - 1) / block_size;
  clip_kernel<<<grid_size, block_size, 0, stream_>>>(reinterpret_cast<float *>(odata),
                                                     reinterpret_cast<float *>(idata), count,
                                                     d_min_.get(), d_max_.get());

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

namespace {

__device__ bool in_ellipse(int x, int y, int W, int H, PctClipSettings::Ellipse roi) {
  if (W <= 0 || H <= 0 || roi.rx <= 0.f || roi.ry <= 0.f) {
    return false;
  }

  // pixel center coordinates in [0,1]
  float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
  float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);

  // translate to ellipse center
  float dx = xn - roi.cx;
  float dy = yn - roi.cy;

  // rotate by -angle
  float th = roi.angle * (CUDART_PI_F / 180.0f);
  float c  = cosf(th);
  float s  = sinf(th);
  float xr = c * dx + s * dy;
  float yr = -s * dx + c * dy;

  // check if in ellipse
  return (xr * xr) / (roi.rx * roi.rx) + (yr * yr) / (roi.ry * roi.ry) <= 1.0f;
}

__global__ void roi_mask_kernel(uint8_t *mask, int W, int H, int Z, PctClipSettings::Ellipse roi) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= W || y >= H || z >= Z) {
    return;
  }

  int idx   = z * W * H + y * W + x;
  mask[idx] = in_ellipse(x, y, W, H, roi);
}

} // namespace

holoflow::core::InferResult
PctClipFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto check = [](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[PctClipFactory::infer] error: {}", msg);
      throw std::invalid_argument("PctClipFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<PctClipSettings>();

  // Validate
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
  auto idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(idesc.rank() == 3, "input must be 3D");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");

  // Success
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

  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<PctClipSettings>();
  auto idesc    = input_descs[0];

  int Z = static_cast<int>(idesc.shape[0]);
  int H = static_cast<int>(idesc.shape[1]);
  int W = static_cast<int>(idesc.shape[2]);
  int N = static_cast<int>(idesc.num_elements());

  // Clip
  auto d_min      = make_unique_device_ptr<float>(1);
  auto d_max      = make_unique_device_ptr<float>(1);
  auto d_roi_mask = make_unique_device_ptr<uint8_t>(N);

  // CUB sort
  size_t sort_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_tmp_bytes, static_cast<float *>(nullptr),
                                            static_cast<float *>(nullptr), N, 0, 32, ctx.stream));
  auto d_sort_tmp = make_unique_device_ptr<uint8_t>(sort_tmp_bytes);

  // CUB select
  size_t select_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Flagged(
      nullptr, select_tmp_bytes, static_cast<float *>(nullptr), static_cast<uint8_t *>(nullptr),
      static_cast<float *>(nullptr), static_cast<int *>(nullptr), N, ctx.stream));
  auto d_select_tmp = make_unique_device_ptr<uint8_t>(select_tmp_bytes);
  auto d_roi_count  = make_unique_device_ptr<int>(1);

  // Compute ROI mask
  auto d_roi = make_unique_device_ptr<float>(N);
  dim3 block_size(16, 16, 1);
  dim3 grid_size((W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y,
                 (Z + block_size.z - 1) / block_size.z);
  roi_mask_kernel<<<grid_size, block_size, 0, ctx.stream>>>(d_roi_mask.get(), W, H, Z,
                                                            settings.roi);

  // Success
  auto *task =
      new PctClip(settings, std::move(d_min), std::move(d_max), std::move(d_roi_mask),
                  sort_tmp_bytes, std::move(d_sort_tmp), select_tmp_bytes, std::move(d_select_tmp),
                  std::move(d_roi), std::move(d_roi_count), ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks
