#include "holovibes/tasks/percentile_clip_task.hh"

#include <cassert>
#include <cstdlib>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <numeric>
#include <spdlog/spdlog.h>

#include "curaii/cuda_runtime.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     PercentileClip Implementation
// ==========================================================================

namespace {

__global__ void clip_kernel(const float *input, float *output, int count,
                            float *d_lower, float *d_upper) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    float lower = *d_lower > 1.0f ? *d_lower : 1.0f;
    float upper = *d_upper > 1.0f ? *d_upper : 1.0f;
    float value = input[idx];
    output[idx] = (value < lower) ? lower : ((value > upper) ? upper : value);
  }
}

} // namespace

PercentileClipTask::PercentileClipTask(const TaskMeta &meta,
                                       CudaStreamRef stream,
                                       unique_device_ptr<float> d_lower_thresh,
                                       unique_device_ptr<float> d_upper_thresh,
                                       unique_device_ptr<uint8_t> d_sort_tmp,
                                       unique_device_ptr<float> d_roi_values,
                                       unique_device_ptr<uint8_t> d_select_tmp,
                                       unique_device_ptr<uint8_t> d_roi_mask,
                                       unique_device_ptr<int> d_roi_count,
                                       size_t sort_tmp_bytes,
                                       size_t select_tmp_bytes, float pct_low,
                                       float pct_high, float roi_radius)
    : Task(meta, stream), d_lower_thresh_(std::move(d_lower_thresh)),
      d_upper_thresh_(std::move(d_upper_thresh)),
      d_sort_tmp_(std::move(d_sort_tmp)),
      d_select_tmp_(std::move(d_select_tmp)),
      d_roi_values_(std::move(d_roi_values)),
      d_roi_mask_(std::move(d_roi_mask)), d_roi_count_(std::move(d_roi_count)),
      sort_tmp_bytes_(sort_tmp_bytes), select_tmp_bytes_(select_tmp_bytes),
      pct_low_(pct_low), pct_high_(pct_high), roi_radius_(roi_radius) {}

tl::expected<void, Error> PercentileClipTask::run(TensorView input,
                                                  TensorView output) {
  // 1) Aliases
  const size_t total_px = input.size();
  float *d_in = static_cast<float *>(input.data());   // full frame
  float *d_out = static_cast<float *>(output.data()); // clip result

  // 2) Filter ellipse ROI pixels
  cub::DeviceSelect::Flagged(d_select_tmp_.get(), // d_temp_storage
                             select_tmp_bytes_,   // temp_storage_byts
                             d_in,                // d_in
                             d_roi_mask_.get(),   // d_flags
                             d_roi_values_.get(), // d_out
                             d_roi_count_.get(),  // d_num_selected
                             total_px,            // num_items
                             stream_.stream());   // stream

  // 3) Fetch how many pixels we kept
  int roi_px = 0;
  cudaMemcpyAsync(&roi_px, d_roi_count_.get(), sizeof(int),
                  cudaMemcpyDeviceToHost, stream_.stream());
  cudaStreamSynchronize(stream_.stream());

  if (roi_px == 0)
    return tl::unexpected(Error::INTERNAL_ERROR);

  // 4) Sort ROI values to d_out (will be overwritten later)
  cub::DeviceRadixSort::SortKeys(d_sort_tmp_.get(),   // d_temp_storage
                                 sort_tmp_bytes_,     // temp_storage_bytes
                                 d_roi_values_.get(), // d_keys_in
                                 d_out,               // d_keys_out
                                 roi_px,              // num_items
                                 0,                   // begin_bit
                                 sizeof(float) * 8,   // end_bit
                                 stream_.stream());   // stream

  // 5) Percentile indices inside ROI
  const int idx_low = static_cast<int>(roi_px * (pct_low_ / 100.f));
  const int idx_high = static_cast<int>(roi_px * (pct_high_ / 100.f));

  // 6) Copy thresholds
  cudaMemcpyAsync(d_lower_thresh_.get(), d_out + idx_low, sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_.stream());
  cudaMemcpyAsync(d_upper_thresh_.get(), d_out + idx_high, sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_.stream());

  // 7) Clip the frame
  constexpr int block = 256;
  const int grid = static_cast<int>((total_px + block - 1) / block);

  clip_kernel<<<grid, block, 0, stream_.stream()>>>(
      d_in, d_out, total_px, d_lower_thresh_.get(), d_upper_thresh_.get());

  return {};
}

// ==========================================================================
//                     PercentileClipFactory Implementation
// ==========================================================================

namespace {

__global__ void ellipse_mask_kernel(uint8_t *flagged, int width, int height,
                                    int batch, float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || z >= batch) {
    return;
  }

  const float cx = 0.5f * (width - 1);
  const float cy = 0.5f * (height - 1);

  const float sx = static_cast<float>(width) / static_cast<float>(width);
  const float sy = static_cast<float>(height) / static_cast<float>(height);

  const float r_square = radius * 0.5f * static_cast<float>(min(width, height));
  const float rx2 = (r_square / sx) * (r_square / sx);
  const float ry2 = (r_square / sy) * (r_square / sy);

  const float dx = static_cast<float>(x) - cx;
  const float dy = static_cast<float>(y) - cy;
  const bool inside = (dx * dx) / rx2 + (dy * dy) / ry2 <= 1.0f;

  const size_t idx = static_cast<size_t>(z) * height * width +
                     static_cast<size_t>(y) * width + static_cast<size_t>(x);

  flagged[idx] = static_cast<uint8_t>(inside);
}

} // namespace

tl::expected<TaskMeta, Error>
PercentileClipTaskFactory::type_check(const TensorMeta &imeta,
                                      const json &jparams) {
  try {
    // 0) Unpack parameters
    const Params params = jparams.get<Params>();

    const auto check = [&](bool cond, std::string_view what) {
      if (!cond) {
        throw std::runtime_error(std::string(what));
      }
    };

    // 1) Parameter sanity
    check(params.lower_percentile >= 0.0f && params.lower_percentile <= 100.0f,
          "lower_percentile out of [0,100]");

    check(params.upper_percentile >= 0.0f && params.upper_percentile <= 100.0f,
          "upper_percentile out of [0,100]");

    check(params.lower_percentile < params.upper_percentile,
          "lower_percentile >= upper_percentile");

    check(params.radius >= 0.0f && params.radius <= 1.0f,
          "radius out of [0,1]");

    // 2) Tensor meta sanity
    check(imeta.shape().size() == 3, "tensor rank != 3");

    check(imeta.memory_location() == MemoryLocation::DEVICE,
          "tensor not in DEVICE memory");

    check(imeta.data_type() == DataType::F32, "tensor data_type != F32");

    // 3) Success
    return TaskMeta(imeta, imeta, false);
  }

  catch (const std::exception &e) {
    holovibes_logger()->warn("[PercentileClipTaskFactory::type_check] {}",
                             e.what());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
}

tl::expected<std::unique_ptr<Task>, Error>
PercentileClipTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                                  CudaStreamRef stream) {
  // 1) Validate
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result)
    return tl::unexpected(meta_result.error());

  const auto meta = meta_result.value();
  const auto params = jparams.get<Params>();

  const int B = static_cast<int>(imeta.shape()[0]);
  const int H = static_cast<int>(imeta.shape()[1]);
  const int W = static_cast<int>(imeta.shape()[2]);
  const size_t N = imeta.size();

  // 2) CUB temp size
  size_t sort_tmp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, sort_tmp_bytes, // d_temp_storage
                                 static_cast<float *>(nullptr), // d_keys_in
                                 static_cast<float *>(nullptr), // d_keys_out
                                 N,                             // num_items
                                 0,                             // begin_bit
                                 sizeof(float) * 8,             // end_bit
                                 stream.stream());              // stream

  size_t select_tmp_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, select_tmp_bytes,       // d_temp_storage
                             static_cast<float *>(nullptr),   // d_in
                             static_cast<uint8_t *>(nullptr), // d_flags
                             static_cast<float *>(nullptr),   // d_out
                             static_cast<int *>(nullptr),     // d_num_selected
                             N,                               // num_items
                             stream.stream());                // stream

  // 3) Allocations
  auto d_sort_tmp =
      make_unique_device_ptr<uint8_t>(sort_tmp_bytes, stream.stream());
  auto d_select_tmp =
      make_unique_device_ptr<uint8_t>(select_tmp_bytes, stream.stream());

  auto d_upper_thr = make_unique_device_ptr<float>(1, stream.stream());
  auto d_lower_thr = make_unique_device_ptr<float>(1, stream.stream());

  auto d_flags = make_unique_device_ptr<uint8_t>(N, stream.stream());
  auto d_roi = make_unique_device_ptr<float>(N, stream.stream());
  auto d_n_selected = make_unique_device_ptr<int>(1, stream.stream());

  // 4) Build ellipse mask
  const dim3 block(16, 16, 1);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y,
                  (B + block.z - 1) / block.z);

  ellipse_mask_kernel<<<grid, block, 0, stream.stream()>>>(d_flags.get(), W, H,
                                                           B, params.radius);

  // 5) Assemble task
  auto *task = new PercentileClipTask(
      meta, stream, std::move(d_lower_thr), std::move(d_upper_thr),
      std::move(d_sort_tmp), std::move(d_roi), std::move(d_select_tmp),
      std::move(d_flags), std::move(d_n_selected), sort_tmp_bytes,
      select_tmp_bytes, params.lower_percentile, params.upper_percentile,
      params.radius);

  return std::unique_ptr<PercentileClipTask>(task);
}

} // namespace dh