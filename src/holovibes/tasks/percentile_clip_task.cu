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

    // printf("%f | %f\n", lower, upper);
  }
}

} // namespace

PercentileClipTask::PercentileClipTask(
    const TaskMeta &meta, CudaStreamRef stream,
    unique_device_ptr<float> lower_threshold,
    unique_device_ptr<float> upper_threshold,
    unique_device_ptr<uint8_t> d_temp_storage, size_t temp_storage_bytes)
    : Task(meta, stream), lower_threshold_(std::move(lower_threshold)),
      upper_threshold_(std::move(upper_threshold)),
      d_temp_storage_(std::move(d_temp_storage)),
      temp_storage_bytes_(std::move(temp_storage_bytes)) {}

tl::expected<void, Error> PercentileClipTask::run(TensorView input,
                                                  TensorView output) {
  size_t count = input.size();
  float *idata = static_cast<float *>(input.data());
  float *odata = static_cast<float *>(output.data());
  cub::DeviceRadixSort::SortKeys(d_temp_storage_.get(), temp_storage_bytes_,
                                 idata, odata, count, 0, sizeof(float) * 8,
                                 stream_.stream());

  int lower_idx = static_cast<int>(count * (0.02f / 100.0f));
  int upper_idx = static_cast<int>(count * (99.98f / 100.0f));

  cudaMemcpyAsync(lower_threshold_.get(), odata + lower_idx, sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_.stream());
  cudaMemcpyAsync(upper_threshold_.get(), odata + upper_idx, sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_.stream());

  int block_size = 256;
  int grid_size = (count + block_size - 1) / block_size;
  clip_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      idata, odata, count, lower_threshold_.get(), upper_threshold_.get());

  return {};
}

// ==========================================================================
//                     PercentileClipFactory Implementation
// ==========================================================================

tl::expected<TaskMeta, Error>
PercentileClipTaskFactory::type_check(const TensorMeta &imeta, const json &) {
  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::type_check] invalid rank \"{}\"",
        imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn("[PercentileClipTaskFactory::type_check] invalid "
                             "memory location \"{}\"",
                             (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  switch (imeta.data_type()) {
  case DataType::F32:
    break;
  default:
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::type_check] invalid input type \"{}\"",
        (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return TaskMeta(imeta, imeta, false);
}

tl::expected<std::unique_ptr<Task>, Error>
PercentileClipTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                                  CudaStreamRef stream) {

  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();

  // Temp storage size
  size_t count = imeta.size();
  float *idata = static_cast<float *>(nullptr);
  float *odata = static_cast<float *>(nullptr);
  unique_device_ptr<uint8_t> d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage.get(), temp_storage_bytes,
                                 idata, odata, count, 0, sizeof(float) * 8,
                                 stream.stream());

  // Temp storage
  auto d_temp_storage_result =
      try_make_unique_device_ptr<uint8_t>(temp_storage_bytes, stream.stream());
  if (!d_temp_storage_result) {
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::create] failed with error \"{}\"",
        CudaError(d_temp_storage_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  d_temp_storage = std::move(d_temp_storage_result.value());

  // Upper threshold
  auto d_upper_threshold_result =
      try_make_unique_device_ptr<float>(1, stream.stream());
  if (!d_upper_threshold_result) {
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::create] failed with error \"{}\"",
        CudaError(d_upper_threshold_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_upper_threshold = std::move(d_upper_threshold_result.value());

  // Lower threshold
  auto d_lower_threshold_result =
      try_make_unique_device_ptr<float>(1, stream.stream());
  if (!d_lower_threshold_result) {
    holovibes_logger()->warn(
        "[PercentileClipTaskFactory::create] failed with error \"{}\"",
        CudaError(d_lower_threshold_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_lower_threshold = std::move(d_lower_threshold_result.value());

  auto *task = new PercentileClipTask(
      meta, stream, std::move(d_lower_threshold), std::move(d_upper_threshold),
      std::move(d_temp_storage), temp_storage_bytes);

  return std::unique_ptr<PercentileClipTask>(task);
}

} // namespace dh