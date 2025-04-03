#include "holovibes/tasks/fft_shift_task.hh"

#include <spdlog/spdlog.h>

#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     FresnelDiffractionTask Implementation
// ==========================================================================

namespace {

__global__ void swap_corners_kernel(float *in, float *out, int width,
                                    int height, int batch_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int width_half = width / 2;
  int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch_size) {
    return;
  }

  int batch_offset = z * width * height;
  float *in_frame = in + batch_offset;
  float *out_frame = out + batch_offset;

  // --- Swap top-left with bottom-right ---
  int top_left_idx = x + y * width;
  int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  float tmp = in_frame[top_left_idx];
  out_frame[top_left_idx] = in_frame[bottom_right_idx];
  out_frame[bottom_right_idx] = tmp;

  // --- Swap top-right with bottom-left ---
  int top_right_idx = (x + width_half) + y * width;
  int bottom_left_idx = x + (y + height_half) * width;

  tmp = in_frame[top_right_idx];
  out_frame[top_right_idx] = in_frame[bottom_left_idx];
  out_frame[bottom_left_idx] = tmp;
}

} // namespace

FFTShiftTask::FFTShiftTask(const TaskMeta &meta, CudaStreamRef stream)
    : Task(meta, stream) {}

tl::expected<void, Error> FFTShiftTask::run(TensorView input,
                                            TensorView output) {
  auto idata = static_cast<float *>(input.data());
  auto odata = static_cast<float *>(output.data());

  int batch_size = input.meta().shape().at(0);
  int height = input.meta().shape().at(1);
  int width = input.meta().shape().at(2);

  int width_half = width / 2;
  int height_half = height / 2;

  dim3 block_size(16, 16, 1);
  dim3 grid_size((width_half + block_size.x - 1) / block_size.x,
                 (height_half + block_size.y - 1) / block_size.y,
                 (batch_size + block_size.z - 1) / block_size.z);

  swap_corners_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      idata, odata, width, height, batch_size);

  return {};
}

// ==========================================================================
//                     FresnelDiffractionTaskFactory Implementation
// ==========================================================================

tl::expected<TaskMeta, Error>
FFTShiftTaskFactory::type_check(const TensorMeta &imeta, const json &) {
  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn(
        "[FFTShiftTaskFactory::type_check] invalid memory location: {}",
        (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::F32) {
    holovibes_logger()->warn(
        "[FFTShiftTaskFactory::type_check] invalid data type: {}",
        (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn(
        "[FFTShiftTaskFactory::type_check] invalid rank: {}",
        (int)imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return TaskMeta(imeta, imeta, true);
}

tl::expected<std::unique_ptr<Task>, Error>
FFTShiftTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                            CudaStreamRef stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("[PCATaskFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();

  auto *task = new FFTShiftTask(meta, stream);
  return std::unique_ptr<FFTShiftTask>(task);
}

} // namespace dh