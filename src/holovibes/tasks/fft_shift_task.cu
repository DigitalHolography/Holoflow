#include "holovibes/tasks/fft_shift_task.hh"

#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
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

void FFTShiftTask::run(TensorView input, TensorView output) {
  // 1) Aliases
  auto idata = static_cast<float *>(input.data());
  auto odata = static_cast<float *>(output.data());
  const int B = input.meta().shape().at(0);
  const int H = input.meta().shape().at(1);
  const int W = input.meta().shape().at(2);
  int W_h = W / 2;
  int H_h = H / 2;

  // 2) Swap corners
  dim3 block_size(16, 16, 1);
  dim3 grid_size((W_h + block_size.x - 1) / block_size.x,
                 (H_h + block_size.y - 1) / block_size.y,
                 (B + block_size.z - 1) / block_size.z);

  swap_corners_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      idata, odata, W, H, B);

  CUDA_CHECK(cudaPeekAtLastError());
}

// ==========================================================================
//                     FresnelDiffractionTaskFactory Implementation
// ==========================================================================

TaskMeta FFTShiftTaskFactory::type_check(const TensorMeta &imeta,
                                         const json &) {
  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.data_type() == DataType::F32, "tensor data_type != F32");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  // 2) Success
  return TaskMeta(imeta, imeta, true);
}

std::unique_ptr<Task> FFTShiftTaskFactory::create(const TensorMeta &imeta,
                                                  const json &jparams,
                                                  CudaStreamRef stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);

  // 2) Assemble task
  auto *task = new FFTShiftTask(meta, stream);
  return std::unique_ptr<FFTShiftTask>(task);
}

} // namespace dh