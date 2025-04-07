#include "holovibes/tasks/angular_spectrum_task.hh"

#include <cassert>
#include <cstdlib>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <numeric>
#include <spdlog/spdlog.h>

#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     AngularSpectrumTask Implementation
// ==========================================================================

namespace {

__global__ void apply_lens_kernel(cuFloatComplex *idata, cuFloatComplex *odata,
                                  const cuFloatComplex *lens, int lens_size,
                                  int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= lens_size)
    return;

  cuFloatComplex val = lens[idx];
  for (int i = 0; i < batch_size; i++) {
    odata[i * lens_size + idx] = cuCmulf(idata[i * lens_size + idx], val);
  }
}

} // namespace

AngularSpectrumTask::AngularSpectrumTask(const TaskMeta &meta,
                                         CudaStreamRef stream, float lambda,
                                         float z, float pixel_size,
                                         unique_device_ptr<cuFloatComplex> lens,
                                         CufftHandle handle)
    : Task(meta, stream), lambda_(lambda), z_(z), pixel_size_(pixel_size),
      lens_(std::move(lens)), handle_(std::move(handle)) {}

tl::expected<void, Error> AngularSpectrumTask::run(TensorView input,
                                                   TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();
  int batch_size = input.meta().shape().at(0);
  int lens_size = input.meta().size() / batch_size;

  if (auto result =
          handle_.try_xt_exec(idata, odata, CufftDirection(CUFFT_FORWARD));
      !result) {
    dh::holovibes_logger()->warn("[AngularSpectrumTask::run] Fourier "
                                 "transform failed with error: \"{}\"",
                                 result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  dim3 block_size = 256;
  dim3 grid_size = (lens_size + block_size.x - 1) / block_size.x;

  apply_lens_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      idata, odata, lens_.get(), lens_size, batch_size);

  if (auto result =
          handle_.try_xt_exec(idata, odata, CufftDirection(CUFFT_INVERSE));
      !result) {
    dh::holovibes_logger()->warn("[AngularSpectrumTask::run] Fourier "
                                 "transform failed with error: \"{}\"",
                                 result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return {};
}

// ==========================================================================
//                     AngularSpectrumTaskFactory Implementation
// ==========================================================================

namespace {

__global__ void spectral_lens_kernel(cuFloatComplex *lens, int width,
                                     int height, float lambda, float z,
                                     float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  float du = 1.0f / (width * pixel_size);
  float dv = 1.0f / (height * pixel_size);
  float u = (col - width / 2) * du;
  float v = (row - height / 2) * dv;

  float tmp = 1.0f - (lambda * lambda) * (u * u + v * v);
  tmp = fmaxf(tmp, 0.0f);

  float phase = 2.0f * CUDART_PI_F * z / lambda * sqrt(tmp);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

__global__ void swap_corners_kernel(cuFloatComplex *in, cuFloatComplex *out,
                                    int width, int height, int batch_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int width_half = width / 2;
  int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch_size) {
    return;
  }

  int batch_offset = z * width * height;
  cuFloatComplex *in_frame = in + batch_offset;
  cuFloatComplex *out_frame = out + batch_offset;

  // --- Swap top-left with bottom-right ---
  int top_left_idx = x + y * width;
  int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  cuFloatComplex tmp = in_frame[top_left_idx];
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

tl::expected<TaskMeta, Error>
AngularSpectrumTaskFactory::type_check(const TensorMeta &imeta,
                                       const json &jparams) {
  auto params = jparams.get<Params>();

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid memory location: {}",
        (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::CF32) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid data type: {}",
        (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid rank: {}",
        (int)imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.lambda <= 0) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid lambda: {}",
        params.lambda);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.pixel_size <= 0) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid pixel_size: {}",
        params.pixel_size);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.z <= 0) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::type_check] Invalid z: {}", params.z);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return TaskMeta(imeta, imeta, true);
}

tl::expected<std::unique_ptr<Task>, Error>
AngularSpectrumTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                                   CudaStreamRef stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }

  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  int batch = imeta.shape().at(0);
  int height = imeta.shape().at(1);
  int width = imeta.shape().at(2);

  // Initialize lens
  auto d_lens_result = try_make_unique_device_ptr<cuFloatComplex>(
      imeta.size_in_bytes() / batch, stream.stream());
  if (!d_lens_result) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::create] Cuda error: {}",
        cudaGetErrorString(d_lens_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_lens = std::move(d_lens_result.value());

  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  spectral_lens_kernel<<<grid_size, block_size, 0, stream.stream()>>>(
      d_lens.get(), width, height, params.lambda, params.z, params.pixel_size);

  swap_corners_kernel<<<grid_size, block_size, 0, stream.stream()>>>(
      d_lens.get(), d_lens.get(), width, height, 1);

  if (auto result = cudaPeekAtLastError(); result != cudaSuccess) {
    holovibes_logger()->warn(
        "[AngularSpectrumTaskFactory::create] Cuda error: {}",
        cudaGetErrorString(result));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  // Initialize cufft plan
  int rank = 2;
  long long int n[2] = {height, width};
  long long int inembed[2] = {height, width};
  int istride = 1;
  int idist = height * width;
  CudaDataType inputtype(CUDA_C_32F);
  long long int onembed[2] = {height, width};
  int ostride = 1;
  int odist = height * width;
  CudaDataType outputtype(CUDA_C_32F);
  CudaDataType executiontype(CUDA_C_32F);

  auto plan_result = CufftHandle::try_create();
  if (!plan_result) {
    holovibes_logger()->warn("[AngularSpectrumTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             plan_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto handle = std::move(plan_result.value());

  auto stream_result = handle.try_set_stream(stream);
  if (!stream_result) {
    holovibes_logger()->warn("[AngularSpectrumTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             stream_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  size_t work_size = 0;
  if (auto result = handle.try_xt_get_size_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[AngularSpectrumTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (auto result = handle.try_xt_make_plan_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[AngularSpectrumTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto *task = new AngularSpectrumTask(meta, stream, params.lambda, params.z,
                                       params.pixel_size, std::move(d_lens),
                                       std::move(handle));
  return std::unique_ptr<AngularSpectrumTask>(task);
}

} // namespace dh