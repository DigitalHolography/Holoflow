#include "holovibes/tasks/fresnel_diffraction_task.hh"

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
//                     FresnelDiffractionTask Implementation
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

FresnelDiffractionTask::FresnelDiffractionTask(
    const TaskMeta &meta, CudaStreamRef stream, float lambda, float z,
    float pixel_size, bool skip_phase_shift,
    unique_device_ptr<cuFloatComplex> lens, CufftHandle handle)
    : Task(meta, stream), lambda_(lambda), z_(z), pixel_size_(pixel_size),
      skip_phase_shift_(skip_phase_shift), lens_(std::move(lens)),
      handle_(std::move(handle)) {}

tl::expected<void, Error> FresnelDiffractionTask::run(TensorView input,
                                                      TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();
  int batch_size = input.meta().shape().at(0);
  int lens_size = input.meta().size() / batch_size;

  dim3 block_size = 256;
  dim3 grid_size = (lens_size + block_size.x - 1) / block_size.x;

  apply_lens_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      idata, odata, lens_.get(), lens_size, batch_size);

  if (auto result =
          handle_.try_xt_exec(odata, odata, CufftDirection(CUFFT_FORWARD));
      !result) {
    dh::holovibes_logger()->warn("[FresnelDiffractionTask::run] Fourier "
                                 "transform failed with error: \"{}\"",
                                 result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (auto result = cudaPeekAtLastError(); result != cudaSuccess) {
    holovibes_logger()->warn("[FresnelDiffractionTask::run] Cuda error: {}",
                             cudaGetErrorString(result));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  // TODO implement phase shift
  return {};
}

// ==========================================================================
//                     FresnelDiffractionTaskFactory Implementation
// ==========================================================================

namespace {

__global__ void quadratic_lens_kernel(cuFloatComplex *lens, int width,
                                      int height, float lambda, float z,
                                      float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size = width > height ? width : height;

  // The intent with offsets is to support non-square images.
  // The are used to "center" the indexes as if the rectangle was extended to
  // a square.
  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * pixel_size;
  float y = ((row + offset_y) - size / 2.0f) * pixel_size;

  float phase = CUDART_PI_F / (lambda * z) * (x * x + y * y);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

} // namespace

tl::expected<TaskMeta, Error>
FresnelDiffractionTaskFactory::type_check(const TensorMeta &imeta,
                                          const json &jparams) {
  auto params = jparams.get<Params>();

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn("Invalid memory location: {}",
                             (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::CF32) {
    holovibes_logger()->warn("Invalid data type: {}", (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn("Invalid rank: {}", (int)imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.lambda <= 0) {
    holovibes_logger()->warn("Invalid lambda: {}", params.lambda);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.pixel_size <= 0) {
    holovibes_logger()->warn("Invalid pixel_size: {}", params.pixel_size);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.z <= 0) {
    holovibes_logger()->warn("Invalid z: {}", params.z);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (!params.skip_phase_shift) {
    holovibes_logger()->warn("Unimplemented skip_phase_shift: {}",
                             params.skip_phase_shift);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return TaskMeta(imeta, imeta, true);
}

tl::expected<std::unique_ptr<Task>, Error>
FresnelDiffractionTaskFactory::create(const TensorMeta &imeta,
                                      const json &jparams,
                                      CudaStreamRef stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("type check failed");
    return tl::unexpected(meta_result.error());
  }

  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  int batch = imeta.shape().at(0);
  int height = imeta.shape().at(1);
  int width = imeta.shape().at(2);

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
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             plan_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto handle = std::move(plan_result.value());

  size_t work_size = 0;
  if (auto result = handle.try_xt_get_size_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (auto result = handle.try_xt_make_plan_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  // auto plan_result = CufftHandle::try_xt_make_plan_many(
  //     rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
  //     outputtype, batch, executiontype);
  // if (!plan_result) {
  //   holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create]
  //   Fourrier "
  //                            "transform creation failed with error: \"{}\"",
  //                            plan_result.error());
  //   return tl::unexpected(Error::INTERNAL_ERROR);
  // }
  // auto handle = std::move(plan_result.value());

  auto stream_result = handle.try_set_stream(stream);
  if (!stream_result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             stream_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  // Initialize lens
  auto d_lens_result = try_make_unique_device_ptr<cuFloatComplex>(
      imeta.size_in_bytes() / batch, stream.stream());
  if (!d_lens_result) {
    holovibes_logger()->warn("[FresnelDiffractionTask::create] Cuda error: {}",
                             cudaGetErrorString(d_lens_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_lens = std::move(d_lens_result.value());

  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  quadratic_lens_kernel<<<grid_size, block_size, 0, stream.stream()>>>(
      d_lens.get(), width, height, params.lambda, params.z, params.pixel_size);

  if (auto result = cudaPeekAtLastError(); result != cudaSuccess) {
    holovibes_logger()->warn("[FresnelDiffractionTask::create] Cuda error: {}",
                             cudaGetErrorString(result));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto *task = new FresnelDiffractionTask(meta, stream, params.lambda, params.z,
                                          params.pixel_size, true,
                                          std::move(d_lens), std::move(handle));
  return std::unique_ptr<FresnelDiffractionTask>(task);
}

} // namespace dh