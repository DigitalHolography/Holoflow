#include "holovibes/tasks/angular_spectrum_task.hh"

#include <cassert>
#include <cstdlib>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <numeric>
#include <spdlog/spdlog.h>

#include "curaii/v2/cuda.hh"
#include "curaii/v2/cufft.hh"
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
                                         curaii::cufft::Handle handle)
    : Task(meta, stream), lambda_(lambda), z_(z), pixel_size_(pixel_size),
      lens_(std::move(lens)), handle_(std::move(handle)) {}

void AngularSpectrumTask::run(TensorView input, TensorView output) {
  // 1) Aliase
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();
  int B = input.meta().shape().at(0);
  int frame_size = input.meta().size() / B;

  // 2) Forward FFT
  CUFFT_CHECK(cufftXtExec(handle_.get(), idata, odata, CUFFT_FORWARD));

  // 3) Apply lens in fourrier space
  dim3 block_size = 256;
  dim3 grid_size = (frame_size + block_size.x - 1) / block_size.x;

  apply_lens_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      odata, odata, lens_.get(), frame_size, B);

  // 4) Backward FFT
  CUFFT_CHECK(cufftXtExec(handle_.get(), odata, odata, CUFFT_INVERSE));
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

TaskMeta AngularSpectrumTaskFactory::type_check(const TensorMeta &imeta,
                                                const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.lambda > 0, "lambda <= 0");
  check(params.pixel_size > 0, "pixel_size <= 0");
  check(params.z > 0, "z <= 0");

  // 2) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.data_type() == DataType::CF32, "tensor data_type != CF32");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  return TaskMeta(imeta, imeta, true);
}

std::unique_ptr<Task>
AngularSpectrumTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                                   CudaStreamRef stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  const int B = static_cast<int>(imeta.shape()[0]);
  const int H = static_cast<int>(imeta.shape()[1]);
  const int W = static_cast<int>(imeta.shape()[2]);

  // 2) Buffer sizes
  const int frame_size = imeta.size_in_bytes() / B;

  // 3) Allocations
  auto d_lens =
      make_unique_device_ptr<cuFloatComplex>(frame_size, stream.stream());

  // 4) Compute lens
  dim3 block_size(16, 16);
  dim3 grid_size((W + block_size.x - 1) / block_size.x,
                 (H + block_size.y - 1) / block_size.y);

  spectral_lens_kernel<<<grid_size, block_size, 0, stream.stream()>>>(
      d_lens.get(), W, H, params.lambda, params.z, params.pixel_size);

  swap_corners_kernel<<<grid_size, block_size, 0, stream.stream()>>>(
      d_lens.get(), d_lens.get(), W, H, 1);

  CUDA_CHECK(cudaPeekAtLastError());

  // 5) Initialize cufft plan
  int rank = 2;
  long long int n[2] = {H, W};
  long long int inembed[2] = {H, W};
  int istride = 1;
  int idist = H * W;
  cudaDataType inputtype = CUDA_C_32F;
  long long int onembed[2] = {H, W};
  int ostride = 1;
  int odist = H * W;
  cudaDataType outputtype = CUDA_C_32F;
  int batch = B;
  size_t work_size = 0;
  cudaDataType executiontype = CUDA_C_32F;

  curaii::cufft::Handle handle;
  CUFFT_CHECK(cufftSetStream(handle.get(), stream.stream()));
  CUFFT_CHECK(cufftXtGetSizeMany(handle.get(), rank, n, inembed, istride, idist,
                                 inputtype, onembed, ostride, odist, outputtype,
                                 batch, &work_size, executiontype));
  CUFFT_CHECK(cufftXtMakePlanMany(
      handle.get(), rank, n, inembed, istride, idist, inputtype, onembed,
      ostride, odist, outputtype, batch, &work_size, executiontype));

  // 6) Assemble task
  auto *task = new AngularSpectrumTask(meta, stream, params.lambda, params.z,
                                       params.pixel_size, std::move(d_lens),
                                       std::move(handle));
  return std::unique_ptr<AngularSpectrumTask>(task);
}

} // namespace dh