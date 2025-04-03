#include "holovibes/tasks/average_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "curaii/cuda_runtime.hh"
#include "holovibes/holovibes.hh"

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    dh::holovibes_logger()->critical("Unreachable code reached at {}:{} - {}", \
                                     __FILE__, __LINE__, msg);                 \
    std::abort();                                                              \
  } while (0)

namespace dh {

// ==========================================================================
//                     AverageTask Implementation
// ==========================================================================

namespace {

__global__ void u8_avg_kernel(const uint8_t *idata, uint8_t *odata, int nx,
                              int ny, int nz) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  uint32_t sum = 0;

  for (int z = 0; z < nz; z++) {
    int z_offset = z * ny * nx;
    sum += idata[z_offset + idx];
  }

  odata[idx] = sum / nz;
}

__global__ void f32_avg_kernel(const float *idata, float *odata, int nx, int ny,
                               int nz) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  float sum = 0.0f;

  for (int z = 0; z < nz; z++) {
    int z_offset = z * ny * nx;
    sum += idata[z_offset + idx];
  }

  odata[idx] = sum / nz;
}

__global__ void cf32_avg_kernel(const cuFloatComplex *idata,
                                cuFloatComplex *odata, int nx, int ny, int nz) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

  for (int z = 0; z < nz; z++) {
    int z_offset = z * ny * nx;
    sum = cuCaddf(sum, idata[z_offset + idx]);
  }

  odata[idx] = make_cuFloatComplex(sum.x / nz, sum.y / nz);
}

} // namespace

AverageTask::AverageTask(const TaskMeta &meta, cudaStream_t stream, int begin,
                         int end, Kind kind)
    : Task(meta, stream), begin_(begin), end_(end), kind_(kind) {}

tl::expected<void, Error> AverageTask::run(TensorView input,
                                           TensorView output) {
  size_t batch = input.meta().shape().at(0);
  off_t offset = begin_ * input.meta().size_in_bytes() / batch;
  void *idata = (uint8_t *)input.data() + offset;
  void *odata = output.data();
  int nx = input.meta().shape().at(2);
  int ny = input.meta().shape().at(1);
  int nz = end_ - begin_;
  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x,
                 (ny + block_size.y - 1) / block_size.y);

  switch (kind_) {
  case Kind::U8_AVG:
    u8_avg_kernel<<<grid_size, block_size, 0, stream_>>>(
        (uint8_t *)idata, (uint8_t *)odata, nx, ny, nz);
    break;
  case Kind::F32_AVG:
    f32_avg_kernel<<<grid_size, block_size, 0, stream_>>>(
        (float *)idata, (float *)odata, nx, ny, nz);
    break;
  case Kind::CF32_AVG:
    cf32_avg_kernel<<<grid_size, block_size, 0, stream_>>>(
        (cuFloatComplex *)idata, (cuFloatComplex *)odata, nx, ny, nz);
    break;
  default:
    UNREACHABLE("invalid average task kind");
  }

  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    holovibes_logger()->warn("[AverageTask::run] failed with error \"{}\"",
                             CudaError(error));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return {};
}

// ==========================================================================
//                     AverageTask Implementation
// ==========================================================================

tl::expected<TaskMeta, Error>
AverageTaskFactory::type_check(const TensorMeta &imeta, const json &jparams) {
  auto params = jparams.get<Params>();

  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn(
        "[AverageTaskFactory::type_check] invalid rank \"{}\"",
        imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn(
        "[AverageTaskFactory::type_check] invalid memory location \"{}\"",
        (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.begin < 0) {
    holovibes_logger()->warn(
        "[AverageTaskFactory::type_check] invalid param (begin < 0) \"{}\"",
        params.begin);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.begin >= params.end) {
    holovibes_logger()->warn("[AverageTaskFactory::type_check] invalid param "
                             "(begin >= end) \"{}\" \"{}\"",
                             params.begin, params.end);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.end > imeta.shape().at(0)) {
    holovibes_logger()->warn("[AverageTaskFactory::type_check] invalid param "
                             "(end > input batch size) \"{}\" \"{}\"",
                             params.end, imeta.shape().at(0));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  switch (imeta.data_type()) {
  case DataType::U8:
  case DataType::F32:
  case DataType::CF32:
    break;
  default:
    holovibes_logger()->warn(
        "[AverageTaskFactory::type_check] invalid input type \"{}\"",
        (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto oshape = imeta.shape();
  oshape.at(0) = 1;
  TensorMeta ometa(imeta.data_type(), MemoryLocation::DEVICE, oshape);
  return TaskMeta(imeta, ometa, false);
}

tl::expected<std::unique_ptr<Task>, Error>
AverageTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                           cudaStream_t stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("[AverageTaskFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  AverageTask::Kind kind;
  switch (imeta.data_type()) {
  case DataType::U8:
    kind = AverageTask::Kind::U8_AVG;
    break;
  case DataType::F32:
    kind = AverageTask::Kind::F32_AVG;
    break;
  case DataType::CF32:
    kind = AverageTask::Kind::CF32_AVG;
    break;
  default:
    UNREACHABLE("invalid data type was checked before");
  }

  auto *task = new AverageTask(meta, stream, params.begin, params.end, kind);
  return std::unique_ptr<AverageTask>(task);
}

} // namespace dh