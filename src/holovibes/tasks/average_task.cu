#include "holovibes/tasks/average_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
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

void AverageTask::run(TensorView input, TensorView output) {
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
    DH_BUG("unreachable statement reached");
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

// ==========================================================================
//                     AverageTask Implementation
// ==========================================================================

TaskMeta AverageTaskFactory::type_check(const TensorMeta &imeta,
                                        const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.begin >= 0, "begin < 0");
  check(params.begin < params.end, "begin >= end");

  // 2) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(params.end <= imeta.shape().at(0), "end > tensor dim 0");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  switch (imeta.data_type()) {
  case DataType::U8:
  case DataType::F32:
  case DataType::CF32:
    break;
  default:
    throw std::invalid_argument("tensor datatype not in [U8, F32, CF32]");
  }

  // 3) Success
  auto oshape = imeta.shape();
  oshape.at(0) = 1;
  TensorMeta ometa(imeta.data_type(), MemoryLocation::DEVICE, oshape);
  return TaskMeta(imeta, ometa, false);
}

std::unique_ptr<Task> AverageTaskFactory::create(const TensorMeta &imeta,
                                                 const json &jparams,
                                                 cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
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

  // 6) Assemble task
  auto *task = new AverageTask(meta, stream, params.begin, params.end, kind);
  return std::unique_ptr<AverageTask>(task);
}

} // namespace dh