#include "holovibes/tasks/convert_task.hh"

#include <cassert>
#include <cstdlib>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <numeric>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <unordered_set>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     ConvertTask Implementation
// ==========================================================================

ConvertTask::ConvertTask(
    const TaskMeta &meta, cudaStream_t stream, Conversion conv,
    size_t min_temp_storage_bytes,
    curaii::cuda::unique_device_ptr<uint8_t> d_min_temp_storage,
    curaii::cuda::unique_device_ptr<uint8_t> d_min,
    size_t max_temp_storage_bytes,
    curaii::cuda::unique_device_ptr<uint8_t> d_max_temp_storage,
    curaii::cuda::unique_device_ptr<uint8_t> d_max)
    : Task(meta, stream), conversion_(conv),
      min_temp_storage_bytes_(min_temp_storage_bytes),
      d_min_temp_storage_(std::move(d_min_temp_storage)),
      d_min_(std::move(d_min)), max_temp_storage_bytes_(max_temp_storage_bytes),
      d_max_temp_storage_(std::move(d_max_temp_storage)),
      d_max_(std::move(d_max)) {}

namespace {

__global__ void u8_f32_kernel(const uint8_t *idata, float *odata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = static_cast<float>(idata[idx]);
}

__global__ void u8_cf32_real_kernel(const uint8_t *idata, cuFloatComplex *odata,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx].x = static_cast<float>(idata[idx]);
  odata[idx].y = 0.0f;
}

__global__ void u16_cf32_real_kernel(const uint16_t *idata,
                                     cuFloatComplex *odata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx].x = static_cast<float>(idata[idx]);
  odata[idx].y = 0.0f;
}

__global__ void f32_u8_scaled_kernel(const float *idata, uint8_t *odata,
                                     int size, float *min, float *max) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  float scale = 255.0f / (*max - *min);
  int val = static_cast<int>((idata[idx] - *min) * scale + 0.5f);
  odata[idx] = (val > 255) ? 255 : ((val < 0) ? 0 : val);
}

__global__ void f32_u16_scaled_kernel(const float *idata, uint16_t *odata,
                                      int size, float *min, float *max) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  float scale = 65535.0f / (*max - *min);
  int val = static_cast<int>((idata[idx] - *min) * scale + 0.5f);
  odata[idx] = (val > 65535) ? 65535 : ((val < 0) ? 0 : val);
}

__global__ void cf32_f32_modu_kernel(const cuFloatComplex *idata, float *odata,
                                     int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = cuCabsf(idata[idx]);
}

__global__ void cf32_f32_argu_kernel(const cuFloatComplex *idata, float *odata,
                                     int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = atan2f(idata[idx].y, idata[idx].x);
}

} // namespace

void ConvertTask::run(TensorView input, TensorView output) {
  void *d_min = d_min_.get();
  void *d_max = d_max_.get();
  void *idata = input.data();
  void *odata = output.data();
  int size = input.meta().size();
  dim3 block_size = 256;
  dim3 grid_size = (size + block_size.x - 1) / block_size.x;

  switch (conversion_) {
  case Conversion::U8_F32:
    u8_f32_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<uint8_t *>(idata), static_cast<float *>(odata), size);
    break;
  case Conversion::U8_CF32_REAL:
    u8_cf32_real_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<uint8_t *>(idata), static_cast<cuFloatComplex *>(odata),
        size);
    break;
  case Conversion::U16_CF32_REAL:
    u16_cf32_real_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<uint16_t *>(idata), static_cast<cuFloatComplex *>(odata),
        size);
    break;

  case Conversion::F32_U8_SCALED:
    CUDA_CHECK(cub::DeviceReduce::Min(
        d_min_temp_storage_.get(), min_temp_storage_bytes_,
        static_cast<float *>(idata), static_cast<float *>(d_min), size,
        stream_));
    CUDA_CHECK(cub::DeviceReduce::Max(
        d_max_temp_storage_.get(), max_temp_storage_bytes_,
        static_cast<float *>(idata), static_cast<float *>(d_max), size,
        stream_));
    f32_u8_scaled_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<float *>(idata), static_cast<uint8_t *>(odata), size,
        static_cast<float *>(d_min), static_cast<float *>(d_max));
    break;

  case Conversion::F32_U16_SCALED:
    CUDA_CHECK(cub::DeviceReduce::Min(
        d_min_temp_storage_.get(), min_temp_storage_bytes_,
        static_cast<float *>(idata), static_cast<float *>(d_min), size,
        stream_));
    CUDA_CHECK(cub::DeviceReduce::Max(
        d_max_temp_storage_.get(), max_temp_storage_bytes_,
        static_cast<float *>(idata), static_cast<float *>(d_max), size,
        stream_));
    f32_u16_scaled_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<float *>(idata), static_cast<uint16_t *>(odata), size,
        static_cast<float *>(d_min), static_cast<float *>(d_max));
    break;

  case Conversion::CF32_F32_MODU:
    cf32_f32_modu_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<cuFloatComplex *>(idata), static_cast<float *>(odata),
        size);
    break;

  case Conversion::CF32_F32_ARGU:
    cf32_f32_argu_kernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<cuFloatComplex *>(idata), static_cast<float *>(odata),
        size);
    break;

  default:
    DH_BUG("unreachable statement reached");
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

// ==========================================================================
//                     ConvertTaskFactory Implementation
// ==========================================================================

TaskMeta ConvertTaskFactory::type_check(const TensorMeta &imeta,
                                        const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  const std::unordered_set<std::string> valid_conversions = {
      "U8_F32",         "U8_CF32_REAL",  "U16_CF32_REAL", "F32_U8_SCALED",
      "F32_U16_SCALED", "CF32_F32_MODU", "CF32_F32_ARGU"};
  check(valid_conversions.contains(params.conversion), "conversion invalid");

  // 2) Tensor meta sanity
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");
  const std::unordered_map<std::string, DataType> expected_types = {
      {"U8_F32", DataType::U8},          {"U8_CF32_REAL", DataType::U8},
      {"U16_CF32_REAL", DataType::U16},  {"F32_U8_SCALED", DataType::F32},
      {"F32_U16_SCALED", DataType::F32}, {"CF32_F32_MODU", DataType::CF32},
      {"CF32_F32_ARGU", DataType::CF32},
  };
  check(imeta.data_type() == expected_types.at(params.conversion),
        "invalid tensor data type for conversion");

  // 3) Sucess
  const std::unordered_map<std::string, DataType> output_data_types = {
      {"U8_F32", DataType::F32},         {"U8_CF32_REAL", DataType::CF32},
      {"U16_CF32_REAL", DataType::CF32}, {"F32_U8_SCALED", DataType::U8},
      {"F32_U16_SCALED", DataType::U16}, {"CF32_F32_MODU", DataType::F32},
      {"CF32_F32_ARGU", DataType::F32},
  };
  TensorMeta ometa(output_data_types.at(params.conversion),
                   MemoryLocation::DEVICE, imeta.shape());
  return TaskMeta(imeta, ometa, false);
}

std::unique_ptr<Task> ConvertTaskFactory::create(const TensorMeta &imeta,
                                                 const json &jparams,
                                                 cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  // 2) Extract conversion
  const std::unordered_map<std::string, ConvertTask::Conversion>
      string_to_conversion = {
          {"U8_F32", ConvertTask::Conversion::U8_F32},
          {"U8_CF32_REAL", ConvertTask::Conversion::U8_CF32_REAL},
          {"U16_CF32_REAL", ConvertTask::Conversion::U16_CF32_REAL},
          {"F32_U8_SCALED", ConvertTask::Conversion::F32_U8_SCALED},
          {"F32_U16_SCALED", ConvertTask::Conversion::F32_U16_SCALED},
          {"CF32_F32_MODU", ConvertTask::Conversion::CF32_F32_MODU},
          {"CF32_F32_ARGU", ConvertTask::Conversion::CF32_F32_ARGU}};
  auto conversion = string_to_conversion.at(params.conversion);

  size_t min_temp_storage_bytes = 0;
  curaii::cuda::unique_device_ptr<uint8_t> d_min_temp_storage;
  curaii::cuda::unique_device_ptr<uint8_t> d_min;

  size_t max_temp_storage_bytes = 0;
  curaii::cuda::unique_device_ptr<uint8_t> d_max_temp_storage = nullptr;
  curaii::cuda::unique_device_ptr<uint8_t> d_max = nullptr;

  // 3) CUB temp storage
  if (conversion == ConvertTask::Conversion::F32_U8_SCALED ||
      conversion == ConvertTask::Conversion::F32_U16_SCALED) {
    CUDA_CHECK(cub::DeviceReduce::Min(
        d_min_temp_storage.get(), min_temp_storage_bytes,
        static_cast<float *>(nullptr), static_cast<float *>(nullptr),
        imeta.size(), stream));

    CUDA_CHECK(cub::DeviceReduce::Max(
        d_max_temp_storage.get(), max_temp_storage_bytes,
        static_cast<float *>(nullptr), static_cast<float *>(nullptr),
        imeta.size(), stream));

    d_min_temp_storage = curaii::cuda::make_unique_device_ptr<uint8_t>(
        min_temp_storage_bytes, stream);
    d_max_temp_storage = curaii::cuda::make_unique_device_ptr<uint8_t>(
        max_temp_storage_bytes, stream);
    d_min =
        curaii::cuda::make_unique_device_ptr<uint8_t>(sizeof(float), stream);
    d_max =
        curaii::cuda::make_unique_device_ptr<uint8_t>(sizeof(float), stream);
  }

  // 6) Assemble task
  auto *task = new ConvertTask(meta, stream, conversion, min_temp_storage_bytes,
                               std::move(d_min_temp_storage), std::move(d_min),
                               max_temp_storage_bytes,
                               std::move(d_max_temp_storage), std::move(d_max));

  return std::unique_ptr<ConvertTask>(task);
}

} // namespace dh