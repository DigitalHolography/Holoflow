#include "holovibes/accumulators/sliding_average_accumulator.hh"

#include <cassert>
#include <cstdlib>
#include <numeric>
#include <spdlog/spdlog.h>
#include <vector>

#include "curaii/cuda_runtime.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     SlidingAverageAccumulator Implementation
// ==========================================================================

namespace {

__global__ void f32_add_avg_kernel(const float *idata, float *odata, int nx,
                                   int ny, int avg_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  odata[idx] += idata[idx] / avg_size;
}

__global__ void f32_sub_avg_kernel(const float *idata, float *odata, int nx,
                                   int ny, int avg_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  odata[idx] -= idata[idx] / avg_size;
}

} // namespace

SlidingAverageAccumulator::SlidingAverageAccumulator(
    const AccumulatorMeta &meta, CudaStreamRef stream, size_t nb_slots,
    size_t window_size, unique_device_ptr<uint8_t> d_buffer,
    unique_device_ptr<uint8_t> d_avg_frame)
    : Accumulator(meta, stream), window_size_(window_size), nb_slots_(nb_slots),
      element_size_(meta_.imeta().size_in_bytes()),
      d_buffer_(std::move(d_buffer)), d_running_avg_(std::move(d_avg_frame)),
      avg_idx_(nb_slots_ - window_size_), write_idx_(0),
      read_idx_(nb_slots_ - 1) {}

tl::expected<std::optional<TensorView>, Error>
SlidingAverageAccumulator::write_tensor() {
  if (nb_slots_ - writer_size() == 0) {
    return std::nullopt;
  }

  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *data = d_buffer_.get() + write_idx * element_size_;
  return TensorView(data, meta_.imeta());
}

tl::expected<void, Error> SlidingAverageAccumulator::commit_write() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *write_data = d_buffer_.get() + write_idx * element_size_;

  size_t avg_idx = avg_idx_.load(std::memory_order_relaxed);
  uint8_t *avg_data = d_buffer_.get() + avg_idx * element_size_;

  int ny = meta_.imeta().shape().at(1);
  int nx = meta_.imeta().shape().at(2);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x,
                 (ny + block_size.y - 1) / block_size.y);

  f32_add_avg_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      reinterpret_cast<float *>(write_data),
      reinterpret_cast<float *>(d_running_avg_.get()), nx, ny, window_size_);

  f32_sub_avg_kernel<<<grid_size, block_size, 0, stream_.stream()>>>(
      reinterpret_cast<float *>(avg_data),
      reinterpret_cast<float *>(d_running_avg_.get()), nx, ny, window_size_);

  if (auto error =
          cudaMemcpyAsync(avg_data, d_running_avg_.get(), element_size_,
                          cudaMemcpyDeviceToDevice, stream_.stream());
      error != cudaSuccess) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulator::commit_write] failed with error \"{}\"",
        CudaError(error));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (auto error = cudaStreamSynchronize(stream_.stream());
      error != cudaSuccess) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulator::commit_write] failed with error \"{}\"",
        CudaError(error));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  size_t next_avg_idx = avg_idx + 1;
  if (next_avg_idx == nb_slots_)
    next_avg_idx = 0;

  avg_idx_.store(next_avg_idx, std::memory_order_release);

  size_t next_write_idx = write_idx + 1;
  if (next_write_idx == nb_slots_)
    next_write_idx = 0;

  write_idx_.store(next_write_idx, std::memory_order_release);
  return {};
}

tl::expected<std::optional<TensorView>, Error>
SlidingAverageAccumulator::read_tensor() {
  if (reader_size() <= window_size_)
    return std::nullopt;

  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  uint8_t *data = d_buffer_.get() + read_idx * element_size_;
  return TensorView(data, meta_.ometa());
}

tl::expected<void, Error> SlidingAverageAccumulator::commit_read() {
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + 1;
  if (next_read_idx == nb_slots_)
    next_read_idx = 0;

  read_idx_.store(next_read_idx, std::memory_order_release);
  return {};
}

size_t SlidingAverageAccumulator::writer_size() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t read_idx = read_idx_.load(std::memory_order_acquire);

  size_t diff = write_idx - read_idx;
  if (write_idx <= read_idx)
    diff += nb_slots_;

  return diff;
}

size_t SlidingAverageAccumulator::reader_size() {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);

  size_t diff = write_idx - read_idx;
  if (write_idx <= read_idx)
    diff += nb_slots_;

  return diff;
}

// ==========================================================================
//                     SlidingAverageAccumulator Implementation
// ==========================================================================

tl::expected<AccumulatorMeta, Error>
SlidingAverageAccumulatorFactory::type_check(const TensorMeta &imeta,
                                             const json &jparams) {
  auto params = jparams.get<Params>();

  if (params.window_size == 0) {
    holovibes_logger()->warn("[SlidingAverageAccumulatorFactory::type_check] "
                             "Invalid window_size: \"{}\"",
                             params.window_size);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.nb_slots <= params.window_size + 1) {
    holovibes_logger()->warn("[SlidingAverageAccumulatorFactory::type_check] "
                             "Invalid nb_slots: \"{}\"",
                             params.nb_slots);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulatorFactory::type_check] invalid rank \"{}\"",
        imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().at(0) != 1) {
    holovibes_logger()->warn("[SlidingAverageAccumulatorFactory::type_check] "
                             "invalid batch size \"{}\"",
                             imeta.shape().at(0));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn("[SlidingAverageAccumulatorFactory::type_check] "
                             "invalid memory location \"{}\"",
                             (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  switch (imeta.data_type()) {
  case DataType::F32:
    break;
  default:
    holovibes_logger()->warn("[SlidingAverageAccumulatorFactory::type_check] "
                             "invalid input type \"{}\"",
                             (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return AccumulatorMeta(imeta, imeta);
}

tl::expected<std::unique_ptr<Accumulator>, Error>
SlidingAverageAccumulatorFactory ::create(const TensorMeta &imeta,
                                          const json &jparams,
                                          CudaStreamRef stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulatorFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }

  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  auto element_size = meta.imeta().size_in_bytes();

  auto d_buffer_result = try_make_unique_device_ptr<uint8_t>(
      params.nb_slots * element_size, stream.stream());
  if (!d_buffer_result) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulatorFactory::create] failed with error \"{}\"",
        CudaError(d_buffer_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_buffer = std::move(d_buffer_result.value());

  auto d_running_avg_result =
      try_make_unique_device_ptr<uint8_t>(element_size, stream.stream());
  if (!d_running_avg_result) {
    holovibes_logger()->warn(
        "[SlidingAverageAccumulatorFactory::create] failed with error \"{}\"",
        CudaError(d_running_avg_result.error()));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto d_running_avg = std::move(d_running_avg_result.value());

  auto *accumulator = new SlidingAverageAccumulator(
      meta, stream, params.nb_slots, params.window_size, std::move(d_buffer),
      std::move(d_running_avg));

  return std::unique_ptr<SlidingAverageAccumulator>(accumulator);
}

} // namespace dh