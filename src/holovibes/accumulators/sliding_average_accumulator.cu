#include "holovibes/accumulators/sliding_average_accumulator.hh"

#include <cassert>
#include <cstdlib>
#include <numeric>
#include <spdlog/spdlog.h>
#include <vector>

#include "curaii/cuda_runtime.hh"
#include "curaii/v2/cuda.hh"
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

std::optional<TensorView> SlidingAverageAccumulator::write_tensor() {
  if (nb_slots_ - writer_size() == 0) {
    return std::nullopt;
  }

  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *data = d_buffer_.get() + write_idx * element_size_;
  return TensorView(data, meta_.imeta());
}

void SlidingAverageAccumulator::commit_write() {
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

  CUDA_CHECK(cudaMemcpyAsync(avg_data, d_running_avg_.get(), element_size_,
                             cudaMemcpyDeviceToDevice, stream_.stream()));

  CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));

  size_t next_avg_idx = avg_idx + 1;
  if (next_avg_idx == nb_slots_)
    next_avg_idx = 0;

  avg_idx_.store(next_avg_idx, std::memory_order_release);

  size_t next_write_idx = write_idx + 1;
  if (next_write_idx == nb_slots_)
    next_write_idx = 0;

  write_idx_.store(next_write_idx, std::memory_order_release);
}

std::optional<TensorView> SlidingAverageAccumulator::read_tensor() {
  if (reader_size() <= window_size_)
    return std::nullopt;

  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  uint8_t *data = d_buffer_.get() + read_idx * element_size_;
  return TensorView(data, meta_.ometa());
}

void SlidingAverageAccumulator::commit_read() {
  size_t read_idx = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + 1;
  if (next_read_idx == nb_slots_)
    next_read_idx = 0;

  read_idx_.store(next_read_idx, std::memory_order_release);
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

AccumulatorMeta
SlidingAverageAccumulatorFactory::type_check(const TensorMeta &imeta,
                                             const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.window_size != 0, "window_size == 0");
  check(params.nb_slots > params.window_size + 1,
        "nb_slots <= window_size + 1");

  // 2) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.shape().at(0) == 1, "tensor dim 0 != 1");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");
  check(imeta.data_type() == DataType::F32, "tensor data_type != F32");

  // 3) Success
  return AccumulatorMeta(imeta, imeta);
}

std::unique_ptr<Accumulator> SlidingAverageAccumulatorFactory ::create(
    const TensorMeta &imeta, const json &jparams, CudaStreamRef stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  // 2) Buffer size
  auto element_size = meta.imeta().size_in_bytes();
  auto buffer_size = params.nb_slots * element_size;

  // 3) Allocations
  auto d_buffer = make_unique_device_ptr<uint8_t>(
      params.nb_slots * element_size, stream.stream());

  auto d_running_avg =
      make_unique_device_ptr<uint8_t>(element_size, stream.stream());

  // 4) Assemble accumulator
  auto *accumulator = new SlidingAverageAccumulator(
      meta, stream, params.nb_slots, params.window_size, std::move(d_buffer),
      std::move(d_running_avg));

  return std::unique_ptr<SlidingAverageAccumulator>(accumulator);
}

} // namespace dh