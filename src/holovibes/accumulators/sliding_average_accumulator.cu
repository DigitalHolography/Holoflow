#include "holovibes/accumulators/sliding_average_accumulator.hh"

#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>
#include <numeric>
#include <spdlog/spdlog.h>
#include <vector>

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

__global__ void cf32_conjugate_kernel(const cuFloatComplex *idata,
                                      cuFloatComplex *odata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  auto c     = idata[idx];
  c.y        = -c.y;
  odata[idx] = c;
}

__global__ void cf32_hadamard_kernel(const cuFloatComplex *idata1,
                                     const cuFloatComplex *idata2,
                                     cuFloatComplex *odata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = cuCmulf(idata1[idx], idata2[idx]);
}

__global__ void f32_shift_kernel(const float *idata, float *odata, int shift_x,
                                 int shift_y, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int src_x   = x - shift_x;
  int src_y   = y - shift_y;
  int dst_idx = y * w + x;

  if (src_x < 0 || src_x >= w || src_y < 0 || src_y >= h) {
    odata[dst_idx] = 0.0f;
  } else {
    int src_idx    = src_y * w + src_x;
    odata[dst_idx] = idata[src_idx];
  }
}

__global__ void f32_sub_mean_kernel(const float *idata, float *odata, int count,
                                    float *sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  __shared__ float mean;
  if (threadIdx.x == 0) {
    mean = (*sum) / static_cast<float>(count);
  }
  __syncthreads();

  odata[idx] = idata[idx] - mean;
}

} // namespace

SlidingAverageAccumulator::SlidingAverageAccumulator(
    const AccumulatorMeta &meta, cudaStream_t stream,
    Accumulator::EventListeners event_listeners, size_t nb_slots,
    size_t window_size, DevPtr<uint8_t> d_buffer, DevPtr<uint8_t> d_avg_frame,
    size_t freq_size, curaii::cufft::Handle r2c_handle,
    curaii::cufft::Handle c2r_handle, curaii::cublas::Handle handle,
    DevPtr<cuFloatComplex> d_freq1, DevPtr<cuFloatComplex> d_freq2,
    DevPtr<float> d_xcorr, DevPtr<float> d_shifted, DevPtr<float> d_ref,
    DevPtr<uint8_t> d_cub_sum_tmp, DevPtr<float> d_cub_sum,
    size_t cub_sum_tmp_bytes)
    : Accumulator(meta, stream, event_listeners), window_size_(window_size),
      nb_slots_(nb_slots), element_size_(meta_.imeta().size_in_bytes()),
      d_buffer_(std::move(d_buffer)), d_running_avg_(std::move(d_avg_frame)),
      freq_size_(freq_size), r2c_handle_(std::move(r2c_handle)),
      c2r_handle_(std::move(c2r_handle)), handle_(std::move(handle)),
      d_freq1_(std::move(d_freq1)), d_freq2_(std::move(d_freq2)),
      d_xcorr_(std::move(d_xcorr)), d_shifted_(std::move(d_shifted)),
      d_ref_(std::move(d_ref)), d_cub_sum_tmp_(std::move(d_cub_sum_tmp)),
      d_cub_sum_(std::move(d_cub_sum)), cub_sum_tmp_bytes_(cub_sum_tmp_bytes),
      avg_idx_(nb_slots_ - window_size_), write_idx_(0),
      read_idx_(nb_slots_ - 1), d_ref_initialized_(false) {}

std::optional<TensorView> SlidingAverageAccumulator::write_tensor() {
  if (nb_slots_ - writer_size() == 0) {
    return std::nullopt;
  }

  size_t   write_idx = write_idx_.load(std::memory_order_relaxed);
  uint8_t *data      = d_buffer_.get() + write_idx * element_size_;
  return TensorView(data, meta_.imeta());
}

void SlidingAverageAccumulator::commit_write() {
  size_t   write_idx  = write_idx_.load(std::memory_order_relaxed);
  uint8_t *write_data = d_buffer_.get() + write_idx * element_size_;

  size_t   avg_idx  = avg_idx_.load(std::memory_order_relaxed);
  uint8_t *avg_data = d_buffer_.get() + avg_idx * element_size_;

  int ny = meta_.imeta().shape().at(1);
  int nx = meta_.imeta().shape().at(2);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x,
                 (ny + block_size.y - 1) / block_size.y);

  // registration((float *)write_data, (float *)write_data);

  f32_add_avg_kernel<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<float *>(write_data),
      reinterpret_cast<float *>(d_running_avg_.get()), nx, ny, window_size_);

  f32_sub_avg_kernel<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<float *>(avg_data),
      reinterpret_cast<float *>(d_running_avg_.get()), nx, ny, window_size_);

  CUDA_CHECK(cudaMemcpyAsync(avg_data, d_running_avg_.get(), element_size_,
                             cudaMemcpyDeviceToDevice, stream_));

  CUDA_CHECK(cudaStreamSynchronize(stream_));

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

  size_t   read_idx = read_idx_.load(std::memory_order_relaxed);
  uint8_t *data     = d_buffer_.get() + read_idx * element_size_;
  return TensorView(data, meta_.ometa());
}

void SlidingAverageAccumulator::commit_read() {
  size_t read_idx      = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + 1;
  if (next_read_idx == nb_slots_)
    next_read_idx = 0;

  read_idx_.store(next_read_idx, std::memory_order_release);
}

size_t SlidingAverageAccumulator::writer_size() {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t read_idx  = read_idx_.load(std::memory_order_acquire);

  size_t diff = write_idx - read_idx;
  if (write_idx <= read_idx)
    diff += nb_slots_;

  return diff;
}

size_t SlidingAverageAccumulator::reader_size() {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx  = read_idx_.load(std::memory_order_relaxed);

  size_t diff = write_idx - read_idx;
  if (write_idx <= read_idx)
    diff += nb_slots_;

  return diff;
}

void SlidingAverageAccumulator::registration(float *odata, float *idata) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  center_mean(idata);

  if (!d_ref_initialized_) {
    cudaMemcpyAsync(d_ref_.get(), idata, w * h * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    if (odata != idata) {
      cudaMemcpyAsync(odata, idata, w * h * sizeof(float),
                      cudaMemcpyDeviceToDevice, stream_);
    }

    d_ref_initialized_ = true;
    return;
  }

  cross_correlation(d_xcorr_.get(), idata, (float *)d_ref_.get());

  int amax = 0;
  CUBLAS_CHECK(cublasIsamax(handle_.get(), w * h, d_xcorr_.get(), 1, &amax));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  amax -= 1;
  int x       = amax % w;
  int y       = amax / w;
  int shift_x = x <= w / 2 ? -x : w - x;
  int shift_y = y <= h / 2 ? -y : h - y;
  holovibes_logger()->info("shift_x, shift_y: ({}, {})", shift_x, shift_y);

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x,
                 (h + block_size.y - 1) / block_size.y);
  f32_shift_kernel<<<grid_size, block_size, 0, stream_>>>(
      idata, d_shifted_.get(), shift_x, shift_y, w, h);
  CUDA_CHECK(cudaMemcpyAsync(idata, d_shifted_.get(), w * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream_));
}

void SlidingAverageAccumulator::center_mean(float *iodata) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  CUDA_CHECK(cub::DeviceReduce::Sum(d_cub_sum_tmp_.get(), cub_sum_tmp_bytes_,
                                    iodata, d_cub_sum_.get(), w * h, stream_));

  int block_dim = 256;
  int grid_dim  = (w * h + block_dim - 1) / block_dim;
  f32_sub_mean_kernel<<<grid_dim, block_dim>>>(iodata, iodata, w * h,
                                               d_cub_sum_.get());
}

void SlidingAverageAccumulator::cross_correlation(float *odata, float *idata1,
                                                  float *idata2) {
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata1, d_freq1_.get()));
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata2, d_freq2_.get()));

  int block_size = 256;
  int grid_size  = (freq_size_ + block_size - 1) / block_size;
  cf32_conjugate_kernel<<<grid_size, block_size, 0, stream_>>>(
      d_freq2_.get(), d_freq2_.get(), freq_size_);
  cf32_hadamard_kernel<<<grid_size, block_size, 0, stream_>>>(
      d_freq1_.get(), d_freq2_.get(), d_freq1_.get(), freq_size_);

  CUFFT_CHECK(cufftExecC2R(c2r_handle_.get(), d_freq1_.get(), odata));
}

// ==========================================================================
//                     SlidingAverageAccumulator Implementation
// ==========================================================================

AccumulatorMeta
SlidingAverageAccumulatorFactory::type_check(const TensorMeta &imeta,
                                             const json       &jparams) {
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
    const TensorMeta &imeta, const json &jparams, cudaStream_t stream,
    Accumulator::EventListeners event_listeners) {
  // Validate
  auto meta   = type_check(imeta, jparams);
  auto params = jparams.get<Params>();
  auto w      = imeta.shape().at(2);
  auto h      = imeta.shape().at(1);

  // Handles
  size_t r2c_ws     = 0;
  size_t c2r_ws     = 0;
  auto   r2c_handle = curaii::cufft::Handle();
  auto   c2r_handle = curaii::cufft::Handle();
  auto   handle     = curaii::cublas::Handle();
  CUFFT_CHECK(cufftMakePlan2d(r2c_handle.get(), w, h, CUFFT_R2C, &r2c_ws));
  CUFFT_CHECK(cufftMakePlan2d(c2r_handle.get(), w, h, CUFFT_C2R, &c2r_ws));
  CUFFT_CHECK(cufftSetStream(r2c_handle.get(), stream));
  CUFFT_CHECK(cufftSetStream(c2r_handle.get(), stream));
  CUBLAS_CHECK(cublasSetStream(handle.get(), stream));

  // Buffer size
  auto element_size = meta.imeta().size_in_bytes();
  auto buffer_size  = params.nb_slots * element_size;
  auto freq_size    = w * h;

  void  *d_temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  float  sum                = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, (float *)nullptr,
                         &sum, w * h, stream);

  // Allocations
  using namespace curaii::cuda;
  auto d_buffer      = make_unique_device_ptr<uint8_t>(buffer_size);
  auto d_running_avg = make_unique_device_ptr<uint8_t>(element_size);
  auto d_freq1       = make_unique_device_ptr<cuFloatComplex>(freq_size);
  auto d_freq2       = make_unique_device_ptr<cuFloatComplex>(freq_size);
  auto d_xcorr       = make_unique_device_ptr<float>(w * h);
  auto d_shifted     = make_unique_device_ptr<float>(w * h);
  auto d_ref         = make_unique_device_ptr<float>(w * h);
  auto d_cub_sum_tmp = make_unique_device_ptr<uint8_t>(temp_storage_bytes);
  auto d_cub_sum     = make_unique_device_ptr<float>(1);

  // Assemble accumulator
  auto *accumulator = new SlidingAverageAccumulator(
      meta, stream, event_listeners, params.nb_slots, params.window_size,
      std::move(d_buffer), std::move(d_running_avg), freq_size,
      std::move(r2c_handle), std::move(c2r_handle), std::move(handle),
      std::move(d_freq1), std::move(d_freq2), std::move(d_xcorr),
      std::move(d_shifted), std::move(d_ref), std::move(d_cub_sum_tmp),
      std::move(d_cub_sum), temp_storage_bytes);

  return std::unique_ptr<SlidingAverageAccumulator>(accumulator);
}

} // namespace dh