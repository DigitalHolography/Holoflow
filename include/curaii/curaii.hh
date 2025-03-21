#pragma once

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <memory>
#include <tl/expected.hpp>

#define CUDA_CHECK_LOG(severity, call)                                         \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    LOG_IF(severity, status != cudaSuccess)                                    \
        << "CUDA call failed with error: " << cudaGetErrorString(status)       \
        << " at " << __FILE__ << ":" << __LINE__;                              \
  } while (0)

#define CUDA_TRY_EXPECTED(call)                                                \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      return tl::unexpected(status);                                           \
    }                                                                          \
  } while (0)

#define CUDA_UNWRAP_EXPECTED(expr)                                             \
  ([&]() -> auto {                                                             \
    auto _result = std::move(expr);                                            \
    if (!_result) {                                                            \
      LOG(FATAL) << "CUDA error: " << cudaGetErrorString(_result.error())      \
                 << " at " << __FILE__ << ":" << __LINE__;                     \
    }                                                                          \
    return std::move(_result.value());                                         \
  }())

namespace dh {

struct HostDeleter {
  void operator()(void *ptr) const {
    if (ptr) {
      CUDA_CHECK_LOG(ERROR, cudaFreeHost(ptr));
    }
  }
};

struct DeviceDeleter {
  cudaStream_t stream;

  DeviceDeleter(cudaStream_t s = 0) : stream(s) {}

  void operator()(void *ptr) const {
    if (ptr) {
      if (stream) {
        CUDA_CHECK_LOG(ERROR, cudaFreeAsync(ptr, stream));
      } else {
        CUDA_CHECK_LOG(ERROR, cudaFree(ptr));
      }
    }
  }
};

template <typename T> using unique_host_ptr = std::unique_ptr<T, HostDeleter>;

template <typename T>
tl::expected<unique_host_ptr<T>, cudaError_t>
try_make_unique_host_ptr(std::size_t count) {
  T *raw_ptr = nullptr;
  CUDA_TRY_EXPECTED(cudaMallocHost(&raw_ptr, count * sizeof(T)));
  return unique_host_ptr<T>(raw_ptr);
}

template <typename T>
unique_host_ptr<T> make_unique_host_ptr(std::size_t count) {
  auto result = try_make_unique_host_ptr<T>(count);
  return std::move(CUDA_UNWRAP_EXPECTED(result));
}

template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;

template <typename T>
tl::expected<unique_device_ptr<T>, cudaError_t>
try_make_unique_device_ptr(std::size_t count, cudaStream_t stream = 0) {
  T *raw_ptr = nullptr;
  CUDA_TRY_EXPECTED(cudaMallocAsync(&raw_ptr, count * sizeof(T), stream));
  return unique_device_ptr<T>(raw_ptr, DeviceDeleter(stream));
}

template <typename T>
unique_device_ptr<T> make_unique_device_ptr(std::size_t count,
                                            cudaStream_t stream = 0) {
  auto result = try_make_unique_device_ptr<T>(count, stream);
  return std::move(CUDA_UNWRAP_EXPECTED(result));
}

class unique_cuda_stream {
public:
  explicit unique_cuda_stream(cudaStream_t s = 0) noexcept : stream_(s) {}

  unique_cuda_stream(unique_cuda_stream &&other) noexcept
      : stream_(other.stream_) {
    other.stream_ = 0;
  }

  unique_cuda_stream &operator=(unique_cuda_stream &&other) noexcept {
    if (this != &other) {
      reset();
      stream_ = other.stream_;
      other.stream_ = 0;
    }
    return *this;
  }

  unique_cuda_stream(const unique_cuda_stream &) = delete;
  unique_cuda_stream &operator=(const unique_cuda_stream &) = delete;

  ~unique_cuda_stream() {
    if (stream_) {
      CUDA_CHECK_LOG(ERROR, cudaStreamDestroy(stream_));
    }
  }

  cudaStream_t release() noexcept {
    cudaStream_t tmp = stream_;
    stream_ = 0;
    return tmp;
  }

  void reset(cudaStream_t s = 0) noexcept {
    if (!stream_ || stream_ == s) {
      return;
    }

    CUDA_CHECK_LOG(ERROR, cudaStreamDestroy(stream_));
    stream_ = s;
  }

  void swap(unique_cuda_stream &other) noexcept {
    std::swap(stream_, other.stream_);
  }

  cudaStream_t get() const noexcept { return stream_; }

  explicit operator bool() const noexcept { return stream_ != 0; }

private:
  cudaStream_t stream_;
};

inline tl::expected<unique_cuda_stream, cudaError_t>
try_make_unique_cuda_stream(unsigned int flags = cudaStreamDefault) {
  cudaStream_t s = 0;
  CUDA_TRY_EXPECTED(cudaStreamCreateWithFlags(&s, flags));
  return unique_cuda_stream(s);
}

inline unique_cuda_stream
make_unique_cuda_stream(unsigned int flags = cudaStreamDefault) {
  auto result = try_make_unique_cuda_stream(flags);
  return std::move(CUDA_UNWRAP_EXPECTED(result));
}

} // namespace dh