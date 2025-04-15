#pragma once

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tl/expected.hpp>

namespace dh {

inline std::shared_ptr<spdlog::logger> &cuda_logger() {
  static std::shared_ptr<spdlog::logger> logger = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");

    auto log = std::make_shared<spdlog::logger>("cuda", sink);
    log->set_level(spdlog::default_logger()->level());
    log->flush_on(spdlog::level::warn);

    spdlog::register_logger(log);

    return log;
  }();
  return logger;
}

#define CUDA_CHECK_LOG(severity, call)                                         \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      dh::cuda_logger()->log(spdlog::level::severity,                          \
                             "CUDA call failed with error: {} at {}:{}",       \
                             cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                          \
  } while (0)

#define CUDA_TRY_EXPECTED(call)                                                \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      dh::cuda_logger()->trace("CUDA call failed (expected): {}",              \
                               cudaGetErrorString(status));                    \
      return tl::unexpected(status);                                           \
    }                                                                          \
  } while (0)

#define CUDA_UNWRAP_EXPECTED(expr)                                             \
  ([&]() -> auto {                                                             \
    auto _result = std::move(expr);                                            \
    if (!_result) {                                                            \
      dh::cuda_logger()->critical("CUDA error: {} at {}:{}",                   \
                                  cudaGetErrorString(_result.error()),         \
                                  __FILE__, __LINE__);                         \
      std::abort();                                                            \
    }                                                                          \
    return std::move(_result.value());                                         \
  }())

struct HostDeleter {
  void operator()(void *ptr) const {
    if (ptr) {
      CUDA_CHECK_LOG(err, cudaFreeHost(ptr));
    }
  }
};

struct DeviceDeleter {
  cudaStream_t stream;

  DeviceDeleter(cudaStream_t s = 0) : stream(s) {}

  void operator()(void *ptr) const {
    if (ptr) {
      if (stream) {
        cuda_logger()->info("[device deleter] {}", (void *)(stream));
        CUDA_CHECK_LOG(err, cudaFreeAsync(ptr, stream));
      } else {
        CUDA_CHECK_LOG(err, cudaFree(ptr));
      }
    }
  }
};

template <typename T> using unique_host_ptr = std::unique_ptr<T, HostDeleter>;

template <typename T>
tl::expected<unique_host_ptr<T>, cudaError_t>
try_make_unique_host_ptr(std::size_t count) {
  T *raw_ptr = nullptr;
  dh::cuda_logger()->trace("Allocating {} bytes (host)", count * sizeof(T));
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
  dh::cuda_logger()->trace("Allocating {} bytes (device, stream={})",
                           count * sizeof(T), (void *)stream);
  CUDA_TRY_EXPECTED(cudaMallocAsync(&raw_ptr, count * sizeof(T), stream));
  return unique_device_ptr<T>(raw_ptr, DeviceDeleter(stream));
}

template <typename T>
unique_device_ptr<T> make_unique_device_ptr(std::size_t count,
                                            cudaStream_t stream = 0) {
  auto result = try_make_unique_device_ptr<T>(count, stream);
  return std::move(CUDA_UNWRAP_EXPECTED(result));
}

} // namespace dh