/**
 * @file include/curaii/v2/cuda.hh
 * @author Jules Guillou
 * @brief Convenience RAII helpers and error‑checking macros for the CUDA
 *        runtime API.
 *
 * This header provides:
 *   - Two macros (CUDA_CHECK and CUDA_CHECK_NT) that validate CUDA
 *     runtime calls, log through a global spdlog logger and either
 *     throw an exception or abort the process on failure.
 *   - A strongly‑typed exception class (curaii::cuda::Error).
 *   - Smart‑pointer aliases and factory helpers for host and device memory
 *     with safe deleters.
 *   - RAII wrapper for cudaStream_t.
 *
 * All primitives live in the curaii::cuda namespace.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

/**
 * @def CUDA_CHECK
 * @brief Evaluate a CUDA expression and throw on error.
 *
 * The macro executes @p expr, checks the returned ::cudaError_t and, if it is
 * not cudaSuccess, logs a message at @c spdlog::level::warn and throws
 * a @ref curaii::cuda::Error containing diagnostic information.
 */
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      ::curaii::cuda::detail::log_cuda_failure(spdlog::level::warn, err__,     \
                                               #expr, __FILE__, __LINE__);     \
      throw ::curaii::cuda::Error(err__, #expr, __FILE__, __LINE__);           \
    }                                                                          \
  } while (false)

/**
 * @def CUDA_CHECK_NT
 * @brief Non‑throwing variant suitable for noexcept contexts.
 *
 * Executes @p expr and logs a message at @c spdlog::level::err if the result is
 * not cudaSuccess. After logging, the macro calls
 * std::abort() to terminate the process.
 */
#define CUDA_CHECK_NT(expr)                                                    \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      ::curaii::cuda::detail::log_cuda_failure(spdlog::level::err, err__,      \
                                               #expr, __FILE__, __LINE__);     \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

namespace curaii::cuda {

/**
 * @class Error
 * @brief Exception type representing a CUDA runtime failure.
 *
 * The class stores the failing ::cudaError_t and exposes it via @ref code().
 * The constructor builds a rich error message that includes the numeric code,
 * the textual description returned by ::cudaGetErrorString, the expression
 * that failed, and the source location.
 */
class Error : public std::runtime_error {
public:
  /**
   * @brief Construct an error object.
   * @param code   CUDA error code returned by the runtime.
   * @param what   C‑string containing the textual form of the expression.
   * @param file   Source file where the error originated.
   * @param line   Line number in the source file.
   */
  explicit Error(cudaError_t code, const char *what, const char *file,
                 int line);

  /**
   * @brief Retrieve the underlying CUDA error code.
   */
  [[nodiscard]] cudaError_t code() const noexcept;

private:
  /// Build the what() message.
  static std::string make_message(cudaError_t code, const char *what,
                                  const char *file, int line);

  /// Store the cuda error code.
  cudaError_t code_;
};

/**
 * @brief Deleter for host‑pinned memory allocated with ::cudaMallocHost.
 *
 * Safe for use with smart pointers. The deleter never throws; on
 * failure it logs the runtime error and aborts the process via
 * @ref CUDA_CHECK_NT.
 */
struct HostDeleter {
  void operator()(void *ptr) const noexcept;
};

/**
 * @brief Deleter for device memory allocated with ::cudaMallocAsync or
 *        ::cudaMalloc.
 *
 * Safe for use with smart pointers. The deleter never throws; on
 * failure it logs the runtime error and aborts the process via
 * @ref CUDA_CHECK_NT.
 *
 * @note If a non‑null stream is supplied, memory is released asynchronously via
 *       ::cudaFreeAsync; otherwise ::cudaFree is used.
 */
struct DeviceDeleter {
  explicit DeviceDeleter(cudaStream_t s = 0) noexcept;
  void operator()(void *ptr) const noexcept;
  cudaStream_t stream;
};

/// Unique pointer alias owning pinned host memory.
template <typename T> using unique_host_ptr = std::unique_ptr<T, HostDeleter>;

/// Unique pointer alias owning device memory.
template <typename T>
using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;

/**
 * @brief Allocate pinned host memory.
 *
 * @tparam T     Element type.
 * @param count  Number of elements to allocate.
 * @return unique_host_ptr managing the new memory.
 *
 * @throw curaii::cuda::Error if ::cudaMallocHost fails.
 */
template <typename T>
[[nodiscard]] unique_host_ptr<T> make_unique_host_ptr(std::size_t count);

/**
 * @brief Allocate device memory in the given stream’s async pool.
 *
 * @tparam T     Element type.
 * @param count  Number of elements to allocate.
 * @param stream CUDA stream controlling the allocation’s memory pool
 *               (defaults to 0 = legacy stream).
 * @return unique_device_ptr managing the new memory.
 *
 * @throw curaii::cuda::Error if ::cudaMallocAsync fails.
 */
template <typename T>
[[nodiscard]] unique_device_ptr<T>
make_unique_device_ptr(std::size_t count, cudaStream_t stream = 0);

/**
 * @class Stream
 * @brief RAII wrapper for CUDA streams.
 *
 * Calls cudaStreamCreateWithPriority on construction and
 * cudaStreamDestroy on destruction. Movable but not copyable.
 */
class Stream {
public:
  /**
   * @brief Create a new CUDA stream.
   * @param flags    Creation flags (defaults to cudaStreamDefault).
   * @param priority Priority within [0..maxPriority] (defaults to 0).
   * @throws curaii::cuda::Error if creation fails.
   */
  explicit Stream(unsigned flags = cudaStreamDefault, int priority = 0);

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  /**
   * @brief Move‑construct, taking ownership from @p other.
   */
  Stream(Stream &&other) noexcept;

  /**
   * @brief Move‑assign, destroying any existing stream and taking ownership.
   */
  Stream &operator=(Stream &&other) noexcept;

  /**
   * @brief Destroy the CUDA stream if valid.
   */
  ~Stream() noexcept;

  /**
   * @brief Get the raw cudaStream_t.
   */
  cudaStream_t get() const noexcept;

  /**
   * @brief Release ownership of the stream without destroying it.
   * @return The raw stream handle; this object becomes empty.
   */
  cudaStream_t release() noexcept;

  /**
   * @brief Replace the managed stream, destroying the old one if valid.
   * @param s New raw stream (or nullptr to clear).
   */
  void reset(cudaStream_t s = nullptr) noexcept;

  /**
   * @brief Check whether there is a valid stream.
   * @return true if get() != nullptr.
   */
  explicit operator bool() const noexcept;

private:
  cudaStream_t stream_{nullptr};
};

} // namespace curaii::cuda

namespace curaii::cuda::detail {

/**
 * @brief Log a CUDA failure with the global logger.
 *
 * @param lvl   Desired @c spdlog log level (e.g. warn, err).
 * @param code  CUDA error code.
 * @param expr  Stringified failing expression.
 * @param file  Source file where the error occurred.
 * @param line  Line number.
 */
void log_cuda_failure(spdlog::level::level_enum lvl, cudaError_t code,
                      const char *expr, const char *file, int line);

} // namespace curaii::cuda::detail

namespace curaii::cuda {

template <typename T>
[[nodiscard]] inline unique_host_ptr<T>
make_unique_host_ptr(std::size_t count) {
  T *p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, count * sizeof(T)));
  return unique_host_ptr<T>(p);
}

template <typename T>
[[nodiscard]] inline unique_device_ptr<T>
make_unique_device_ptr(std::size_t count, cudaStream_t stream) {
  T *p = nullptr;
  CUDA_CHECK(cudaMallocAsync(&p, count * sizeof(T), stream));
  return unique_device_ptr<T>(p, DeviceDeleter(stream));
}

} // namespace curaii::cuda