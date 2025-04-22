/**
 * @file include/curaii/v2/cublas.hh
 * @author Jules Guillou
 * @brief Convenience RAII helpers and error‑checking macros for the
 *        cuBLAS library.
 *
 * This header provides:
 *   - Two macros (CUBLAS_CHECK and CUBLAS_CHECK_NT) that validate cuBLAS
 *     library calls, log through a global spdlog logger and either throw an
 *     exception or abort the process on failure.
 *   - A strongly‑typed exception class (curaii::cublas::Error).
 *   - RAII wrapper for cublasHandle_t.
 *
 * All primitives live in the curaii::cublas namespace.
 */

#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

/**
 * @def CUBLAS_CHECK
 * @brief Evaluate a cuBLAS expression and throw on error.
 *
 * The macro executes @p expr, checks the returned ::cublasStatus_t and, if it
 * is not CUBLAS_STATUS_SUCCESS, logs a message at spdlog::level::warn and
 * throws a @ref curaii::cublas::Error containing diagnostic information.
 */
#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t err__ = (expr);                                             \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                      \
      ::curaii::cublas::detail::log_cublas_failure(spdlog::level::warn, err__, \
                                                   #expr, __FILE__, __LINE__); \
      throw ::curaii::cublas::Error(err__, #expr, __FILE__, __LINE__);         \
    }                                                                          \
  } while (false)

/**
 * @def CUBLAS_CHECK_NT
 * @brief Non‑throwing variant suitable for noexcept contexts.
 *
 * Executes @p expr and logs a message at spdlog::level::warn if the result is
 * not CUBLAS_STATUS_SUCCESS. After logging, the macro calls std::abort().
 */
#define CUBLAS_CHECK_NT(expr)                                                  \
  do {                                                                         \
    cublasStatus_t err__ = (expr);                                             \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                      \
      ::curaii::cublas::detail::log_cublas_failure(spdlog::level::warn, err__, \
                                                   #expr, __FILE__, __LINE__); \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

namespace curaii::cublas {

/**
 * @class Error
 * @brief Exception type representing a cuBLAS library failure.
 *
 * Stores the failing ::cublasStatus_t and exposes it via @ref code().
 * The constructor builds a rich error message that includes the numeric code,
 * a textual description, the expression that failed, and the source location.
 */
class Error : public std::runtime_error {
public:
  /**
   * @brief Construct an error object.
   * @param code   cuBLAS status code returned by the library.
   * @param what   C‑string of the expression that failed.
   * @param file   Source file where the error originated.
   * @param line   Line number in the source file.
   */
  explicit Error(cublasStatus_t code, const char *what, const char *file,
                 int line);

  /**
   * @brief Retrieve the underlying cuBLAS status code.
   */
  [[nodiscard]] cublasStatus_t code() const noexcept;

private:
  /// Build the what() message.
  static std::string make_message(cublasStatus_t code, const char *what,
                                  const char *file, int line);

  /// Store the cuBLAS status code.
  cublasStatus_t code_;
};

/**
 * @class Handle
 * @brief RAII wrapper for cuBLAS handles.
 *
 * Manages a ::cublasHandle_t, calling cublasCreate on construction
 * and cublasDestroy on destruction. Movable but not copyable.
 */
class Handle {
public:
  /**
   * @brief Create a new cuBLAS handle.
   * @throws curaii::cublas::Error if cublasCreate fails.
   */
  Handle();

  /**
   * @brief Take ownership of an existing handle.
   * @param raw A valid cublasHandle_t.
   * @warning Passing an invalid handle leads to undefined behaviour.
   */
  explicit Handle(cublasHandle_t raw) noexcept;

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;

  /**
   * @brief Move-construct, stealing the handle.
   */
  Handle(Handle &&other) noexcept;

  /**
   * @brief Move-assign, cleaning up the old handle.
   */
  Handle &operator=(Handle &&other) noexcept;

  /**
   * @brief Destroy the cuBLAS handle if valid.
   */
  ~Handle() noexcept;

  /**
   * @brief Get the raw cuBLAS handle.
   */
  cublasHandle_t get() const noexcept;

  /**
   * @brief Release ownership of the handle without destroying it.
   * @return The raw handle; this object becomes empty.
   */
  cublasHandle_t release() noexcept;

  /**
   * @brief Replace the managed handle, destroying the old one if valid.
   * @param raw A valid handle (or nullptr to clear).
   */
  void reset(cublasHandle_t raw = nullptr) noexcept;

  /**
   * @brief Check whether the handle is valid.
   */
  explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
  cublasHandle_t handle_{nullptr};
};

} // namespace curaii::cublas

namespace curaii::cublas::detail {

/**
 * @brief Log a cuBLAS failure with the global logger.
 *
 * @param lvl   spdlog log level (e.g. warn, err).
 * @param code  cuBLAS status code.
 * @param expr  Stringified failing expression.
 * @param file  Source file.
 * @param line  Line number.
 */
void log_cublas_failure(spdlog::level::level_enum lvl, cublasStatus_t code,
                        const char *expr, const char *file, int line);

} // namespace curaii::cublas::detail
