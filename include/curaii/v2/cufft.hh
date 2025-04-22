/**
 * @file include/curaii/v2/cuda.hh
 * @author Jules Guillou
 * @brief Convenience RAII helpers and error-checking macros for the
 *        CUFFT library.
 *
 * This header provides:
 *   - Two macros (CUFFT_CHECK and CUFFT_CHECK_NT) that validate CUFFT
 *     library calls, log through a global spdlog logger and either throw an
 *     exception or abort the process on failure.
 *   - A strongly-typed exception class (curaii::cufft::Error).
 *   - RAII wrapper for cufft types.
 *
 * All primitives live in the curaii::cufft namespace.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cufft.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

/**
 * @def CUFFT_CHECK
 * @brief Evaluate a CUFFT expression and throw on error.
 *
 * The macro executes @p expr, checks the returned ::cufftResult and, if it is
 * not CUFFT_SUCCESS, logs a message at @c spdlog::level::warn and throws
 * a @ref curaii::cufft::Error containing diagnostic information.
 */
#define CUFFT_CHECK(expr)                                                      \
  do {                                                                         \
    cufftResult err__ = (expr);                                                \
    if (err__ != CUFFT_SUCCESS) {                                              \
      ::curaii::cufft::detail::log_cufft_failure(spdlog::level::warn, err__,   \
                                                 #expr, __FILE__, __LINE__);   \
      throw ::curaii::cufft::Error(err__, #expr, __FILE__, __LINE__);          \
    }                                                                          \
  } while (false)

/**
 * @def CUFFT_CHECK_NT
 * @brief Non‑throwing variant suitable for noexcept contexts.
 *
 * Executes @p expr and logs a message at @c spdlog::level::err if the result is
 * not CUFFT_SUCCESS. After logging, the macro calls
 * std::abort() to terminate the process.
 */
#define CUFFT_CHECK_NT(expr)                                                   \
  do {                                                                         \
    cufftResult err__ = (expr);                                                \
    if (err__ != CUFFT_SUCCESS) {                                              \
      ::curaii::cufft::detail::log_cufft_failure(spdlog::level::warn, err__,   \
                                                 #expr, __FILE__, __LINE__);   \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

namespace curaii::cufft {

/**
 * @class Error
 * @brief Exception type representing a CUFFT library failure.
 *
 * The class stores the failing ::cufftResult and exposes it via @ref code().
 * The constructor builds a rich error message that includes the numeric code,
 * the textual description, the expression that failed, and the source location.
 */
class Error : public std::runtime_error {
public:
  /**
   * @brief Construct an error object.
   * @param code   CUFFT error code returned by the library.
   * @param what   C‑string containing the textual form of the expression.
   * @param file   Source file where the error originated.
   * @param line   Line number in the source file.
   */
  explicit Error(cufftResult code, const char *what, const char *file,
                 int line);

  /**
   * @brief Retrieve the underlying CUFFT error code.
   */
  [[nodiscard]] cufftResult code() const noexcept;

private:
  /// Build the what() message.
  static std::string make_message(cufftResult code, const char *what,
                                  const char *file, int line);

  /// Store the cufft error code.
  cufftResult code_;
};

/**
 * @class Handle
 * @brief RAII wrapper for CUFFT library handles.
 *
 * This class manages the lifetime of a ::cufftHandle, ensuring that
 * cufftCreate is called on construction and cufftDestroy on destruction.
 * It is non-copyable but movable, supports implicit conversion to
 * cufftHandle for API interoperability, and provides methods to
 * release or reset the underlying handle.
 */
class Handle {
public:
  /**
   * @brief Create a new CUFFT handle.
   * @throws curaii::cufft::Error if cufftCreate fails.
   */
  Handle();

  /**
   * @brief Take ownership of an existing CUFFT handle.
   * @param raw A valid cufftHandle.
   * @warning Passing an invalid cufftHandle leads to undifined behaviour.
   */
  explicit Handle(cufftHandle raw) noexcept;

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;

  /**
   * @brief Move-construct from another Handle, stealing its handle.
   */
  Handle(Handle &&other) noexcept;

  /**
   * @brief Move-assign from another Handle, cleaning up current handle.
   */
  Handle &operator=(Handle &&other) noexcept;

  /**
   * @brief Destroy the CUFFT handle if valid.
   */
  ~Handle() noexcept;

  /**
   * @brief Get the raw CUFFT handle.
   * @return The underlying cufftHandle.
   */
  cufftHandle get() const noexcept;

  /**
   * @brief Release ownership of the handle without destroying it.
   * @return The raw handle; this object becomes empty.
   */
  cufftHandle release() noexcept;

  /**
   * @brief Replace the managed handle, destroying the old one if valid.
   * @param raw A valid cufftHandle (or 0 to clear).
   */
  void reset(cufftHandle raw = 0) noexcept;

  /**
   * @brief Check whether the handle is valid.
   * @return true if handle_ != 0.
   */
  explicit operator bool() const noexcept;

private:
  cufftHandle handle_;
};

} // namespace curaii::cufft

namespace curaii::cufft::detail {

/**
 * @brief Log a CUFFT failure with the global logger.
 *
 * @param lvl   Desired @c spdlog log level (e.g. warn, err).
 * @param code  CUFFT error code.
 * @param expr  Stringified failing expression.
 * @param file  Source file where the error occurred.
 * @param line  Line number.
 */
void log_cufft_failure(spdlog::level::level_enum lvl, cufftResult code,
                       const char *expr, const char *file, int line);

} // namespace curaii::cufft::detail