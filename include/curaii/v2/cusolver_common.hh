/**
 * @file include/curaii/v2/cusolver_common.hh
 * @author Jules Guillou
 * @brief Convenience RAII helpers and error-checking macros for the
 *        CUSOLVER library.
 *
 * This header provides:
 *   - Two macros (CUSOLVER_CHECK and CUSOLVER_CHECK_NT) that validate CUSOLVER
 *     library calls, log through a global spdlog logger and either throw an
 *     exception or abort the process on failure.
 *   - A strongly-typed exception class (curaii::cusolver::Error).
 *   - RAII wrapper for cusolver types.
 *
 * All primitives live in the curaii::cusolver namespace.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cusolver_common.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

/**
 * @def CUSOLVER_CHECK
 * @brief Evaluate a cuSOLVER expression and throw on error.
 *
 * The macro executes @p expr, checks the returned ::cusolverStatus_t and, if it
 * is not CUSOLVER_STATUS_SUCCESS, logs a message at spdlog::level::warn and
 * throws a @ref curaii::cusolver::Error containing diagnostic information.
 */
#define CUSOLVER_CHECK(expr)                                                   \
  do {                                                                         \
    cusolverStatus_t err__ = (expr);                                           \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                    \
      ::curaii::cusolver::detail::log_cusolver_failure(                        \
          spdlog::level::warn, err__, #expr, __FILE__, __LINE__);              \
      throw ::curaii::cusolver::Error(err__, #expr, __FILE__, __LINE__);       \
    }                                                                          \
  } while (false)

/**
 * @def CUSOLVER_CHECK_NT
 * @brief Non‑throwing variant suitable for noexcept contexts.
 *
 * Executes @p expr and logs a message at spdlog::level::err if the result is
 * not CUSOLVER_STATUS_SUCCESS. After logging, the macro calls std::abort().
 */
#define CUSOLVER_CHECK_NT(expr)                                                \
  do {                                                                         \
    cusolverStatus_t err__ = (expr);                                           \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                    \
      ::curaii::cusolver::detail::log_cusolver_failure(                        \
          spdlog::level::err, err__, #expr, __FILE__, __LINE__);               \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

namespace curaii::cusolver {

/**
 * @class Error
 * @brief Exception type representing a cuSOLVER library failure.
 *
 * Stores the failing ::cusolverStatus_t and exposes it via @ref code().
 * The constructor builds a rich error message that includes the numeric code,
 * a textual description, the expression that failed, and the source location.
 */
class Error : public std::runtime_error {
public:
  /**
   * @brief Construct an error object.
   * @param code   cuSOLVER status code returned by the library.
   * @param what   C‑string containing the textual form of the expression.
   * @param file   Source file where the error originated.
   * @param line   Line number in the source file.
   */
  explicit Error(cusolverStatus_t code, const char *what, const char *file,
                 int line);

  /**
   * @brief Retrieve the underlying cuSOLVER status code.
   */
  [[nodiscard]] cusolverStatus_t code() const noexcept;

private:
  /// Build the what() message.
  static std::string make_message(cusolverStatus_t code, const char *what,
                                  const char *file, int line);

  /// Store the cuSOLVER status code.
  cusolverStatus_t code_;
};

} // namespace curaii::cusolver

namespace curaii::cusolver::detail {

/**
 * @brief Log a cuSOLVER failure with the global logger.
 *
 * @param lvl   spdlog log level (e.g. warn, err).
 * @param code  cuSOLVER status code.
 * @param expr  Stringified failing expression.
 * @param file  Source file.
 * @param line  Line number.
 */
void log_cusolver_failure(spdlog::level::level_enum lvl, cusolverStatus_t code,
                          const char *expr, const char *file, int line);

} // namespace curaii::cusolver::detail