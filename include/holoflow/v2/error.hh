#pragma once

#include <fmt/base.h>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <vector>

namespace dh::v2 {

/**
 * @brief Enumerates the kinds of errors that can occur in transactions and
 * model operations.
 */
enum class ErrorType {
  NotFound,        ///< Entity was not found.
  InvalidArgument, ///< Invalid argument provided.
  ValidationError, ///< Validation failed.
  InternalError,   ///< Internal logic error.
  ConnectionError, ///< Node connection failure.
  RemovalError,    ///< Node removal error.
  TransactionError ///< General transaction error.
};

/**
 * @brief Represents an error, possibly aggregating multiple sub-errors.
 */
class Error {
public:
  /// Creates a simple typed error with formatted message.
  template <typename... Args>
  static Error make(ErrorType type, fmt::format_string<Args...> fmt_str,
                    Args &&...args) {
    return Error(type, fmt::format(fmt_str, std::forward<Args>(args)...));
  }

  /// Aggregates multiple errors under a common message and error type.
  static Error aggregate(ErrorType type, const std::string &message,
                         const std::vector<Error> &errors);

  /// Retrieves the error message.
  const std::string &message() const;

  /// Retrieves the error type.
  ErrorType type() const;

  /// Checks if this error aggregates multiple sub-errors.
  bool is_aggregate() const;

  /// Retrieves sub-errors if any.
  const std::vector<Error> &sub_errors() const;

  /// Formats the error and its sub-errors into a readable string.
  std::string to_string(std::size_t indent = 0) const;

private:
  Error(ErrorType type, std::string message);
  Error(ErrorType type, std::string message, std::vector<Error> sub_errors);

  ErrorType type_;
  std::string message_;
  std::vector<Error> sub_errors_;
};

} // namespace dh::v2

template <> struct fmt::formatter<dh::v2::ErrorType> : formatter<string_view> {
  auto format(dh::v2::ErrorType type, format_context &ctx) const
      -> format_context::iterator;
};

template <> struct fmt::formatter<dh::v2::Error> : formatter<string_view> {
  auto format(dh::v2::Error error, format_context &ctx) const
      -> format_context::iterator;
};