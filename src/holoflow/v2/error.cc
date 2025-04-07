#include "holoflow/v2/error.hh"

#include "holoflow/holoflow.hh"

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    dh::holoflow_logger()->critical("Unreachable code reached at {}:{} - {}",  \
                                    __FILE__, __LINE__, msg);                  \
    std::abort();                                                              \
  } while (0)

// ==========================================================================
//                     ErrorType Implementation
// ==========================================================================

auto fmt::formatter<dh::v2::ErrorType>::format(dh::v2::ErrorType type,
                                               format_context &ctx) const
    -> format_context::iterator {
  std::string_view name;
  switch (type) {
  case dh::v2::ErrorType::NotFound:
    name = "NotFound";
    break;
  case dh::v2::ErrorType::InvalidArgument:
    name = "InvalidArgument";
    break;
  case dh::v2::ErrorType::ValidationError:
    name = "ValidationError";
    break;
  case dh::v2::ErrorType::InternalError:
    name = "InternalError";
    break;
  case dh::v2::ErrorType::ConnectionError:
    name = "ConnectionError";
    break;
  case dh::v2::ErrorType::RemovalError:
    name = "RemovalError";
    break;
  case dh::v2::ErrorType::TransactionError:
    name = "TransactionError";
    break;
  default:
    UNREACHABLE("Invalid error type");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     Error Implementation
// ==========================================================================

namespace dh::v2 {

Error::Error(ErrorType type, std::string message)
    : type_(type), message_(std::move(message)) {}

Error::Error(ErrorType type, std::string message, std::vector<Error> sub_errors)
    : type_(type), message_(std::move(message)),
      sub_errors_(std::move(sub_errors)) {}

Error Error::aggregate(ErrorType type, const std::string &message,
                       const std::vector<Error> &errors) {
  return Error(type, message, errors);
}

const std::string &Error::message() const { return message_; }

ErrorType Error::type() const { return type_; }

bool Error::is_aggregate() const { return !sub_errors_.empty(); }

const std::vector<Error> &Error::sub_errors() const { return sub_errors_; }

std::string Error::to_string(std::size_t indent) const {
  std::string result(indent, ' ');
  result += "- " + message_;

  for (const auto &err : sub_errors_) {
    result += "\n" + err.to_string(indent + 2);
  }
  return result;
}

} // namespace dh::v2

auto fmt::formatter<dh::v2::Error>::format(dh::v2::Error error,
                                           format_context &ctx) const
    -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}:\n{}", error.type(), error.to_string(2));
}
