#include "curaii/v2/cufft.hh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/logger.hh"

namespace {

const char *cufftGetErrorString(cufftResult result) {

  switch (result) {
  case CUFFT_SUCCESS:
    return "The cuFFT operation was successful";
  case CUFFT_INVALID_PLAN:
    return "cuFFT was passed an invalid plan handle";
  case CUFFT_ALLOC_FAILED:
    return "cuFFT failed to allocate GPU or CPU memory";
  case CUFFT_INVALID_TYPE:
    return "No longer used";
  case CUFFT_INVALID_VALUE:
    return "User specified an invalid pointer or parameter";
  case CUFFT_INTERNAL_ERROR:
    return "Driver or internal cuFFT library error";
  case CUFFT_EXEC_FAILED:
    return "Failed to execute an FFT on the GPU";
  case CUFFT_SETUP_FAILED:
    return "The cuFFT library failed to initialize";
  case CUFFT_INVALID_SIZE:
    return "User specified an invalid transform size";
  case CUFFT_UNALIGNED_DATA:
    return "No longer used";
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "Missing parameters in call";
  case CUFFT_INVALID_DEVICE:
    return "Execution of a plan was on different GPU than plan creation";
  case CUFFT_PARSE_ERROR:
    return "Internal plan database error";
  case CUFFT_NO_WORKSPACE:
    return "No workspace has been provided prior to plan execution";
  case CUFFT_NOT_IMPLEMENTED:
    return "Function does not implement functionality for parameters given";
  case CUFFT_LICENSE_ERROR:
    return "Used in previous versions";
  case CUFFT_NOT_SUPPORTED:
    return "Operation is not supported for parameters given";
  default:
    DH_BUG("Invalid cuFFT result");
  }
}

} // namespace

namespace curaii::cufft {

Error::Error(cufftResult code, const char *what, const char *file, int line)
    : std::runtime_error(Error::make_message(code, what, file, line)),
      code_(code) {}

std::string Error::make_message(cufftResult code, const char *what,
                                const char *file, int line) {
  std::ostringstream os;
  os << "CUFFT error: " << cufftGetErrorString(code) << " ("
     << static_cast<int>(code) << ")\n"
     << "  expression : " << what << '\n'
     << "  location   : " << file << ':' << line;
  return os.str();
}

cufftResult Error::code() const noexcept { return code_; }

Handle::Handle() { CUFFT_CHECK(cufftCreate(&handle_)); }

Handle::Handle(cufftHandle raw) noexcept : handle_(raw) {}

Handle::Handle(Handle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = 0;
}

Handle &Handle::operator=(Handle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

Handle::~Handle() noexcept {
  if (handle_) {
    CUFFT_CHECK_NT(cufftDestroy(handle_));
  }
}

cufftHandle Handle::get() const noexcept { return handle_; }

cufftHandle Handle::release() noexcept {
  cufftHandle tmp = handle_;
  handle_ = 0;
  return tmp;
}

void Handle::reset(cufftHandle raw) noexcept {
  if (handle_) {
    CUFFT_CHECK_NT(cufftDestroy(handle_));
  }
  handle_ = raw;
}

Handle::operator bool() const noexcept { return handle_ != 0; }

} // namespace curaii::cufft

namespace curaii::cufft::detail {

void log_cufft_failure(spdlog::level::level_enum lvl, cufftResult code,
                       const char *expr, const char *file, int line) {
  logger()->log(lvl, "CUFFT error {} ({}): \"{}\"  [{}:{}]",
                cufftGetErrorString(code), // e.g. "invalid argument"
                static_cast<int>(code),    // numeric code
                expr,                      // the failed expression
                file, line);               // location
}

} // namespace curaii::cufft::detail