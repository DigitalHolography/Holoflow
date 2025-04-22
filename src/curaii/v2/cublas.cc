#include "curaii/v2/cublas.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/logger.hh"

namespace {

const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "The operation completed successfully";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "The cuBLAS library was not initialized. This is usually caused by "
           "the lack of a prior cublasCreate() call, an error in the CUDA "
           "Runtime API called by the cuBLAS routine, or an error in the "
           "hardware setup.";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "Resource allocation failed inside the cuBLAS library. This is "
           "usually caused by a cudaMalloc() failure.";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "An unsupported value or parameter was passed to the function (a "
           "negative vector size, for example).";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "The function requires a feature absent from the device "
           "architecture; usually caused by compute capability lower than 5.0.";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "An access to GPU memory space failed, which is usually caused by a "
           "failure to bind a texture.";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "The GPU program failed to execute. This is often caused by a "
           "launch failure of the kernel on the GPU, which can be caused by "
           "multiple reasons.";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "An internal cuBLAS operation failed. This error is usually caused "
           "by a cudaMemcpyAsync() failure.";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "The functionality requested is not supported.";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "The functionality requested requires some license and an error was "
           "detected when trying to check the current licensing. This error "
           "can happen if the license is not present or is expired or if the "
           "environment variable NVIDIA_LICENSE_FILE is not set properly.";
  default:
    DH_BUG("Invalid cuBLAS status");
  }
}

} // namespace

namespace curaii::cublas {

Error::Error(cublasStatus_t code, const char *what, const char *file, int line)
    : std::runtime_error(make_message(code, what, file, line)), code_(code) {}

cublasStatus_t Error::code() const noexcept { return code_; }

std::string Error::make_message(cublasStatus_t code, const char *what,
                                const char *file, int line) {
  std::ostringstream os;
  os << "CUBLAS error: " << cublasGetErrorString(code) << " ("
     << static_cast<int>(code) << ")\n"
     << "  expression : " << what << "\n"
     << "  location   : " << file << ":" << line;
  return os.str();
}

Handle::Handle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

Handle::Handle(cublasHandle_t raw) noexcept : handle_(raw) {}

Handle::Handle(Handle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Handle &Handle::operator=(Handle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

Handle::~Handle() noexcept {
  if (handle_) {
    CUBLAS_CHECK_NT(cublasDestroy(handle_));
  }
}

cublasHandle_t Handle::release() noexcept {
  auto tmp = handle_;
  handle_ = nullptr;
  return tmp;
}

void Handle::reset(cublasHandle_t raw) noexcept {
  if (handle_) {
    CUBLAS_CHECK_NT(cublasDestroy(handle_));
  }
  handle_ = raw;
}

cublasHandle_t Handle::get() const noexcept { return handle_; }

} // namespace curaii::cublas

namespace curaii::cublas::detail {

const char *errorString(cublasStatus_t status) noexcept {
  return cublasGetErrorString(status);
}

void log_cublas_failure(spdlog::level::level_enum lvl, cublasStatus_t code,
                        const char *expr, const char *file, int line) {
  logger()->log(lvl, "cuBLAS error {} ({}): \"{}\"  [{}:{}]",
                cublasGetErrorString(code), static_cast<int>(code), expr, file,
                line);
}

} // namespace curaii::cublas::detail
