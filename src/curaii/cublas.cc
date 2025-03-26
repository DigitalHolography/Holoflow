#include "curaii/cublas.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <tl/expected.hpp>

#include "curaii/logger.hh"

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    dh::curaii_logger()->critical("Unreachable code reached at {}:{} - {}",    \
                                  __FILE__, __LINE__, msg);                    \
    std::abort();                                                              \
  } while (0)

// ==========================================================================
//                     CublasOperation Implementation
// ==========================================================================

dh::CublasOperation::CublasOperation(cublasOperation_t operation) noexcept
    : operation_(operation) {}

cublasOperation_t dh::CublasOperation::operation() const noexcept {
  return operation_;
}

auto fmt::formatter<dh::CublasOperation>::format(dh::CublasOperation operation,
                                                 format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (operation.operation()) {
  case CUBLAS_OP_N:
    name = "CUBLAS_OP_N";
    break;
  case CUBLAS_OP_T:
    name = "CUBLAS_OP_T";
    break;
  case CUBLAS_OP_C:
    name = "CUBLAS_OP_C";
    break;
  default:
    UNREACHABLE("Invalid cublas operation");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     CublasStatus Implementation
// ==========================================================================

dh::CublasStatus::CublasStatus(cublasStatus_t status) noexcept
    : status_(status) {}

const char *dh::CublasStatus::message() const noexcept {
  switch (status_) {
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
    UNREACHABLE("Invalid cublas status");
  }
}

cublasStatus_t dh::CublasStatus::status() const noexcept { return status_; }

auto fmt::formatter<dh::CublasStatus>::format(dh::CublasStatus status,
                                              format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (status.status()) {
  case CUBLAS_STATUS_SUCCESS:
    name = "CUBLAS_STATUS_SUCCESS";
    break;
  case CUBLAS_STATUS_NOT_INITIALIZED:
    name = "CUBLAS_STATUS_NOT_INITIALIZED";
    break;
  case CUBLAS_STATUS_ALLOC_FAILED:
    name = "CUBLAS_STATUS_ALLOC_FAILED";
    break;
  case CUBLAS_STATUS_INVALID_VALUE:
    name = "CUBLAS_STATUS_INVALID_VALUE";
    break;
  case CUBLAS_STATUS_ARCH_MISMATCH:
    name = "CUBLAS_STATUS_ARCH_MISMATCH";
    break;
  case CUBLAS_STATUS_MAPPING_ERROR:
    name = "CUBLAS_STATUS_MAPPING_ERROR";
    break;
  case CUBLAS_STATUS_EXECUTION_FAILED:
    name = "CUBLAS_STATUS_EXECUTION_FAILED";
    break;
  case CUBLAS_STATUS_INTERNAL_ERROR:
    name = "CUBLAS_STATUS_INTERNAL_ERROR";
    break;
  case CUBLAS_STATUS_NOT_SUPPORTED:
    name = "CUBLAS_STATUS_NOT_SUPPORTED";
    break;
  case CUBLAS_STATUS_LICENSE_ERROR:
    name = "CUBLAS_STATUS_LICENSE_ERROR";
    break;
  default:
    UNREACHABLE("Invalid cublas status");
  }
  return fmt::format_to(ctx.out(), "{}: {}", name, status.message());
}

// ==========================================================================
//                     CublasHandle Implementation
// ==========================================================================

dh::CublasHandle::CublasHandle(cublasHandle_t handle) noexcept
    : handle_(handle) {}

dh::CublasHandle::CublasHandle(CublasHandle &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = 0;
}

dh::CublasHandle &dh::CublasHandle::operator=(CublasHandle &&other) noexcept {
  if (this != &other) {
    if (handle_) {
      if (auto result = cublasDestroy(handle_);
          result != CUBLAS_STATUS_SUCCESS) {
        curaii_logger()->warn(
            "[CublasHandle::operator=] failed with error: \"{}\"",
            CublasStatus(result));
      }
    }
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

dh::CublasHandle::~CublasHandle() {
  if (handle_) {
    if (auto result = cublasDestroy(handle_); result != CUBLAS_STATUS_SUCCESS) {
      curaii_logger()->warn(
          "[CublasHandle::~CublasHandle] failed with error: \"{}\"",
          CublasStatus(result));
    }
  }
}

tl::expected<dh::CublasHandle, dh::CublasStatus>
dh::CublasHandle::try_create() noexcept {
  cublasHandle_t handle;
  if (auto result = cublasCreate(&handle); result != CUBLAS_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CublasHandle::try_create] failed with error: \"{}\"",
        CublasStatus(result));

    return tl::unexpected<CublasStatus>(result);
  }

  return CublasHandle(handle);
}

tl::expected<void, dh::CublasStatus>
dh::CublasHandle::try_set_stream(cudaStream_t stream) noexcept {
  if (auto result = cublasSetStream(handle_, stream);
      result != CUBLAS_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CublasHandle::try_set_stream] failed with error: \"{}\"",
        CublasStatus(result));

    return tl::unexpected<CublasStatus>(result);
  }

  return {};
}

tl::expected<void, dh::CublasStatus> dh::CublasHandle::try_c_gemm_3m(
    CublasOperation transa, CublasOperation transb, int m, int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B,
    int ldb, const cuComplex *beta, cuComplex *C, int ldc) noexcept {
  if (auto result =
          cublasCgemm3m(handle_, transa.operation(), transb.operation(), m, n,
                        k, alpha, A, lda, B, ldb, beta, C, ldc);
      result != CUBLAS_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CublasHandle::try_set_stream] failed with error: \"{}\"",
        CublasStatus(result));

    return tl::unexpected<CublasStatus>(result);
  }

  return {};
}

cublasHandle_t dh::CublasHandle::handle() const noexcept { return handle_; }