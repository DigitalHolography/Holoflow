#include "curaii/cusolver_dn.hh"

#include <cusolverDn.h>
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
//                     CusolverStatus Implementation
// ==========================================================================

dh::CusolverStatus::CusolverStatus(cusolverStatus_t status) noexcept
    : status_(status) {}

const char *dh::CusolverStatus::message() const noexcept {
  switch (status_) {
  case CUSOLVER_STATUS_SUCCESS:
    return "The operation completed successfully.";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "The cuSolver library was not initialized. This is usually caused "
           "by the lack of a prior call, an error in the CUDA Runtime API "
           "called by the cuSolver routine, or an error in the hardware setup.";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "Resource allocation failed inside the cuSolver library. This is "
           "usually caused by a cudaMalloc() failure.";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "An unsupported value or parameter was passed to the function (a "
           "negative vector size, for example).";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "The function requires a feature absent from the device "
           "architecture; usually caused by the lack of support for atomic "
           "operations or double precision.";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "The GPU program failed to execute. This is often caused by a "
           "launch failure of the kernel on the GPU, which can be caused by "
           "multiple reasons.";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "An internal cuSolver operation failed. This error is usually "
           "caused by a cudaMemcpyAsync() failure.";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "The matrix type is not supported by this function. This is usually "
           "caused by passing an invalid matrix descriptor to the function.";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "The parameter combination is not supported, for example batched "
           "version is not supported or M < N is not supported.";
  default:
    UNREACHABLE("Invalid cusolver status");
  }
}

cusolverStatus_t dh::CusolverStatus::status() const noexcept { return status_; }

auto fmt::formatter<dh::CusolverStatus>::format(dh::CusolverStatus status,
                                                format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (status.status()) {
  case CUSOLVER_STATUS_SUCCESS:
    name = "CUSOLVER_STATUS_SUCCESS";
    break;
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    name = "CUSOLVER_STATUS_NOT_INITIALIZED";
    break;
  case CUSOLVER_STATUS_ALLOC_FAILED:
    name = "CUSOLVER_STATUS_ALLOC_FAILED";
    break;
  case CUSOLVER_STATUS_INVALID_VALUE:
    name = "CUSOLVER_STATUS_INVALID_VALUE";
    break;
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    name = "CUSOLVER_STATUS_ARCH_MISMATCH";
    break;
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    name = "CUSOLVER_STATUS_EXECUTION_FAILED";
    break;
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    name = "CUSOLVER_STATUS_INTERNAL_ERROR";
    break;
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    name = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    break;
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    name = "CUSOLVER_STATUS_NOT_SUPPORTED";
    break;
  default:
    UNREACHABLE("Invalid cusolver status");
  }
  return fmt::format_to(ctx.out(), "{}: {}", name, status.message());
}

// ==========================================================================
//                     CusolverDnParamsRef Implementation
// ==========================================================================

dh::CusolverDnParamsRef::CusolverDnParamsRef(cusolverDnParams_t params) noexcept
    : params_(params) {}

dh::CusolverDnParamsRef
dh::CusolverDnParamsRef::from_raw(cusolverDnParams_t params) noexcept {
  return CusolverDnParamsRef(params);
}

cusolverDnParams_t dh::CusolverDnParamsRef::params() const noexcept {
  return params_;
}

// ==========================================================================
//                     CusolverDnParams Implementation
// ==========================================================================

dh::CusolverDnParams::CusolverDnParams(cusolverDnParams_t params) noexcept
    : params_(params) {}

dh::CusolverDnParams::CusolverDnParams(CusolverDnParams &&other) noexcept
    : params_(other.params_) {
  other.params_ = 0;
}

dh::CusolverDnParams &
dh::CusolverDnParams::operator=(CusolverDnParams &&other) noexcept {
  if (this != &other) {
    if (params_) {
      if (auto status = cusolverDnDestroyParams(params_);
          status != CUSOLVER_STATUS_SUCCESS) {
        curaii_logger()->warn(
            "[CusolverDnParams::operator=] failed with error: \"{}\"",
            CusolverStatus(status));
      }
    }
    params_ = other.params_;
    other.params_ = 0;
  }
  return *this;
}

dh::CusolverDnParams::~CusolverDnParams() {
  if (params_) {
    if (auto status = cusolverDnDestroyParams(params_);
        status != CUSOLVER_STATUS_SUCCESS) {
      curaii_logger()->warn(
          "[CusolverDnParams::operator=] failed with error: \"{}\"",
          CusolverStatus(status));
    }
  }
}

tl::expected<dh::CusolverDnParams, dh::CusolverStatus>
dh::CusolverDnParams::try_create() noexcept {
  cusolverDnParams_t params;
  if (auto status = cusolverDnCreateParams(&params);
      status != CUSOLVER_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CusolverDnParams::try_create] failed with error: \"{}\"",
        CusolverStatus(status));
  }

  return CusolverDnParams(params);
}

dh::CusolverDnParamsRef dh::CusolverDnParams::ref() noexcept {
  return CusolverDnParamsRef(params_);
}

cusolverDnParams_t dh::CusolverDnParams::params() const noexcept {
  return params_;
}

// ==========================================================================
//                     CusolverEigMode Implementation
// ==========================================================================

dh::CusolverEigMode::CusolverEigMode(cusolverEigMode_t eig_mode) noexcept
    : eig_mode_(eig_mode) {}

cusolverEigMode_t dh::CusolverEigMode::eig_mode() const noexcept {
  return eig_mode_;
}

auto fmt::formatter<dh::CusolverEigMode>::format(dh::CusolverEigMode eig_mode,
                                                 format_context &ctx) const
    -> format_context::iterator {
  std::string_view name;
  switch (eig_mode.eig_mode()) {
  case CUSOLVER_EIG_MODE_NOVECTOR:
    name = "CUSOLVER_EIG_MODE_NOVECTOR";
    break;
  case CUSOLVER_EIG_MODE_VECTOR:
    name = "cudaStreamNonBlocking";
    break;
  default:
    UNREACHABLE("Invalid cusolver eig mode");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     CusolverDnHandle Implementation
// ==========================================================================

dh::CusolverDnHandle::CusolverDnHandle(cusolverDnHandle_t handle) noexcept
    : handle_(handle) {}

dh::CusolverDnHandle::CusolverDnHandle(CusolverDnHandle &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

dh::CusolverDnHandle &
dh::CusolverDnHandle::operator=(CusolverDnHandle &&other) noexcept {
  if (this != &other) {
    if (handle_) {
      if (auto result = cusolverDnDestroy(handle_);
          result != CUSOLVER_STATUS_SUCCESS) {
        curaii_logger()->warn(
            "[CusolverDnHandle::operator=] failed with error: \"{}\"",
            CusolverStatus(result));
      }
    }
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

dh::CusolverDnHandle::~CusolverDnHandle() {
  if (handle_) {
    if (auto result = cusolverDnDestroy(handle_);
        result != CUSOLVER_STATUS_SUCCESS) {
      curaii_logger()->warn(
          "[CusolverDnHandle::~CusolverDnHandle] failed with error: \"{}\"",
          CusolverStatus(result));
    }
  }
}

tl::expected<dh::CusolverDnHandle, dh::CusolverStatus>
dh::CusolverDnHandle::try_create() noexcept {
  cusolverDnHandle_t handle;
  if (auto result = cusolverDnCreate(&handle);
      result != CUSOLVER_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CusolverDnHandle::try_create] failed with error: \"{}\"",
        CusolverStatus(result));
    return tl::unexpected<CusolverStatus>(result);
  }
  return CusolverDnHandle(handle);
}

tl::expected<void, dh::CusolverStatus>
dh::CusolverDnHandle::try_set_stream(CudaStreamRef stream) noexcept {
  if (auto result = cusolverDnSetStream(handle_, stream.stream());
      result != CUSOLVER_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CusolverDnHandle::try_set_stream] failed with error: \"{}\"",
        CusolverStatus(result));
    return tl::unexpected<CusolverStatus>(result);
  }
  return {};
}

tl::expected<void, dh::CusolverStatus>
dh::CusolverDnHandle::try_x_syevd_buffer_size(
    CusolverDnParamsRef params, CusolverEigMode jobz, CublasFillMode uplo,
    int64_t n, CudaDataType dataTypeA, const void *A, int64_t lda,
    CudaDataType dataTypeW, const void *W, CudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) noexcept {
  if (auto result = cusolverDnXsyevd_bufferSize(
          handle_, params.params(), jobz.eig_mode(), uplo.fill_mode(), n,
          dataTypeA.data_type(), A, lda, dataTypeW.data_type(), W,
          computeType.data_type(), workspaceInBytesOnDevice,
          workspaceInBytesOnHost);
      result != CUSOLVER_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CusolverDnHandle::try_x_syevd_buffer_size] failed with error: \"{}\"",
        CusolverStatus(result));
    return tl::unexpected<CusolverStatus>(result);
  }
  return {};
}

tl::expected<void, dh::CusolverStatus> dh::CusolverDnHandle::try_x_syevd(
    CusolverDnParamsRef params, CusolverEigMode jobz, CublasFillMode uplo,
    int64_t n, CudaDataType dataTypeA, void *A, int64_t lda,
    CudaDataType dataTypeW, void *W, CudaDataType computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info) noexcept {
  if (auto result = cusolverDnXsyevd(
          handle_, params.params(), jobz.eig_mode(), uplo.fill_mode(), n,
          dataTypeA.data_type(), A, lda, dataTypeW.data_type(), W,
          computeType.data_type(), bufferOnDevice, workspaceInBytesOnDevice,
          bufferOnHost, workspaceInBytesOnHost, info);
      result != CUSOLVER_STATUS_SUCCESS) {
    curaii_logger()->warn(
        "[CusolverDnHandle::try_x_syevd] failed with error: \"{}\"",
        CusolverStatus(result));
    return tl::unexpected<CusolverStatus>(result);
  }
  return {};
}