#include "curaii/cufft.hh"

#include <cuda_runtime.h>
#include <cufftXt.h>
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
//                     CufftType Implementation
// ==========================================================================

dh::CufftType::CufftType(cufftType type) noexcept : type_(type) {}

cufftType dh::CufftType::type() const noexcept { return type_; }

auto fmt::formatter<dh::CufftType>::format(dh::CufftType type,
                                           format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (type.type()) {
  case CUFFT_R2C:
    name = "CUFFT_R2C";
    break;
  case CUFFT_C2R:
    name = "CUFFT_C2R";
    break;
  case CUFFT_C2C:
    name = "CUFFT_C2C";
    break;
  case CUFFT_D2Z:
    name = "CUFFT_D2Z";
    break;
  case CUFFT_Z2D:
    name = "CUFFT_Z2D";
    break;
  case CUFFT_Z2Z:
    name = "CUFFT_Z2Z";
    break;
  default:
    UNREACHABLE("Invalid cufft type");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     CufftResult Implementation
// ==========================================================================

dh::CufftResult::CufftResult(cufftResult res) noexcept : result_(res) {}

const char *dh::CufftResult::message() const noexcept {
  switch (result_) {
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
    UNREACHABLE("Invalid cufft result");
  }
}

cufftResult dh::CufftResult::result() const noexcept { return result_; }

auto fmt::formatter<dh::CufftResult>::format(dh::CufftResult result,
                                             format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (result.result()) {
  case CUFFT_SUCCESS:
    name = "CUFFT_SUCCESS";
    break;
  case CUFFT_INVALID_PLAN:
    name = "CUFFT_INVALID_PLAN";
    break;
  case CUFFT_ALLOC_FAILED:
    name = "CUFFT_ALLOC_FAILED";
    break;
  case CUFFT_INVALID_TYPE:
    name = "CUFFT_INVALID_TYPE";
    break;
  case CUFFT_INVALID_VALUE:
    name = "CUFFT_INVALID_VALUE";
    break;
  case CUFFT_INTERNAL_ERROR:
    name = "CUFFT_INTERNAL_ERROR";
    break;
  case CUFFT_EXEC_FAILED:
    name = "CUFFT_EXEC_FAILED";
    break;
  case CUFFT_SETUP_FAILED:
    name = "CUFFT_SETUP_FAILED";
    break;
  case CUFFT_INVALID_SIZE:
    name = "CUFFT_INVALID_SIZE";
    break;
  case CUFFT_UNALIGNED_DATA:
    name = "CUFFT_UNALIGNED_DATA";
    break;
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    name = "CUFFT_INCOMPLETE_PARAMETER_LIST";
    break;
  case CUFFT_INVALID_DEVICE:
    name = "CUFFT_INVALID_DEVICE";
    break;
  case CUFFT_PARSE_ERROR:
    name = "CUFFT_PARSE_ERROR";
    break;
  case CUFFT_NO_WORKSPACE:
    name = "CUFFT_NO_WORKSPACE";
    break;
  case CUFFT_NOT_IMPLEMENTED:
    name = "CUFFT_NOT_IMPLEMENTED";
    break;
  case CUFFT_LICENSE_ERROR:
    name = "CUFFT_LICENSE_ERROR";
    break;
  case CUFFT_NOT_SUPPORTED:
    name = "CUFFT_NOT_SUPPORTED";
    break;
  default:
    UNREACHABLE("Invalid cufft result");
  }
  return fmt::format_to(ctx.out(), "{}: {}", name, result.message());
}

// ==========================================================================
//                     CufftDirection Implementation
// ==========================================================================

dh::CufftDirection::CufftDirection(int direction) noexcept
    : direction_(direction) {}

int dh::CufftDirection::direction() const noexcept { return direction_; }

auto fmt::formatter<dh::CufftDirection>::format(dh::CufftDirection direction,
                                                format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (direction.direction()) {
  case CUFFT_FORWARD:
    name = "CUFFT_FORWARD";
    break;
  case CUFFT_INVERSE:
    name = "CUFFT_INVERSE";
    break;
  default:
    UNREACHABLE("Invalid cufft direction");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     CufftHandle Implementation
// ==========================================================================

dh::CufftHandle::CufftHandle(cufftHandle handle) noexcept : handle_(handle) {}

dh::CufftHandle::CufftHandle(CufftHandle &&other) noexcept
    : handle_(other.handle_) {
  other.handle_ = 0;
}

dh::CufftHandle &dh::CufftHandle::operator=(CufftHandle &&other) noexcept {
  if (this != &other) {
    if (handle_) {
      if (auto result = cufftDestroy(handle_); result != CUFFT_SUCCESS) {
        curaii_logger()->warn(
            "[CufftHandle::operator=] failed with error: \"{}\"",
            CufftResult(result));
      }
    }
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

dh::CufftHandle::~CufftHandle() {
  if (handle_) {
    if (auto result = cufftDestroy(handle_); result != CUFFT_SUCCESS) {
      curaii_logger()->warn(
          "[CufftHandle::~CufftHandle] failed with error: \"{}\"",
          CufftResult(result));
    }
  }
}

tl::expected<dh::CufftHandle, dh::CufftResult>
dh::CufftHandle::try_create() noexcept {
  cufftHandle handle;
  if (auto result = cufftCreate(&handle); result != CUFFT_SUCCESS) {
    curaii_logger()->warn("[CufftHandle::try_create] failed with error: \"{}\"",
                          CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return CufftHandle(handle);
}

tl::expected<dh::CufftHandle, dh::CufftResult>
dh::CufftHandle::try_plan_many(int rank, int *n, int *inembed, int istride,
                               int idist, int *onembed, int ostride, int odist,
                               CufftType type, int batch) noexcept {
  cufftHandle handle;
  if (auto result = cufftPlanMany(&handle, rank, n, inembed, istride, idist,
                                  onembed, ostride, odist, type.type(), batch);
      result != CUFFT_SUCCESS) {
    curaii_logger()->warn(
        "[CufftHandle::try_plan_many] failed with error: \"{}\"",
        CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return CufftHandle(handle);
}

tl::expected<void, dh::CufftResult> dh::CufftHandle::try_xt_get_size_many(
    int rank, long long int *n, long long int *inembed, long long int istride,
    long long int idist, CudaDataType inputtype, long long int *onembed,
    long long int ostride, long long int odist, CudaDataType outputtype,
    long long int batch, size_t *workSize,
    CudaDataType executiontype) noexcept {
  if (auto result = cufftXtGetSizeMany(
          handle_, rank, n, inembed, istride, idist, inputtype.data_type(),
          onembed, ostride, odist, outputtype.data_type(), batch, workSize,
          executiontype.data_type());
      result != CUFFT_SUCCESS) {
    curaii_logger()->warn(
        "[CufftHandle::try_xt_get_size_many] failed with error: \"{}\"",
        CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return {};
}

tl::expected<void, dh::CufftResult> dh::CufftHandle::try_xt_make_plan_many(
    int rank, long long int *n, long long int *inembed, long long int istride,
    long long int idist, CudaDataType inputtype, long long int *onembed,
    long long int ostride, long long int odist, CudaDataType outputtype,
    long long int batch, size_t *workSize, CudaDataType executiontype) {
  if (auto result = cufftXtMakePlanMany(
          handle_, rank, n, inembed, istride, idist, inputtype.data_type(),
          onembed, ostride, odist, outputtype.data_type(), batch, workSize,
          executiontype.data_type());
      result != CUFFT_SUCCESS) {
    curaii_logger()->warn(
        "[CufftHandle::try_xt_make_plan_many] failed with error: \"{}\"",
        CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return {};
}

tl::expected<void, dh::CufftResult>
dh::CufftHandle::try_set_stream(CudaStreamRef stream) noexcept {
  if (auto result = cufftSetStream(handle_, stream.stream());
      result != CUFFT_SUCCESS) {
    curaii_logger()->warn(
        "[CufftHandle::try_set_stream] failed with error: \"{}\"",
        CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return {};
}

tl::expected<void, dh::CufftResult>
dh::CufftHandle::try_xt_exec(void *input, void *output,
                             CufftDirection direction) noexcept {
  if (auto result = cufftXtExec(handle_, input, output, direction.direction());
      result != CUFFT_SUCCESS) {
    curaii_logger()->warn(
        "[CufftHandle::try_xt_exec] failed with error: \"{}\"",
        CufftResult(result));

    return tl::unexpected<CufftResult>(result);
  }

  return {};
}

cufftHandle dh::CufftHandle::handle() noexcept { return handle_; }