#include "curaii/v2/cusolver_common.hh"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/logger.hh"

namespace {

const char *cusolverGetErrorString(cusolverStatus_t status) {
  switch (status) {
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
    DH_BUG("Invalid cuSOLVER status");
  }
}

} // namespace

namespace curaii::cusolver {

Error::Error(cusolverStatus_t code, const char *what, const char *file,
             int line)
    : std::runtime_error(make_message(code, what, file, line)), code_(code) {}

cusolverStatus_t Error::code() const noexcept { return code_; }

std::string Error::make_message(cusolverStatus_t code, const char *what,
                                const char *file, int line) {
  std::ostringstream os;
  os << "cuSOLVER error: " << cusolverGetErrorString(code) << " ("
     << static_cast<int>(code) << ")\n"
     << "  expression : " << what << "\n"
     << "  location   : " << file << ":" << line;
  return os.str();
}

} // namespace curaii::cusolver

namespace curaii::cusolver::detail {

void log_cusolver_failure(spdlog::level::level_enum lvl, cusolverStatus_t code,
                          const char *expr, const char *file, int line) {
  logger()->log(lvl, "cuSOLVER error {} ({}): \"{}\"  [{}:{}]",
                cusolverGetErrorString(code), static_cast<int>(code), expr,
                file, line);
}

} // namespace curaii::cusolver::detail