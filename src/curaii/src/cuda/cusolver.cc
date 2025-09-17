// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "curaii/cusolver.hh"

#include <format>

#include "logger.hh"

namespace {

const char *cusolverGetErrorString(cusolverStatus_t status) {
  switch (status) {
  case CUSOLVER_STATUS_SUCCESS:
    return "The operation completed successfully";
  case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "Library not initialized";
  case CUSOLVER_STATUS_ALLOC_FAILED:
    return "Resource allocation failed";
  case CUSOLVER_STATUS_INVALID_VALUE:
    return "Invalid value";
  case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "Architecture mismatch";
  case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "Execution failed";
  case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "Internal error";
  case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "Matrix type not supported";
  case CUSOLVER_STATUS_NOT_SUPPORTED:
    return "Operation not supported";
  case CUSOLVER_STATUS_ZERO_PIVOT:
    return "Zero pivot";
  case CUSOLVER_STATUS_INVALID_LICENSE:
    return "Invalid license";
  default:
    return "Unknown error";
  }
}

} // namespace

namespace curaii {

CusolverError::CusolverError(cusolverStatus_t code, const char *what, const char *file, int line)
    : std::runtime_error(CusolverError::make_message(code, what, file, line)), code_(code) {}

std::string CusolverError::make_message(cusolverStatus_t code, const char *what, const char *file,
                                        int line) {
  return std::format("cuSOLVER error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     cusolverGetErrorString(code), static_cast<int>(code), what, file, line);
}

cusolverStatus_t CusolverError::code() const noexcept { return code_; }

CusolverDnHandle::CusolverDnHandle() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }

CusolverDnHandle::~CusolverDnHandle() noexcept {
  if (handle_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }
}

CusolverDnHandle::CusolverDnHandle(CusolverDnHandle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

CusolverDnHandle &CusolverDnHandle::operator=(CusolverDnHandle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_       = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

cusolverDnHandle_t CusolverDnHandle::get() const noexcept { return handle_; }

cusolverDnHandle_t CusolverDnHandle::release() noexcept {
  cusolverDnHandle_t tmp = handle_;
  handle_                = nullptr;
  return tmp;
}

void CusolverDnHandle::reset(cusolverDnHandle_t handle) noexcept {
  if (handle_ != handle) {
    if (handle_) {
      CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
    }
    handle_ = handle;
  }
}

CusolverDnHandle::operator bool() const noexcept { return handle_ != nullptr; }

CusolverDnParams::CusolverDnParams() { CUSOLVER_CHECK(cusolverDnCreateParams(&params_)); }

CusolverDnParams::~CusolverDnParams() noexcept {
  if (params_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroyParams(params_));
  }
}

CusolverDnParams::CusolverDnParams(CusolverDnParams &&other) noexcept : params_(other.params_) {
  other.params_ = nullptr;
}

CusolverDnParams &CusolverDnParams::operator=(CusolverDnParams &&other) noexcept {
  if (this != &other) {
    reset();
    params_       = other.params_;
    other.params_ = nullptr;
  }
  return *this;
}

cusolverDnParams_t CusolverDnParams::get() const noexcept { return params_; }

cusolverDnParams_t CusolverDnParams::release() noexcept {
  cusolverDnParams_t tmp = params_;
  params_                = nullptr;
  return tmp;
}

void CusolverDnParams::reset(cusolverDnParams_t params) noexcept {
  if (params_ != params) {
    if (params_) {
      CUSOLVER_CHECK_NT(cusolverDnDestroyParams(params_));
    }
    params_ = params;
  }
}

CusolverDnParams::operator bool() const noexcept { return params_ != nullptr; }

} // namespace curaii

namespace curaii::detail {

void log_cusolver_failure(spdlog::level::level_enum level, cusolverStatus_t code, const char *what,
                          const char *file, int line) {
  logger()->log(level, "cuSOLVER error: {} ({})\n  expression : {}\n  location   : {}:{}",
                cusolverGetErrorString(code), static_cast<int>(code), what, file, line);
}

} // namespace curaii::detail
