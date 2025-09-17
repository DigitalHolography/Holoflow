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

#include "curaii/cufft.hh"

#include <format>

#include "logger.hh"

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
  case CUFFT_INVALID_DEVICE:
    return "Execution of a plan was on different GPU than plan creation";
  case CUFFT_NO_WORKSPACE:
    return "No workspace has been provided prior to plan execution";
  case CUFFT_NOT_IMPLEMENTED:
    return "Function does not implement functionality for parameters given";
  case CUFFT_NOT_SUPPORTED:
    return "Operation is not supported for parameters given";
  default:
    return "Unknown error";
  }
}

} // namespace

namespace curaii {

CufftError::CufftError(cufftResult code, const char *what, const char *file, int line)
    : std::runtime_error(CufftError::make_message(code, what, file, line)), code_(code) {}

std::string CufftError::make_message(cufftResult code, const char *what, const char *file,
                                     int line) {
  return std::format("cuFFT error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     cufftGetErrorString(code), static_cast<int>(code), what, file, line);
}

cufftResult CufftError::code() const noexcept { return code_; }

CufftHandle::CufftHandle() { CUFFT_CHECK(cufftCreate(&handle_)); }

CufftHandle::~CufftHandle() noexcept {
  if (handle_ != 0) {
    CUFFT_CHECK_NT(cufftDestroy(handle_));
  }
}

CufftHandle::CufftHandle(CufftHandle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = 0;
}

CufftHandle &CufftHandle::operator=(CufftHandle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_       = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

cufftHandle CufftHandle::get() const noexcept { return handle_; }

cufftHandle CufftHandle::release() noexcept {
  cufftHandle tmp = handle_;
  handle_         = 0;
  return tmp;
}

void CufftHandle::reset(cufftHandle handle) noexcept {
  if (handle_ != handle) {
    if (handle_ != 0) {
      CUFFT_CHECK_NT(cufftDestroy(handle_));
    }
    handle_ = handle;
  }
}

CufftHandle::operator bool() const noexcept { return handle_ != 0; }

} // namespace curaii

namespace curaii::detail {

void log_cufft_failure(spdlog::level::level_enum level, cufftResult code, const char *what,
                       const char *file, int line) {
  logger()->log(level, "cuFFT error: {} ({})\n  expression : {}\n  location   : {}:{}",
                cufftGetErrorString(code), static_cast<int>(code), what, file, line);
}

} // namespace curaii::detail
