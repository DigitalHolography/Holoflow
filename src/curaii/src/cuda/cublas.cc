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

#include "curaii/cublas.hh"

#include <format>

#include "logger.hh"

namespace curaii {

CublasError::CublasError(cublasStatus_t code, const char *what, const char *file, int line)
    : std::runtime_error(CublasError::make_message(code, what, file, line)), code_(code) {}

std::string CublasError::make_message(cublasStatus_t code, const char *what, const char *file,
                                      int line) {
  return std::format("cuBLAS error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     cublasGetStatusString(code), static_cast<int>(code), what, file, line);
}

cublasStatus_t CublasError::code() const noexcept { return code_; }

CublasHandle::CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

CublasHandle::~CublasHandle() noexcept {
  if (handle_) {
    CUBLAS_CHECK_NT(cublasDestroy(handle_));
  }
}

CublasHandle::CublasHandle(CublasHandle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

CublasHandle &CublasHandle::operator=(CublasHandle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_       = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

cublasHandle_t CublasHandle::get() const noexcept { return handle_; }

cublasHandle_t CublasHandle::release() noexcept {
  cublasHandle_t tmp = handle_;
  handle_            = nullptr;
  return tmp;
}

void CublasHandle::reset(cublasHandle_t handle) noexcept {
  if (handle_ != handle) {
    if (handle_) {
      CUBLAS_CHECK_NT(cublasDestroy(handle_));
    }
    handle_ = handle;
  }
}

CublasHandle::operator bool() const noexcept { return handle_ != nullptr; }

} // namespace curaii

namespace curaii::detail {
void log_cublas_failure(spdlog::level::level_enum level, cublasStatus_t code, const char *what,
                        const char *file, int line) {
  logger()->log(level, "cuBLAS error: {} ({})\n  expression : {}\n  location   : {}:{}",
                cublasGetStatusString(code), static_cast<int>(code), what, file, line);
}

} // namespace curaii::detail