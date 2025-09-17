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

#pragma once

#include <cublas_v2.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace curaii::detail {

void log_cublas_failure(spdlog::level::level_enum lvl, cublasStatus_t code, const char *expr,
                        const char *file, int line);

} // namespace curaii::detail

#define CUBLAS_CHECK(expr)                                                                         \
  do {                                                                                             \
    cublasStatus_t err__ = (expr);                                                                 \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                                          \
      ::curaii::detail::log_cublas_failure(spdlog::level::warn, err__, #expr, __FILE__, __LINE__); \
      throw ::curaii::CublasError(err__, #expr, __FILE__, __LINE__);                               \
    }                                                                                              \
  } while (false)

#define CUBLAS_CHECK_NT(expr)                                                                      \
  do {                                                                                             \
    cublasStatus_t err__ = (expr);                                                                 \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                                          \
      ::curaii::detail::log_cublas_failure(spdlog::level::critical, err__, #expr, __FILE__,        \
                                           __LINE__);                                              \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace curaii {

class CublasError : public std::runtime_error {
public:
  explicit CublasError(cublasStatus_t code, const char *what, const char *file, int line);

  [[nodiscard]] cublasStatus_t code() const noexcept;

private:
  static std::string make_message(cublasStatus_t code, const char *what, const char *file,
                                  int line);

  cublasStatus_t code_;
};

class CublasHandle {
public:
  CublasHandle();
  ~CublasHandle() noexcept;

  CublasHandle(const CublasHandle &)            = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

  CublasHandle(CublasHandle &&) noexcept;
  CublasHandle &operator=(CublasHandle &&) noexcept;

  [[nodiscard]] cublasHandle_t get() const noexcept;
  [[nodiscard]] cublasHandle_t release() noexcept;
  void                         reset(cublasHandle_t handle = nullptr) noexcept;
  explicit                     operator bool() const noexcept;

private:
  cublasHandle_t handle_{nullptr};
};

} // namespace curaii
