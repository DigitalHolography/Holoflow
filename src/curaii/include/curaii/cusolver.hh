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

#include <cusolverDn.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace curaii::detail {

void log_cusolver_failure(spdlog::level::level_enum lvl, cusolverStatus_t code, const char *expr,
                          const char *file, int line);

} // namespace curaii::detail

#define CUSOLVER_CHECK(expr)                                                                       \
  do {                                                                                             \
    cusolverStatus_t err__ = (expr);                                                               \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                                        \
      ::curaii::detail::log_cusolver_failure(spdlog::level::warn, err__, #expr, __FILE__,          \
                                             __LINE__);                                            \
      throw ::curaii::CusolverError(err__, #expr, __FILE__, __LINE__);                             \
    }                                                                                              \
  } while (false)

#define CUSOLVER_CHECK_NT(expr)                                                                    \
  do {                                                                                             \
    cusolverStatus_t err__ = (expr);                                                               \
    if (err__ != CUSOLVER_STATUS_SUCCESS) {                                                        \
      ::curaii::detail::log_cusolver_failure(spdlog::level::critical, err__, #expr, __FILE__,      \
                                             __LINE__);                                            \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace curaii {

class CusolverError : public std::runtime_error {
public:
  explicit CusolverError(cusolverStatus_t code, const char *what, const char *file, int line);

  [[nodiscard]] cusolverStatus_t code() const noexcept;

private:
  static std::string make_message(cusolverStatus_t code, const char *what, const char *file,
                                  int line);

  cusolverStatus_t code_;
};

class CusolverDnHandle {
public:
  CusolverDnHandle();
  ~CusolverDnHandle() noexcept;

  CusolverDnHandle(const CusolverDnHandle &)            = delete;
  CusolverDnHandle &operator=(const CusolverDnHandle &) = delete;

  CusolverDnHandle(CusolverDnHandle &&other) noexcept;
  CusolverDnHandle &operator=(CusolverDnHandle &&other) noexcept;

  [[nodiscard]] cusolverDnHandle_t get() const noexcept;
  [[nodiscard]] cusolverDnHandle_t release() noexcept;
  void                             reset(cusolverDnHandle_t handle = nullptr) noexcept;
  explicit                         operator bool() const noexcept;

private:
  cusolverDnHandle_t handle_{nullptr};
};

class CusolverDnParams {
public:
  CusolverDnParams();
  ~CusolverDnParams() noexcept;

  CusolverDnParams(const CusolverDnParams &)            = delete;
  CusolverDnParams &operator=(const CusolverDnParams &) = delete;

  CusolverDnParams(CusolverDnParams &&other) noexcept;
  CusolverDnParams &operator=(CusolverDnParams &&other) noexcept;

  [[nodiscard]] cusolverDnParams_t get() const noexcept;
  [[nodiscard]] cusolverDnParams_t release() noexcept;
  void                             reset(cusolverDnParams_t params = nullptr) noexcept;
  explicit                         operator bool() const noexcept;

private:
  cusolverDnParams_t params_{nullptr};
};

} // namespace curaii
