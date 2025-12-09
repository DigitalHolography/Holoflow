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

#include <cufftXt.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace curaii::detail {

void log_cufft_failure(spdlog::level::level_enum lvl, cufftResult code, const char *expr,
                       const char *file, int line);

} // namespace curaii::detail

#define CUFFT_CHECK(expr)                                                                          \
  do {                                                                                             \
    cufftResult err__ = (expr);                                                                    \
    if (err__ != CUFFT_SUCCESS) {                                                                  \
      ::curaii::detail::log_cufft_failure(spdlog::level::warn, err__, #expr, __FILE__, __LINE__);  \
      throw ::curaii::CufftError(err__, #expr, __FILE__, __LINE__);                                \
    }                                                                                              \
  } while (false)

#define CUFFT_CHECK_NT(expr)                                                                       \
  do {                                                                                             \
    cufftResult err__ = (expr);                                                                    \
    if (err__ != CUFFT_SUCCESS) {                                                                  \
      ::curaii::detail::log_cufft_failure(spdlog::level::critical, err__, #expr, __FILE__,         \
                                          __LINE__);                                               \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace curaii {

class CufftError : public std::runtime_error {
public:
  explicit CufftError(cufftResult code, const char *what, const char *file, int line);

  [[nodiscard]] cufftResult code() const noexcept;

private:
  static std::string make_message(cufftResult code, const char *what, const char *file, int line);

  cufftResult code_;
};

class CufftHandle {
public:
  CufftHandle();
  ~CufftHandle() noexcept;

  CufftHandle(const CufftHandle &)            = delete;
  CufftHandle &operator=(const CufftHandle &) = delete;

  CufftHandle(CufftHandle &&other) noexcept;
  CufftHandle &operator=(CufftHandle &&other) noexcept;

  [[nodiscard]] cufftHandle get() const noexcept;
  [[nodiscard]] cufftHandle release() noexcept;
  void                      reset(cufftHandle handle = 0) noexcept;
  explicit                  operator bool() const noexcept;

private:
  cufftHandle handle_{0};
};

} // namespace curaii
