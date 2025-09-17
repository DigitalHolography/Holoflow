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

#include <nvrtc.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace curaii::detail {

void log_nvrtc_failure(spdlog::level::level_enum lvl, nvrtcResult code, const char *expr,
                       const char *file, int line);

} // namespace curaii::detail

#define NVRTC_CHECK(expr)                                                                          \
  do {                                                                                             \
    nvrtcResult err__ = (expr);                                                                    \
    if (err__ != NVRTC_SUCCESS) {                                                                  \
      ::curaii::detail::log_nvrtc_failure(spdlog::level::warn, err__, #expr, __FILE__, __LINE__);  \
      throw ::curaii::NvrtcError(err__, #expr, __FILE__, __LINE__);                                \
    }                                                                                              \
  } while (false)

#define NVRTC_CHECK_NT(expr)                                                                       \
  do {                                                                                             \
    nvrtcResult err__ = (expr);                                                                    \
    if (err__ != NVRTC_SUCCESS) {                                                                  \
      ::curaii::detail::log_nvrtc_failure(spdlog::level::critical, err__, #expr, __FILE__,         \
                                          __LINE__);                                               \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace curaii {

class NvrtcError : public std::runtime_error {
public:
  explicit NvrtcError(nvrtcResult code, const char *what, const char *file, int line);

  [[nodiscard]] nvrtcResult code() const noexcept;

private:
  static std::string make_message(nvrtcResult code, const char *what, const char *file, int line);

  nvrtcResult code_;
};

class NvrtcProgram {
public:
  NvrtcProgram() noexcept = default;
  NvrtcProgram(const char *source, const char *name, int numHeaders = 0,
               const char *const *headers = nullptr, const char *const *includeNames = nullptr);
  ~NvrtcProgram() noexcept;

  NvrtcProgram(const NvrtcProgram &)            = delete;
  NvrtcProgram &operator=(const NvrtcProgram &) = delete;

  NvrtcProgram(NvrtcProgram &&other) noexcept;
  NvrtcProgram &operator=(NvrtcProgram &&other) noexcept;

  void create(const char *source, const char *name, int numHeaders = 0,
              const char *const *headers = nullptr,
              const char *const *includeNames = nullptr);

  [[nodiscard]] nvrtcProgram get() const noexcept;
  [[nodiscard]] nvrtcProgram release() noexcept;
  void                      reset(nvrtcProgram program = nullptr) noexcept;
  explicit                  operator bool() const noexcept;

private:
  nvrtcProgram program_{nullptr};
};

} // namespace curaii
