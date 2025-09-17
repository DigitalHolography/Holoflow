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

#include "curaii/nvrtc.hh"

#include <format>

#include "logger.hh"

namespace curaii {

NvrtcError::NvrtcError(nvrtcResult code, const char *what, const char *file, int line)
    : std::runtime_error(NvrtcError::make_message(code, what, file, line)), code_(code) {}

std::string NvrtcError::make_message(nvrtcResult code, const char *what, const char *file,
                                     int line) {
  return std::format("NVRTC error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     nvrtcGetErrorString(code), static_cast<int>(code), what, file, line);
}

nvrtcResult NvrtcError::code() const noexcept { return code_; }

NvrtcProgram::NvrtcProgram(const char *source, const char *name, int numHeaders,
                           const char *const *headers, const char *const *includeNames) {
  create(source, name, numHeaders, headers, includeNames);
}

NvrtcProgram::~NvrtcProgram() noexcept { reset(); }

NvrtcProgram::NvrtcProgram(NvrtcProgram &&other) noexcept : program_(other.program_) {
  other.program_ = nullptr;
}

NvrtcProgram &NvrtcProgram::operator=(NvrtcProgram &&other) noexcept {
  if (this != &other) {
    reset();
    program_       = other.program_;
    other.program_ = nullptr;
  }
  return *this;
}

void NvrtcProgram::create(const char *source, const char *name, int numHeaders,
                          const char *const *headers, const char *const *includeNames) {
  reset();
  NVRTC_CHECK(nvrtcCreateProgram(&program_, source, name, numHeaders, headers, includeNames));
}

nvrtcProgram NvrtcProgram::get() const noexcept { return program_; }

nvrtcProgram NvrtcProgram::release() noexcept {
  nvrtcProgram tmp = program_;
  program_         = nullptr;
  return tmp;
}

void NvrtcProgram::reset(nvrtcProgram program) noexcept {
  if (program_ != program) {
    if (program_) {
      nvrtcProgram tmp = program_;
      NVRTC_CHECK_NT(nvrtcDestroyProgram(&tmp));
    }
    program_ = program;
  }
}

NvrtcProgram::operator bool() const noexcept { return program_ != nullptr; }

} // namespace curaii

namespace curaii::detail {

void log_nvrtc_failure(spdlog::level::level_enum level, nvrtcResult code, const char *what,
                       const char *file, int line) {
  logger()->log(level, "NVRTC error: {} ({})\n  expression : {}\n  location   : {}:{}",
                nvrtcGetErrorString(code), static_cast<int>(code), what, file, line);
}

} // namespace curaii::detail
