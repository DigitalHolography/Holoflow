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

#include "curaii/cuda.hh"

#include <format>

#include "logger.hh"

namespace curaii {

CudaError::CudaError(cudaError_t code, const char *what, const char *file, int line)
    : std::runtime_error(CudaError::make_message(code, what, file, line)), code_(code) {}

std::string CudaError::make_message(cudaError_t code, const char *what, const char *file,
                                    int line) {
  return std::format("CUDA error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     cudaGetErrorString(code), static_cast<int>(code), what, file, line);
}

cudaError_t CudaError::code() const noexcept { return code_; }

void HostDeleter::operator()(void *ptr) const noexcept {
  if (ptr) {
    CUDA_CHECK_NT(cudaFreeHost(ptr));
  }
}

void DeviceDeleter::operator()(void *ptr) const noexcept {
  if (ptr) {
    CUDA_CHECK_NT(cudaFree(ptr));
  }
}

CudaStream::CudaStream(unsigned flags, int priority) {
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, flags, priority));
}

CudaStream::~CudaStream() noexcept {
  if (stream_) {
    CUDA_CHECK_NT(cudaStreamDestroy(stream_));
  }
}

CudaStream::CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

CudaStream &CudaStream::operator=(CudaStream &&other) noexcept {
  if (this != &other) {
    reset();
    stream_       = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

cudaStream_t CudaStream::get() const noexcept { return stream_; }

cudaStream_t CudaStream::release() noexcept {
  auto tmp = stream_;
  stream_  = nullptr;
  return tmp;
}

void CudaStream::reset(cudaStream_t s) noexcept {
  if (stream_ != s) {
    if (stream_) {
      CUDA_CHECK_NT(cudaStreamDestroy(stream_));
    }
    stream_ = s;
  }
}

CudaStream::operator bool() const noexcept { return stream_ != nullptr; }

} // namespace curaii

namespace curaii::detail {

void log_cuda_failure(spdlog::level::level_enum lvl, cudaError_t code, const char *expr,
                      const char *file, int line) {
  logger()->log(lvl, "CUDA error {} ({}): \"{}\"  [{}:{}]", cudaGetErrorString(code),
                static_cast<int>(code), expr, file, line);
}

} // namespace curaii::detail