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

#include <cuda_runtime.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace curaii::detail {

void log_cuda_failure(spdlog::level::level_enum lvl, cudaError_t code, const char *expr,
                      const char *file, int line);

} // namespace curaii::detail

#define CUDA_CHECK(expr)                                                                           \
  do {                                                                                             \
    cudaError_t err__ = (expr);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      ::curaii::detail::log_cuda_failure(spdlog::level::warn, err__, #expr, __FILE__, __LINE__);   \
      throw ::curaii::CudaError(err__, #expr, __FILE__, __LINE__);                                 \
    }                                                                                              \
  } while (false)

#define CUDA_CHECK_NT(expr)                                                                        \
  do {                                                                                             \
    cudaError_t err__ = (expr);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      ::curaii::detail::log_cuda_failure(spdlog::level::critical, err__, #expr, __FILE__,          \
                                         __LINE__);                                                \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false)

namespace curaii {

class CudaError : public std::runtime_error {
public:
  explicit CudaError(cudaError_t code, const char *what, const char *file, int line);

  [[nodiscard]] cudaError_t code() const noexcept;

private:
  static std::string make_message(cudaError_t code, const char *what, const char *file, int line);

  cudaError_t code_;
};

struct HostDeleter {
  void operator()(void *ptr) const noexcept;
};

struct DeviceDeleter {
  void operator()(void *ptr) const noexcept;
};

template <typename T> using unique_host_ptr = std::unique_ptr<T, HostDeleter>;

template <typename T> [[nodiscard]] unique_host_ptr<T> make_unique_host_ptr(size_t count);

template <typename T> using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;

template <typename T>
[[nodiscard]] unique_device_ptr<T> make_unique_device_ptr(size_t count, cudaStream_t stream = 0);

class CudaStream {
public:
  explicit CudaStream(unsigned flags = cudaStreamDefault, int priority = 0);
  ~CudaStream() noexcept;

  CudaStream(const CudaStream &)            = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  CudaStream(CudaStream &&other) noexcept;
  CudaStream &operator=(CudaStream &&other) noexcept;

  cudaStream_t get() const noexcept;
  cudaStream_t release() noexcept;
  void         reset(cudaStream_t s = nullptr) noexcept;
  explicit     operator bool() const noexcept;

private:
  cudaStream_t stream_{nullptr};
};

} // namespace curaii

#define CURAII_CUDA_HXX_INCLUDED
#include "curaii/cuda.hxx"
#undef CURAII_CUDA_HXX_INCLUDED