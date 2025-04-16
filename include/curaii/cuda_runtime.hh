#pragma once

#include <cuda_runtime.h>
#include <fmt/base.h>
#include <stdexcept>
#include <tl/expected.hpp>

namespace dh {

class CudaError {
public:
  explicit CudaError(cudaError_t error) noexcept;

  const char *message() const noexcept;

  cudaError_t error() const noexcept;

private:
  cudaError_t error_;
};

class CudaException : public std::runtime_error {
public:
  explicit CudaException(const CudaError &error);

  const CudaError &error() const noexcept;

private:
  CudaError error_;
};

class CudaStreamFlags {
public:
  explicit CudaStreamFlags(unsigned int flags) noexcept;

  unsigned int flags() const noexcept;

private:
  unsigned int flags_;
};

class CudaStreamRef {
public:
  static CudaStreamRef from_raw(cudaStream_t stream) noexcept;

  void synchronize() const;

  cudaStream_t stream() const noexcept;

  friend class CudaStream;

private:
  CudaStreamRef(cudaStream_t stream) noexcept;

  cudaStream_t stream_;
};

class CudaStream {
public:
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  CudaStream(CudaStream &&other);
  CudaStream &operator=(CudaStream &&);

  ~CudaStream();

  static CudaStream create();

  void synchronize() const;

  CudaStreamRef ref() noexcept;

  cudaStream_t stream() const noexcept;

private:
  CudaStream(cudaStream_t stream) noexcept;

  cudaStream_t stream_;
};

} // namespace dh

template <> struct fmt::formatter<dh::CudaError> : formatter<string_view> {

  auto format(dh::CudaError error, format_context &ctx) const
      -> format_context::iterator;
};

template <>
struct fmt::formatter<dh::CudaStreamFlags> : formatter<string_view> {

  auto format(dh::CudaStreamFlags flags, format_context &ctx) const
      -> format_context::iterator;
};

namespace curaii::cuda {

void peek_at_last_error();

}