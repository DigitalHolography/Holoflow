#pragma once

#include <cuda_runtime.h>
#include <fmt/base.h>
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

  [[nodiscard]]
  tl::expected<void, CudaError> try_synchronize() const noexcept;

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

  CudaStream(CudaStream &&other) noexcept;
  CudaStream &operator=(CudaStream &&) noexcept;

  ~CudaStream() noexcept;

  [[nodiscard]]
  static tl::expected<CudaStream, CudaError> try_create() noexcept;

  [[nodiscard]]
  tl::expected<void, CudaError> try_synchronize() const noexcept;

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