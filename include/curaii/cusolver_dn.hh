#pragma once

#include <cusolverDn.h>
#include <fmt/base.h>
#include <tl/expected.hpp>

#include "curaii/cublas.hh"
#include "curaii/cuda_runtime.hh"
#include "curaii/library_types.hh"

namespace dh {

class CusolverStatus {
public:
  explicit CusolverStatus(cusolverStatus_t status) noexcept;

  const char *message() const noexcept;

  cusolverStatus_t status() const noexcept;

private:
  cusolverStatus_t status_;
};

class CusolverDnParamsRef {
public:
  static CusolverDnParamsRef from_raw(cusolverDnParams_t params) noexcept;

  cusolverDnParams_t params() const noexcept;

  friend class CusolverDnParams;

private:
  CusolverDnParamsRef(cusolverDnParams_t params) noexcept;

  cusolverDnParams_t params_;
};

class CusolverDnParams {
public:
  CusolverDnParams(const CusolverDnParams &) = delete;
  CusolverDnParams &operator=(const CusolverDnParams &) = delete;

  CusolverDnParams(CusolverDnParams &&other) noexcept;
  CusolverDnParams &operator=(CusolverDnParams &&other) noexcept;

  ~CusolverDnParams();

  [[nodiscard]]
  static tl::expected<CusolverDnParams, CusolverStatus> try_create() noexcept;

  CusolverDnParamsRef ref() noexcept;

  cusolverDnParams_t params() const noexcept;

private:
  CusolverDnParams(cusolverDnParams_t params) noexcept;

  cusolverDnParams_t params_;
};

class CusolverEigMode {
public:
  explicit CusolverEigMode(cusolverEigMode_t eig_mode) noexcept;

  cusolverEigMode_t eig_mode() const noexcept;

private:
  cusolverEigMode_t eig_mode_;
};

class CusolverDnHandle {
public:
  CusolverDnHandle(const CusolverDnHandle &) = delete;
  CusolverDnHandle &operator=(const CusolverDnHandle &) = delete;

  CusolverDnHandle(CusolverDnHandle &&other) noexcept;
  CusolverDnHandle &operator=(CusolverDnHandle &&other) noexcept;

  ~CusolverDnHandle();

  [[nodiscard]]
  static tl::expected<CusolverDnHandle, CusolverStatus> try_create() noexcept;

  [[nodiscard]]
  tl::expected<void, CusolverStatus>
  try_set_stream(CudaStreamRef stream) noexcept;

  [[nodiscard]]
  tl::expected<void, CusolverStatus> try_x_syevd_buffer_size(
      CusolverDnParamsRef params, CusolverEigMode jobz, CublasFillMode uplo,
      int64_t n, CudaDataType dataTypeA, const void *A, int64_t lda,
      CudaDataType dataTypeW, const void *W, CudaDataType computeType,
      size_t *workspaceInBytesOnDevice,
      size_t *workspaceInBytesOnHost) noexcept;

  [[nodiscard]]
  tl::expected<void, CusolverStatus>
  try_x_syevd(CusolverDnParamsRef params, CusolverEigMode jobz,
              CublasFillMode uplo, int64_t n, CudaDataType dataTypeA, void *A,
              int64_t lda, CudaDataType dataTypeW, void *W,
              CudaDataType computeType, void *bufferOnDevice,
              size_t workspaceInBytesOnDevice, void *bufferOnHost,
              size_t workspaceInBytesOnHost, int *info) noexcept;

private:
  CusolverDnHandle(cusolverDnHandle_t handle) noexcept;

  cusolverDnHandle_t handle_;
};

} // namespace dh

template <> struct fmt::formatter<dh::CusolverStatus> : formatter<string_view> {

  auto format(dh::CusolverStatus status, format_context &ctx) const
      -> format_context::iterator;
};

template <>
struct fmt::formatter<dh::CusolverEigMode> : formatter<string_view> {

  auto format(dh::CusolverEigMode eig_mode, format_context &ctx) const
      -> format_context::iterator;
};