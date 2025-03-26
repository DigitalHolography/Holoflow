#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fmt/base.h>
#include <tl/expected.hpp>

namespace dh {

class CublasOperation {
public:
  explicit CublasOperation(cublasOperation_t operation) noexcept;

  cublasOperation_t operation() const noexcept;

private:
  cublasOperation_t operation_;
};

class CublasStatus {
public:
  explicit CublasStatus(cublasStatus_t status) noexcept;

  const char *message() const noexcept;

  cublasStatus_t status() const noexcept;

private:
  cublasStatus_t status_;
};

class CublasHandle {
public:
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

  CublasHandle(CublasHandle &&other) noexcept;
  CublasHandle &operator=(CublasHandle &&other) noexcept;

  ~CublasHandle();

  [[nodiscard]]
  static tl::expected<CublasHandle, CublasStatus> try_create() noexcept;

  [[nodiscard]]
  tl::expected<void, CublasStatus> try_set_stream(cudaStream_t stream) noexcept;

  [[nodiscard]]
  tl::expected<void, CublasStatus>
  try_c_gemm_3m(CublasOperation transa, CublasOperation transb, int m, int n,
                int k, const cuComplex *alpha, const cuComplex *A, int lda,
                const cuComplex *B, int ldb, const cuComplex *beta,
                cuComplex *C, int ldc) noexcept;

  cublasHandle_t handle() const noexcept;

private:
  CublasHandle(cublasHandle_t handle) noexcept;

  cublasHandle_t handle_;
};

} // namespace dh

template <>
struct fmt::formatter<dh::CublasOperation> : formatter<string_view> {

  auto format(dh::CublasOperation operation, format_context &ctx) const
      -> format_context::iterator;
};

template <> struct fmt::formatter<dh::CublasStatus> : formatter<string_view> {

  auto format(dh::CublasStatus status, format_context &ctx) const
      -> format_context::iterator;
};