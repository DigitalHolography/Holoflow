#pragma once

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <fmt/base.h>
#include <tl/expected.hpp>

#include "curaii/cuda_runtime.hh"
#include "curaii/library_types.hh"

namespace dh {

class CufftType {
public:
  explicit CufftType(cufftType type) noexcept;

  cufftType type() const noexcept;

private:
  cufftType type_;
};

class CufftResult {
public:
  explicit CufftResult(cufftResult res) noexcept;

  const char *message() const noexcept;

  cufftResult result() const noexcept;

private:
  cufftResult result_;
};

class CufftDirection {
public:
  explicit CufftDirection(int direction) noexcept;

  int direction() const noexcept;

private:
  int direction_;
};

class CufftHandle {
public:
  CufftHandle(const CufftHandle &) = delete;
  CufftHandle &operator=(const CufftHandle &) = delete;

  CufftHandle(CufftHandle &&other) noexcept;
  CufftHandle &operator=(CufftHandle &&other) noexcept;

  ~CufftHandle();

  [[nodiscard]]
  static tl::expected<CufftHandle, CufftResult>
  try_plan_many(int rank, int *n, int *inembed, int istride, int idist,
                int *onembed, int ostride, int odist, CufftType type,
                int batch) noexcept;

  [[nodiscard]]
  static tl::expected<CufftHandle, CufftResult> try_xt_make_plan_many(
      int rank, long long int *n, long long int *inembed, long long int istride,
      long long int idist, CudaDataType inputtype, long long int *onembed,
      long long int ostride, long long int odist, CudaDataType outputtype,
      long long int batch, CudaDataType executiontype);

  [[nodiscard]]
  tl::expected<void, CufftResult> try_set_stream(CudaStreamRef stream) noexcept;

  [[nodiscard]]
  tl::expected<void, CufftResult>
  try_xt_exec(void *input, void *output, CufftDirection direction) noexcept;

  cufftHandle handle() noexcept;

private:
  CufftHandle(cufftHandle handle) noexcept;

  cufftHandle handle_;
};

} // namespace dh

template <> struct fmt::formatter<dh::CufftType> : formatter<string_view> {

  auto format(dh::CufftType type, format_context &ctx) const
      -> format_context::iterator;
};

template <> struct fmt::formatter<dh::CufftResult> : formatter<string_view> {

  auto format(dh::CufftResult result, format_context &ctx) const
      -> format_context::iterator;
};

template <> struct fmt::formatter<dh::CufftDirection> : formatter<string_view> {

  auto format(dh::CufftDirection direction, format_context &ctx) const
      -> format_context::iterator;
};