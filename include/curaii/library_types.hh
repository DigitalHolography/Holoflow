#pragma once

#include <fmt/base.h>
#include <library_types.h>
#include <tl/expected.hpp>

namespace dh {

class CudaDataType {
public:
  explicit CudaDataType(cudaDataType_t data_type) noexcept;

  cudaDataType_t data_type() const noexcept;

private:
  cudaDataType_t data_type_;
};

} // namespace dh

template <> struct fmt::formatter<dh::CudaDataType> : formatter<string_view> {

  auto format(dh::CudaDataType data_type, format_context &ctx) const
      -> format_context::iterator;
};