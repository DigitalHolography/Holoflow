#include "curaii/library_types.hh"

#include <fmt/base.h>
#include <fmt/format.h>
#include <library_types.h>
#include <tl/expected.hpp>

#include "curaii/logger.hh"

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    dh::curaii_logger()->critical("Unreachable code reached at {}:{} - {}",    \
                                  __FILE__, __LINE__, msg);                    \
    std::abort();                                                              \
  } while (0)

// ==========================================================================
//                     CudaDataType Implementation
// ==========================================================================

dh::CudaDataType::CudaDataType(cudaDataType_t data_type) noexcept
    : data_type_(data_type) {}

cudaDataType_t dh::CudaDataType::data_type() const noexcept {
  return data_type_;
}

auto fmt::formatter<dh::CudaDataType>::format(dh::CudaDataType data_type,
                                              format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (data_type.data_type()) {
  case CUDA_R_16F:
    name = "CUDA_R_16F";
    break;
  case CUDA_C_16F:
    name = "CUDA_C_16F";
    break;
  case CUDA_R_16BF:
    name = "CUDA_R_16BF";
    break;
  case CUDA_C_16BF:
    name = "CUDA_C_16BF";
    break;
  case CUDA_R_32F:
    name = "CUDA_R_32F";
    break;
  case CUDA_C_32F:
    name = "CUDA_C_32F";
    break;
  case CUDA_R_64F:
    name = "CUDA_R_64F";
    break;
  case CUDA_C_64F:
    name = "CUDA_C_64F";
    break;
  case CUDA_R_4I:
    name = "CUDA_R_4I";
    break;
  case CUDA_C_4I:
    name = "CUDA_C_4I";
    break;
  case CUDA_R_4U:
    name = "CUDA_R_4U";
    break;
  case CUDA_C_4U:
    name = "CUDA_C_4U";
    break;
  case CUDA_R_8I:
    name = "CUDA_R_8I";
    break;
  case CUDA_C_8I:
    name = "CUDA_C_8I";
    break;
  case CUDA_R_8U:
    name = "CUDA_R_8U";
    break;
  case CUDA_C_8U:
    name = "CUDA_C_8U";
    break;
  case CUDA_R_16I:
    name = "CUDA_R_16I";
    break;
  case CUDA_C_16I:
    name = "CUDA_C_16I";
    break;
  case CUDA_R_16U:
    name = "CUDA_R_16U";
    break;
  case CUDA_C_16U:
    name = "CUDA_C_16U";
    break;
  case CUDA_R_32I:
    name = "CUDA_R_32I";
    break;
  case CUDA_C_32I:
    name = "CUDA_C_32I";
    break;
  case CUDA_R_32U:
    name = "CUDA_R_32U";
    break;
  case CUDA_C_32U:
    name = "CUDA_C_32U";
    break;
  case CUDA_R_64I:
    name = "CUDA_R_64I";
    break;
  case CUDA_C_64I:
    name = "CUDA_C_64I";
    break;
  case CUDA_R_64U:
    name = "CUDA_R_64U";
    break;
  case CUDA_C_64U:
    name = "CUDA_C_64U";
    break;
  case CUDA_R_8F_E4M3:
    name = "CUDA_R_8F_E4M3 || CUDA_R_8F_UE4M3";
    break;
  case CUDA_R_8F_E5M2:
    name = "CUDA_R_8F_E5M2";
    break;
  case CUDA_R_8F_UE8M0:
    name = "CUDA_R_8F_UE8M0";
    break;
  case CUDA_R_6F_E2M3:
    name = "CUDA_R_6F_E2M3";
    break;
  case CUDA_R_6F_E3M2:
    name = "CUDA_R_6F_E3M2";
    break;
  case CUDA_R_4F_E2M1:
    name = "CUDA_R_4F_E2M1";
    break;
  default:
    UNREACHABLE("Invalid cuda data type");
  }
  return formatter<string_view>::format(name, ctx);
}