#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "holoflow/core/tensor.hh"

namespace holonp::utils {

inline bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

inline std::vector<size_t> get_elem_strides(const holoflow::core::TDesc &desc) {
  const size_t element_size = holoflow::core::size_of(desc.dtype);
  if (!desc.strides.empty()) {
    std::vector<size_t> result;
    result.reserve(desc.strides.size());
    for (const auto stride : desc.strides) {
      if (stride % element_size != 0)
        throw std::invalid_argument("tensor stride is not element-aligned");
      result.push_back(stride / element_size);
    }
    return result;
  }

  std::vector<size_t> result(desc.shape.size());
  size_t              stride = 1;
  for (size_t i = desc.shape.size(); i-- > 0;) {
    result[i] = stride;
    stride *= desc.shape[i];
  }
  return result;
}

inline std::vector<size_t> compact_strides(std::span<const size_t> shape) {
  std::vector<size_t> result(shape.size());
  size_t              stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    result[i] = stride;
    stride *= shape[i];
  }
  return result;
}

inline std::vector<std::int64_t> compact_strides_i64(std::span<const size_t> shape) {
  std::vector<std::int64_t> result(shape.size());
  std::int64_t              stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    if (shape[i] > static_cast<size_t>(std::numeric_limits<std::int64_t>::max()) /
                       static_cast<size_t>(std::max<std::int64_t>(stride, 1)))
      throw std::overflow_error("tensor shape is too large");
    result[i] = stride;
    stride *= static_cast<std::int64_t>(shape[i]);
  }
  return result;
}

inline size_t product_shape(std::span<const size_t> shape) {
  size_t product = 1;
  for (const size_t dimension : shape) {
    if (dimension != 0 && product > std::numeric_limits<size_t>::max() / dimension)
      throw std::overflow_error("tensor shape product overflows size_t");
    product *= dimension;
  }
  return product;
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0)
    axis += ndim;
  return axis;
}

inline std::vector<size_t> get_byte_strides(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty())
    return desc.strides;

  std::vector<size_t> strides(desc.shape.size());
  size_t              stride = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= desc.shape[i];
  }
  return strides;
}

inline std::vector<int> normalize_axes(std::span<const int> axes, int ndim) {
  std::vector<int> result;
  if (axes.empty()) {
    result.resize(static_cast<size_t>(ndim));
    for (int i = 0; i < ndim; ++i)
      result[static_cast<size_t>(i)] = i;
    return result;
  }

  result.reserve(axes.size());
  for (int axis : axes) {
    if (axis < 0)
      axis += ndim;
    if (axis < 0 || axis >= ndim)
      throw std::invalid_argument("axis out of range");
    result.push_back(axis);
  }

  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::invalid_argument("axes must be unique");
  return result;
}

inline std::vector<size_t> bytes_to_elements(std::span<const size_t> byte_strides,
                                             size_t                  element_size) {
  std::vector<size_t> result;
  result.reserve(byte_strides.size());
  for (const auto stride : byte_strides) {
    if (stride % element_size != 0)
      throw std::invalid_argument("tensor stride is not element-aligned");
    result.push_back(stride / element_size);
  }
  return result;
}

inline bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size() && !desc.strides.empty())
    return false;
  if (desc.strides.empty())
    return true;

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected)
      return false;
    expected *= desc.shape[i];
  }
  return true;
}

inline holoflow::core::TDesc make_contiguous_desc(std::vector<size_t>    shape,
                                                  holoflow::core::DType  dtype,
                                                  holoflow::core::MemLoc mem_loc) {
  if (shape.empty())
    return holoflow::core::TDesc({}, dtype, mem_loc,
                                 std::vector<size_t>{holoflow::core::size_of(dtype)});
  return holoflow::core::TDesc(std::move(shape), dtype, mem_loc);
}

} // namespace holonp::utils
