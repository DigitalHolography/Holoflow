// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "reference_ops.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>

namespace holonp_test {
namespace {

using holoflow::core::DType;
using holoflow::core::TDesc;

using CF32 = std::complex<float>;

struct Array {
  std::vector<size_t> shape;
  DType               dtype;
  std::vector<std::byte> bytes;
};

size_t element_count(const std::vector<size_t> &shape) {
  size_t result = 1;
  for (const auto extent : shape) {
    if (extent == 0)
      return 0;
    if (result > std::numeric_limits<size_t>::max() / extent)
      throw std::overflow_error("reference array size overflow");
    result *= extent;
  }
  return result;
}

std::vector<size_t> strides_for(const std::vector<size_t> &shape, size_t element_size) {
  std::vector<size_t> strides(shape.size());
  size_t              stride = element_size;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

std::vector<size_t> unravel(size_t linear, const std::vector<size_t> &shape) {
  std::vector<size_t> coords(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    if (shape[i] == 0) {
      coords[i] = 0;
    } else {
      coords[i] = linear % shape[i];
      linear /= shape[i];
    }
  }
  return coords;
}

size_t ravel(const std::vector<size_t> &coords, const std::vector<size_t> &shape) {
  size_t linear = 0;
  for (size_t i = 0; i < shape.size(); ++i)
    linear = linear * shape[i] + coords[i];
  return linear;
}

template <typename T> T read_at(const std::vector<std::byte> &bytes, size_t offset) {
  T value{};
  if (offset + sizeof(T) > bytes.size())
    throw std::invalid_argument("reference input payload is too small");
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

template <typename T> void write_at(std::vector<std::byte> &bytes, size_t offset, const T &value) {
  if (offset + sizeof(T) > bytes.size())
    throw std::invalid_argument("reference output payload is too small");
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <typename T> std::vector<T> values(const Array &array) {
  std::vector<T> result(element_count(array.shape));
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = read_at<T>(array.bytes, i * sizeof(T));
  return result;
}

template <typename T>
Array make_array(std::vector<size_t> shape, DType dtype, const std::vector<T> &data) {
  if (data.size() != element_count(shape))
    throw std::invalid_argument("reference output element count mismatch");
  Array result{std::move(shape), dtype, std::vector<std::byte>(data.size() * sizeof(T))};
  for (size_t i = 0; i < data.size(); ++i)
    write_at(result.bytes, i * sizeof(T), data[i]);
  return result;
}

Array load_array(const TDesc &desc, const std::vector<std::byte> &raw) {
  const size_t count      = element_count(desc.shape);
  const size_t dense_size = count * holoflow::core::size_of(desc.dtype);
  Array        result{desc.shape, desc.dtype, std::vector<std::byte>(dense_size)};

  // Preserve the old reference contract: a payload with the logical dense size is interpreted as
  // already starting at the tensor origin. Otherwise it is a backing store for the descriptor.
  if (raw.size() == dense_size) {
    result.bytes = raw;
    return result;
  }

  const auto strides = desc.strides.empty()
                           ? strides_for(desc.shape, holoflow::core::size_of(desc.dtype))
                           : desc.strides;
  for (size_t linear = 0; linear < count; ++linear) {
    const auto coords = unravel(linear, desc.shape);
    size_t     offset = desc.offset;
    for (size_t axis = 0; axis < coords.size(); ++axis)
      offset += coords[axis] * strides.at(axis);
    const size_t item_size = holoflow::core::size_of(desc.dtype);
    if (offset + item_size > raw.size())
      throw std::invalid_argument("reference strided input payload is too small");
    std::memcpy(result.bytes.data() + linear * item_size, raw.data() + offset, item_size);
  }
  return result;
}

int normalize_axis(int axis, size_t rank) {
  if (axis < 0)
    axis += static_cast<int>(rank);
  if (axis < 0 || static_cast<size_t>(axis) >= rank)
    throw std::invalid_argument("reference axis is out of range");
  return axis;
}

int setting_axis(const nlohmann::json &settings, std::string_view key, int default_axis) {
  if (!settings.contains(key) || settings.at(key).is_null())
    return default_axis;
  return settings.at(key).get<int>();
}

bool setting_bool(const nlohmann::json &settings, std::string_view key, bool default_value) {
  return settings.contains(key) ? settings.at(key).get<bool>() : default_value;
}

DType setting_dtype(const nlohmann::json &settings, std::string_view key, DType default_dtype) {
  if (!settings.contains(key))
    return default_dtype;
  DType dtype = default_dtype;
  settings.at(key).get_to(dtype);
  return dtype;
}

std::vector<size_t> setting_shape(const nlohmann::json &settings) {
  return settings.at("shape").get<std::vector<size_t>>();
}

std::vector<int> setting_axes(const nlohmann::json &settings, size_t rank, size_t count) {
  if (!settings.contains("axes") || settings.at("axes").is_null()) {
    std::vector<int> axes;
    for (size_t i = count; i-- > 0;)
      axes.push_back(static_cast<int>(rank - count + i));
    std::reverse(axes.begin(), axes.end());
    return axes;
  }
  auto axes = settings.at("axes").get<std::vector<int>>();
  for (auto &axis : axes)
    axis = normalize_axis(axis, rank);
  return axes;
}

std::vector<size_t> broadcast_shape(const std::vector<std::vector<size_t>> &shapes) {
  size_t output_rank = 0;
  for (const auto &shape : shapes) output_rank = std::max(output_rank, shape.size());

  std::vector<size_t> result(output_rank, 1);
  for (const auto &shape : shapes) {
    const size_t offset = output_rank - shape.size();
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      const size_t extent = shape[axis];
      auto       &target = result[offset + axis];
      if (target != 1 && extent != 1 && target != extent)
        throw std::invalid_argument("reference broadcast shapes do not match");
      target = std::max(target, extent);
    }
  }
  return result;
}

size_t broadcast_index(const std::vector<size_t> &output_coords,
                       const std::vector<size_t> &input_shape) {
  const size_t offset = output_coords.size() - input_shape.size();
  std::vector<size_t> input_coords(input_shape.size());
  for (size_t axis = 0; axis < input_shape.size(); ++axis)
    input_coords[axis] = input_shape[axis] == 1 ? 0 : output_coords[offset + axis];
  return ravel(input_coords, input_shape);
}

template <typename T, typename F>
Array unary(const Array &input, DType output_dtype, F &&function) {
  auto data = values<T>(input);
  for (auto &value : data)
    value = function(value);
  return make_array(input.shape, output_dtype, data);
}

template <typename T, typename F>
Array binary_same(const Array &lhs, const Array &rhs, F &&function) {
  if (lhs.dtype != rhs.dtype)
    throw std::invalid_argument("reference binary inputs do not match");
  const auto output_shape = broadcast_shape({lhs.shape, rhs.shape});
  auto left  = values<T>(lhs);
  auto right = values<T>(rhs);
  std::vector<T> result(element_count(output_shape));
  for (size_t i = 0; i < result.size(); ++i) {
    const auto coords = unravel(i, output_shape);
    result[i] = function(left[broadcast_index(coords, lhs.shape)],
                          right[broadcast_index(coords, rhs.shape)]);
  }
  return make_array(output_shape, lhs.dtype, result);
}

float as_float(const Array &array, size_t index) {
  switch (array.dtype) {
  case DType::U8:
    return static_cast<float>(values<uint8_t>(array).at(index));
  case DType::U16:
    return static_cast<float>(values<uint16_t>(array).at(index));
  case DType::F32:
    return values<float>(array).at(index);
  case DType::CF32:
    return values<CF32>(array).at(index).real();
  }
  throw std::invalid_argument("unsupported reference dtype");
}

CF32 as_complex(const Array &array, size_t index) {
  switch (array.dtype) {
  case DType::U8:
    return CF32{static_cast<float>(values<uint8_t>(array).at(index)), 0.F};
  case DType::U16:
    return CF32{static_cast<float>(values<uint16_t>(array).at(index)), 0.F};
  case DType::F32:
    return CF32{values<float>(array).at(index), 0.F};
  case DType::CF32:
    return values<CF32>(array).at(index);
  }
  throw std::invalid_argument("unsupported reference dtype");
}

DType promote_multiply(DType lhs, DType rhs) {
  if (lhs == rhs)
    return lhs;
  if (lhs == DType::CF32 || rhs == DType::CF32)
    return DType::CF32;
  if (lhs == DType::F32 || rhs == DType::F32)
    return DType::F32;
  if (lhs == DType::U16 || rhs == DType::U16)
    return DType::U16;
  return DType::U8;
}

Array op_abs(const Array &input) {
  switch (input.dtype) {
  case DType::U8:
    return unary<uint8_t>(input, DType::U8, [](uint8_t value) { return value; });
  case DType::U16:
    return unary<uint16_t>(input, DType::U16, [](uint16_t value) { return value; });
  case DType::F32:
    return unary<float>(input, DType::F32, [](float value) { return std::abs(value); });
  case DType::CF32: {
    const auto source = values<CF32>(input);
    std::vector<float> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) result[i] = std::abs(source[i]);
    return make_array(input.shape, DType::F32, result);
  }
  }
  throw std::invalid_argument("unsupported abs dtype");
}

Array op_conj(const Array &input) {
  if (input.dtype == DType::CF32)
    return unary<CF32>(input, DType::CF32, [](CF32 value) { return std::conj(value); });
  return input;
}

Array op_binary(const Array &lhs, const Array &rhs, std::string_view op) {
  if (op == "add" || op == "subtract") {
    switch (lhs.dtype) {
    case DType::U8:
      return binary_same<uint8_t>(lhs, rhs, [op](uint8_t a, uint8_t b) {
        return op == "add" ? static_cast<uint8_t>(a + b) : static_cast<uint8_t>(a - b);
      });
    case DType::U16:
      return binary_same<uint16_t>(lhs, rhs, [op](uint16_t a, uint16_t b) {
        return op == "add" ? static_cast<uint16_t>(a + b) : static_cast<uint16_t>(a - b);
      });
    case DType::F32:
      return binary_same<float>(lhs, rhs, [op](float a, float b) {
        return op == "add" ? a + b : a - b;
      });
    case DType::CF32:
      return binary_same<CF32>(lhs, rhs, [op](CF32 a, CF32 b) {
        return op == "add" ? a + b : a - b;
      });
    }
  }

  if (op == "equal") {
    if (lhs.dtype != rhs.dtype)
      throw std::invalid_argument("reference equal inputs do not match");
    const auto output_shape = broadcast_shape({lhs.shape, rhs.shape});
    std::vector<uint8_t> result(element_count(output_shape));
    switch (lhs.dtype) {
    case DType::U8: {
      const auto a = values<uint8_t>(lhs), b = values<uint8_t>(rhs);
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = a[broadcast_index(coords, lhs.shape)] == b[broadcast_index(coords, rhs.shape)];
      }
      break;
    }
    case DType::U16: {
      const auto a = values<uint16_t>(lhs), b = values<uint16_t>(rhs);
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = a[broadcast_index(coords, lhs.shape)] == b[broadcast_index(coords, rhs.shape)];
      }
      break;
    }
    case DType::F32: {
      const auto a = values<float>(lhs), b = values<float>(rhs);
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = a[broadcast_index(coords, lhs.shape)] == b[broadcast_index(coords, rhs.shape)];
      }
      break;
    }
    case DType::CF32: {
      const auto a = values<CF32>(lhs), b = values<CF32>(rhs);
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = a[broadcast_index(coords, lhs.shape)] == b[broadcast_index(coords, rhs.shape)];
      }
      break;
    }
    }
    return make_array(output_shape, DType::U8, result);
  }

  if (op == "multiply") {
    const auto output_dtype = promote_multiply(lhs.dtype, rhs.dtype);
    const auto output_shape = broadcast_shape({lhs.shape, rhs.shape});
    if (output_dtype == DType::CF32) {
      std::vector<CF32> result(element_count(output_shape));
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = as_complex(lhs, broadcast_index(coords, lhs.shape)) *
                    as_complex(rhs, broadcast_index(coords, rhs.shape));
      }
      return make_array(output_shape, DType::CF32, result);
    }
    if (output_dtype == DType::F32) {
      std::vector<float> result(element_count(output_shape));
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = as_float(lhs, broadcast_index(coords, lhs.shape)) *
                    as_float(rhs, broadcast_index(coords, rhs.shape));
      }
      return make_array(output_shape, DType::F32, result);
    }
    if (output_dtype == DType::U16) {
      std::vector<uint16_t> result(element_count(output_shape));
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = static_cast<uint16_t>(as_float(lhs, broadcast_index(coords, lhs.shape)) *
                                          as_float(rhs, broadcast_index(coords, rhs.shape)));
      }
      return make_array(output_shape, DType::U16, result);
    }
    std::vector<uint8_t> result(element_count(output_shape));
    for (size_t i = 0; i < result.size(); ++i) {
      const auto coords = unravel(i, output_shape);
      result[i] = static_cast<uint8_t>(as_float(lhs, broadcast_index(coords, lhs.shape)) *
                                       as_float(rhs, broadcast_index(coords, rhs.shape)));
    }
    return make_array(output_shape, DType::U8, result);
  }

  if (op == "divide") {
    const auto output_shape = broadcast_shape({lhs.shape, rhs.shape});
    if (lhs.dtype == DType::CF32) {
      std::vector<CF32> result(element_count(output_shape));
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = as_complex(lhs, broadcast_index(coords, lhs.shape)) /
                    as_complex(rhs, broadcast_index(coords, rhs.shape));
      }
      return make_array(output_shape, DType::CF32, result);
    }
    if (lhs.dtype == DType::F32) {
      std::vector<float> result(element_count(output_shape));
      for (size_t i = 0; i < result.size(); ++i) {
        const auto coords = unravel(i, output_shape);
        result[i] = as_float(lhs, broadcast_index(coords, lhs.shape)) /
                    as_float(rhs, broadcast_index(coords, rhs.shape));
      }
      return make_array(output_shape, DType::F32, result);
    }
    if (lhs.dtype == DType::U16) {
      const auto a = values<uint16_t>(lhs), b = values<uint16_t>(rhs);
      std::vector<uint16_t> result(a.size());
      for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<uint16_t>(a[i] / b[i]);
      return make_array(lhs.shape, DType::U16, result);
    }
    const auto a = values<uint8_t>(lhs), b = values<uint8_t>(rhs);
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<uint8_t>(a[i] / b[i]);
    return make_array(lhs.shape, DType::U8, result);
  }

  throw std::invalid_argument("unknown reference binary operation");
}

Array op_where(const Array &condition, const Array &x, const Array &y) {
  if (x.dtype != y.dtype)
    throw std::invalid_argument("reference where inputs do not match");
  const auto output_shape = broadcast_shape({condition.shape, x.shape, y.shape});
  std::vector<uint8_t> cond(element_count(output_shape));
  for (size_t i = 0; i < cond.size(); ++i) {
    const auto coords = unravel(i, output_shape);
    cond[i] = as_float(condition, broadcast_index(coords, condition.shape)) != 0.F;
  }
  switch (x.dtype) {
  case DType::U8: {
    auto       a = values<uint8_t>(x);
    const auto b = values<uint8_t>(y);
    std::vector<uint8_t> result(cond.size());
    for (size_t i = 0; i < result.size(); ++i) {
      const auto coords = unravel(i, output_shape);
      result[i] = cond[i] ? a[broadcast_index(coords, x.shape)] : b[broadcast_index(coords, y.shape)];
    }
    return make_array(output_shape, DType::U8, result);
  }
  case DType::U16: {
    auto a = values<uint16_t>(x), b = values<uint16_t>(y);
    std::vector<uint16_t> result(cond.size());
    for (size_t i = 0; i < result.size(); ++i) {
      const auto coords = unravel(i, output_shape);
      result[i] = cond[i] ? a[broadcast_index(coords, x.shape)] : b[broadcast_index(coords, y.shape)];
    }
    return make_array(output_shape, DType::U16, result);
  }
  case DType::F32: {
    auto a = values<float>(x), b = values<float>(y);
    std::vector<float> result(cond.size());
    for (size_t i = 0; i < result.size(); ++i) {
      const auto coords = unravel(i, output_shape);
      result[i] = cond[i] ? a[broadcast_index(coords, x.shape)] : b[broadcast_index(coords, y.shape)];
    }
    return make_array(output_shape, DType::F32, result);
  }
  case DType::CF32: {
    auto a = values<CF32>(x), b = values<CF32>(y);
    std::vector<CF32> result(cond.size());
    for (size_t i = 0; i < result.size(); ++i) {
      const auto coords = unravel(i, output_shape);
      result[i] = cond[i] ? a[broadcast_index(coords, x.shape)] : b[broadcast_index(coords, y.shape)];
    }
    return make_array(output_shape, DType::CF32, result);
  }
  }
  throw std::invalid_argument("unsupported reference where dtype");
}

Array op_reduce(const Array &input, const nlohmann::json &settings, std::string_view op) {
  const bool all_axes = !settings.contains("axis") || settings.at("axis").is_null();
  const bool keepdims = setting_bool(settings, "keepdims", false);
  const int  axis     = all_axes ? -1 : normalize_axis(settings.at("axis").get<int>(), input.shape.size());

  std::vector<size_t> output_shape;
  if (all_axes) {
    if (keepdims) output_shape.assign(input.shape.size(), 1);
  } else {
    for (size_t i = 0; i < input.shape.size(); ++i) {
      if (static_cast<int>(i) == axis) {
        if (keepdims) output_shape.push_back(1);
      } else {
        output_shape.push_back(input.shape[i]);
      }
    }
  }

  const size_t output_count = element_count(output_shape);
  const auto selected_for_output = [&](size_t input_index, size_t output_index) {
    if (all_axes) return true;

    const auto input_coords  = unravel(input_index, input.shape);
    const auto output_coords = unravel(output_index, output_shape);
    size_t     output_axis   = 0;
    for (size_t input_axis = 0; input_axis < input.shape.size(); ++input_axis) {
      if (static_cast<int>(input_axis) == axis) continue;
      const size_t expected = output_coords[keepdims ? input_axis : output_axis++];
      if (input_coords[input_axis] != expected) return false;
    }
    return true;
  };

  if (op == "mean") {
    if (input.dtype == DType::CF32) {
      auto source = values<CF32>(input);
      std::vector<CF32> result(output_count);
      for (size_t out = 0; out < output_count; ++out) {
        CF32 sum{};
        size_t count = 0;
        for (size_t in = 0; in < source.size(); ++in) {
          if (selected_for_output(in, out)) { sum += source[in]; ++count; }
        }
        result[out] = sum / static_cast<float>(count);
      }
      return make_array(output_shape, DType::CF32, result);
    }
    std::vector<float> result(output_count);
    for (size_t out = 0; out < output_count; ++out) {
      double sum = 0.0;
      size_t count = 0;
      for (size_t in = 0; in < element_count(input.shape); ++in) {
        if (selected_for_output(in, out)) { sum += as_float(input, in); ++count; }
      }
      result[out] = static_cast<float>(sum / static_cast<double>(count));
    }
    return make_array(output_shape, DType::F32, result);
  }

  auto reduce_one = [&](size_t out) {
    bool first = true;
    CF32 complex_result{};
    float float_result = 0.F;
    uint16_t u16_result = 0;
    for (size_t in = 0; in < element_count(input.shape); ++in) {
      if (!selected_for_output(in, out)) continue;
      if (input.dtype == DType::U8 || input.dtype == DType::U16) {
        const uint16_t value = static_cast<uint16_t>(as_float(input, in));
        if (first || (op == "min" ? value < u16_result : value > u16_result)) u16_result = value;
      } else if (input.dtype == DType::F32) {
        const float value = as_float(input, in);
        if (first || (op == "min" ? value < float_result : value > float_result)) float_result = value;
      } else {
        const CF32 value = as_complex(input, in);
        if (first || (op == "min" ? std::abs(value) < std::abs(complex_result)
                                   : std::abs(value) > std::abs(complex_result))) complex_result = value;
      }
      first = false;
    }
    return std::tuple{u16_result, float_result, complex_result};
  };

  if (input.dtype == DType::U8) {
    std::vector<uint8_t> result(output_count);
    for (size_t i = 0; i < output_count; ++i) result[i] = static_cast<uint8_t>(std::get<0>(reduce_one(i)));
    return make_array(output_shape, DType::U8, result);
  }
  if (input.dtype == DType::U16) {
    std::vector<uint16_t> result(output_count);
    for (size_t i = 0; i < output_count; ++i) result[i] = std::get<0>(reduce_one(i));
    return make_array(output_shape, DType::U16, result);
  }
  if (input.dtype == DType::F32) {
    std::vector<float> result(output_count);
    for (size_t i = 0; i < output_count; ++i) result[i] = std::get<1>(reduce_one(i));
    return make_array(output_shape, DType::F32, result);
  }
  std::vector<CF32> result(output_count);
  for (size_t i = 0; i < output_count; ++i) result[i] = std::get<2>(reduce_one(i));
  return make_array(output_shape, DType::CF32, result);
}

Array op_argmax(const Array &input, const nlohmann::json &settings) {
  const bool all_axes = !settings.contains("axis") || settings.at("axis").is_null();
  const bool keepdims = setting_bool(settings, "keepdims", false);
  if (all_axes) {
    size_t best = 0;
    for (size_t i = 1; i < element_count(input.shape); ++i)
      if (as_float(input, i) > as_float(input, best)) best = i;
    const std::vector<size_t> shape = keepdims ? std::vector<size_t>(input.shape.size(), 1)
                                               : std::vector<size_t>{};
    return make_array(shape, DType::U16, std::vector<uint16_t>{static_cast<uint16_t>(best)});
  }

  const int axis = normalize_axis(settings.at("axis").get<int>(), input.shape.size());
  std::vector<size_t> output_shape;
  for (size_t i = 0; i < input.shape.size(); ++i) {
    if (static_cast<int>(i) == axis) {
      if (keepdims) output_shape.push_back(1);
    } else {
      output_shape.push_back(input.shape[i]);
    }
  }
  std::vector<uint16_t> result(element_count(output_shape));
  for (size_t out = 0; out < result.size(); ++out) {
    const auto out_coords = unravel(out, output_shape);
    size_t     best       = 0;
    float      best_value = -std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < input.shape[axis]; ++k) {
      std::vector<size_t> in_coords(input.shape.size());
      size_t              out_axis = 0;
      for (size_t d = 0; d < input.shape.size(); ++d) {
        if (static_cast<int>(d) == axis) {
          in_coords[d] = k;
        } else {
          in_coords[d] = out_coords[out_axis + (keepdims && d >= static_cast<size_t>(axis) ? 1 : 0)];
          ++out_axis;
        }
      }
      const auto value = as_float(input, ravel(in_coords, input.shape));
      if (value > best_value) { best_value = value; best = k; }
    }
    result[out] = static_cast<uint16_t>(best);
  }
  return make_array(output_shape, DType::U16, result);
}

Array op_slice(const Array &input, const nlohmann::json &settings) {
  struct SliceDim { bool index = false; size_t index_value = 0; int start = 0; int step = 1; size_t count = 0; };
  const auto items = settings.at("slices");
  if (items.size() != input.shape.size())
    throw std::invalid_argument("reference slice rank mismatch");
  std::vector<SliceDim> dims(input.shape.size());
  std::vector<size_t>   output_shape;
  for (size_t axis = 0; axis < input.shape.size(); ++axis) {
    const int length = static_cast<int>(input.shape[axis]);
    if (items[axis].is_number_integer()) {
      int index = items[axis].get<int>();
      if (index < 0) index += length;
      if (index < 0 || index >= length) throw std::out_of_range("reference slice index");
      dims[axis] = {.index = true, .index_value = static_cast<size_t>(index), .start = index,
                    .step = 1, .count = 1};
      continue;
    }
    const auto &item = items[axis];
    const int step = item.contains("step") && !item.at("step").is_null() ? item.at("step").get<int>() : 1;
    if (step == 0) throw std::invalid_argument("reference slice step is zero");
    auto get_bound = [&](std::string_view name) -> std::optional<int> {
      if (!item.contains(name) || item.at(name).is_null()) return std::nullopt;
      return item.at(name).get<int>();
    };
    auto start = get_bound("start");
    auto stop  = get_bound("stop");
    int first, last;
    if (step > 0) {
      first = start.value_or(0); last = stop.value_or(length);
      if (first < 0) first = std::max(first + length, 0); else first = std::min(first, length);
      if (last < 0) last = std::max(last + length, 0); else last = std::min(last, length);
    } else {
      first = start.value_or(length - 1); last = stop.value_or(-1);
      if (first < 0) first = std::max(first + length, -1); else first = std::min(first, length - 1);
      if (last < 0 && stop.has_value()) last = std::max(last + length, -1);
      else if (last >= 0) last = std::min(last, length - 1);
    }
    size_t count = 0;
    if ((step > 0 && first < last) || (step < 0 && first > last))
      count = static_cast<size_t>((std::abs(last - first) + std::abs(step) - 1) / std::abs(step));
    dims[axis] = {.index = false, .index_value = 0, .start = first, .step = step, .count = count};
    output_shape.push_back(count);
  }

  std::vector<std::byte> output(element_count(output_shape) * holoflow::core::size_of(input.dtype));
  for (size_t out = 0; out < element_count(output_shape); ++out) {
    const auto out_coords = unravel(out, output_shape);
    std::vector<size_t> in_coords(input.shape.size());
    size_t              out_axis = 0;
    for (size_t axis = 0; axis < input.shape.size(); ++axis) {
      if (dims[axis].index) in_coords[axis] = dims[axis].index_value;
      else in_coords[axis] = static_cast<size_t>(dims[axis].start + static_cast<int>(out_coords[out_axis++]) * dims[axis].step);
    }
    const size_t item_size = holoflow::core::size_of(input.dtype);
    const size_t source    = ravel(in_coords, input.shape) * item_size;
    std::memcpy(output.data() + out * item_size, input.bytes.data() + source, item_size);
  }
  return {std::move(output_shape), input.dtype, std::move(output)};
}

Array op_transpose(const Array &input, const nlohmann::json &settings) {
  std::vector<int> axes;
  if (settings.contains("axes") && !settings.at("axes").is_null()) {
    axes = settings.at("axes").get<std::vector<int>>();
    for (auto &axis : axes) axis = normalize_axis(axis, input.shape.size());
  } else {
    for (size_t i = input.shape.size(); i-- > 0;) axes.push_back(static_cast<int>(i));
  }
  if (axes.size() != input.shape.size()) throw std::invalid_argument("reference transpose axes rank mismatch");
  std::vector<size_t> output_shape;
  for (const auto axis : axes) output_shape.push_back(input.shape[axis]);
  const size_t item_size = holoflow::core::size_of(input.dtype);
  std::vector<std::byte> output(element_count(output_shape) * item_size);
  for (size_t out = 0; out < element_count(output_shape); ++out) {
    const auto out_coords = unravel(out, output_shape);
    std::vector<size_t> in_coords(input.shape.size());
    for (size_t d = 0; d < axes.size(); ++d) in_coords[axes[d]] = out_coords[d];
    std::memcpy(output.data() + out * item_size,
                input.bytes.data() + ravel(in_coords, input.shape) * item_size, item_size);
  }
  return {std::move(output_shape), input.dtype, std::move(output)};
}

Array op_concatenate(const std::vector<Array> &inputs, const nlohmann::json &settings) {
  if (inputs.empty()) throw std::invalid_argument("reference concatenate needs inputs");
  if (!settings.contains("axis") || settings.at("axis").is_null()) {
    Array result{std::vector<size_t>{element_count(inputs[0].shape)}, inputs[0].dtype, {}};
    for (const auto &input : inputs) result.bytes.insert(result.bytes.end(), input.bytes.begin(), input.bytes.end());
    result.shape[0] = result.bytes.size() / holoflow::core::size_of(result.dtype);
    return result;
  }
  const int axis = normalize_axis(settings.at("axis").get<int>(), inputs[0].shape.size());
  std::vector<size_t> output_shape = inputs[0].shape;
  output_shape[axis] = 0;
  for (const auto &input : inputs) {
    if (input.dtype != inputs[0].dtype || input.shape.size() != output_shape.size())
      throw std::invalid_argument("reference concatenate inputs do not match");
    output_shape[axis] += input.shape[axis];
  }
  const size_t item_size = holoflow::core::size_of(inputs[0].dtype);
  std::vector<std::byte> output(element_count(output_shape) * item_size);
  for (size_t out = 0; out < element_count(output_shape); ++out) {
    auto coords = unravel(out, output_shape);
    size_t base  = 0;
    for (const auto &input : inputs) {
      if (coords[axis] < base + input.shape[axis]) {
        coords[axis] -= base;
        const size_t source = ravel(coords, input.shape) * item_size;
        std::memcpy(output.data() + out * item_size, input.bytes.data() + source, item_size);
        break;
      }
      base += input.shape[axis];
    }
  }
  return {std::move(output_shape), inputs[0].dtype, std::move(output)};
}

Array op_meshgrid_input(const Array &input, const std::vector<size_t> &output_shape,
                        size_t input_axis, const std::vector<size_t> &source_axes) {
  const size_t item_size = holoflow::core::size_of(input.dtype);
  std::vector<std::byte> output(element_count(output_shape) * item_size);
  for (size_t out = 0; out < element_count(output_shape); ++out) {
    const auto coords = unravel(out, output_shape);
    const size_t source_index = coords[source_axes[input_axis]];
    std::memcpy(output.data() + out * item_size, input.bytes.data() + source_index * item_size, item_size);
  }
  return {output_shape, input.dtype, std::move(output)};
}

std::vector<Array> op_meshgrid(const std::vector<Array> &inputs, const nlohmann::json &settings) {
  const bool xy     = settings.value("indexing", "xy") == "xy";
  const bool sparse = settings.value("sparse", false);
  const size_t rank = inputs.size();
  std::vector<size_t> full_shape;
  for (const auto &input : inputs) full_shape.push_back(input.shape.at(0));
  std::vector<size_t> source_axes(rank);
  std::iota(source_axes.begin(), source_axes.end(), 0);
  if (xy && rank >= 2) {
    std::swap(source_axes[0], source_axes[1]);
    std::swap(full_shape[0], full_shape[1]);
  }
  std::vector<Array> result;
  for (size_t input_axis = 0; input_axis < rank; ++input_axis) {
    auto output_shape = full_shape;
    if (sparse) {
      for (size_t d = 0; d < rank; ++d)
        if (d != source_axes[input_axis]) output_shape[d] = 1;
    }
    result.push_back(op_meshgrid_input(inputs[input_axis], output_shape, input_axis, source_axes));
  }
  return result;
}

float transform_scale(const nlohmann::json &settings, size_t total_length, bool inverse) {
  const auto norm = settings.value("norm", "backward");
  if (norm == "ortho") return 1.F / std::sqrt(static_cast<float>(total_length));
  if (norm == "forward") return inverse ? 1.F : 1.F / static_cast<float>(total_length);
  return inverse ? 1.F / static_cast<float>(total_length) : 1.F;
}

Array transform_axis(const Array &input, int axis, bool inverse, size_t output_length,
                     const nlohmann::json &settings) {
  const int          normalized_axis = normalize_axis(axis, input.shape.size());
  const size_t       input_length    = input.shape[normalized_axis];
  auto               output_shape    = input.shape;
  output_shape[normalized_axis] = output_length;
  std::vector<CF32> output(element_count(output_shape));
  const float sign = inverse ? 1.F : -1.F;
  const float scale = transform_scale(settings, input_length, inverse);
  auto base_shape = input.shape;
  base_shape.erase(base_shape.begin() + normalized_axis);
  const size_t slices = element_count(base_shape);
  for (size_t slice = 0; slice < slices; ++slice) {
    const auto outer_coords = unravel(slice, base_shape);
    std::vector<size_t> base_coords(input.shape.size());
    for (size_t dimension = 0, outer_dimension = 0; dimension < input.shape.size(); ++dimension) {
      if (static_cast<int>(dimension) == normalized_axis) continue;
      base_coords[dimension] = outer_coords[outer_dimension++];
    }
    for (size_t k = 0; k < output_length; ++k) {
      CF32 sum{};
      for (size_t n = 0; n < input_length; ++n) {
        auto coords = base_coords;
        coords[normalized_axis] = n;
        const auto value = as_complex(input, ravel(coords, input.shape));
        const float angle = sign * 2.F * std::numbers::pi_v<float> * static_cast<float>(k * n) /
                            static_cast<float>(input_length);
        sum += value * CF32{std::cos(angle), std::sin(angle)};
      }
      auto output_coords = base_coords;
      output_coords[normalized_axis] = k;
      output[ravel(output_coords, output_shape)] = sum * scale;
    }
  }
  return make_array(output_shape, DType::CF32, output);
}

Array op_fft(const Array &input, const nlohmann::json &settings) {
  return transform_axis(input, setting_axis(settings, "axis", -1), false,
                        input.shape.back(), settings);
}

Array op_fft2(const Array &input, const nlohmann::json &settings) {
  const auto axes = setting_axes(settings, input.shape.size(), 2);
  auto       result = transform_axis(input, axes[1], false, input.shape[axes[1]], settings);
  return transform_axis(result, axes[0], false, result.shape[axes[0]], settings);
}

Array op_rfft(const Array &input, const nlohmann::json &settings) {
  const int axis = setting_axis(settings, "axis", -1);
  const auto normalized = normalize_axis(axis, input.shape.size());
  return transform_axis(input, normalized, false, input.shape[normalized] / 2 + 1, settings);
}

Array op_rfft2(const Array &input, const nlohmann::json &settings) {
  const auto axes = setting_axes(settings, input.shape.size(), 2);
  auto       result = transform_axis(input, axes[1], false, input.shape[axes[1]] / 2 + 1, settings);
  return transform_axis(result, axes[0], false, result.shape[axes[0]], settings);
}

Array op_irfft2(const Array &input, const nlohmann::json &settings) {
  const auto axes = setting_axes(settings, input.shape.size(), 2);
  const int  last_axis = axes[1];
  const size_t half_length = input.shape[last_axis];
  const size_t full_length = 2 * (half_length - 1);
  auto full_shape = input.shape;
  full_shape[last_axis] = full_length;
  std::vector<CF32> full(element_count(full_shape));
  for (size_t linear = 0; linear < element_count(full_shape); ++linear) {
    auto coords = unravel(linear, full_shape);
    const size_t k = coords[last_axis];
    if (k < half_length) {
      coords[last_axis] = k;
      full[linear] = as_complex(input, ravel(coords, input.shape));
    } else {
      const size_t mirrored = full_length - k;
      coords[last_axis] = mirrored;
      full[linear] = std::conj(as_complex(input, ravel(coords, input.shape)));
    }
  }
  Array full_array = make_array(full_shape, DType::CF32, full);
  auto inverse_first = transform_axis(full_array, axes[0], true, full_shape[axes[0]], settings);
  auto inverse_last  = transform_axis(inverse_first, axes[1], true, full_length, settings);
  std::vector<float> result(element_count(inverse_last.shape));
  const auto complex_result = values<CF32>(inverse_last);
  for (size_t i = 0; i < result.size(); ++i) result[i] = complex_result[i].real();
  return make_array(inverse_last.shape, DType::F32, result);
}

Array op_fftshift(const Array &input, const nlohmann::json &settings) {
  std::vector<int> axes;
  if (!settings.contains("axes") || settings.at("axes").is_null()) {
    for (size_t axis = 0; axis < input.shape.size(); ++axis) axes.push_back(static_cast<int>(axis));
  } else {
    axes = settings.at("axes").get<std::vector<int>>();
    for (auto &axis : axes) axis = normalize_axis(axis, input.shape.size());
  }
  std::vector<size_t> shifts(input.shape.size());
  for (const auto axis : axes) shifts[axis] = (input.shape[axis] + 1) / 2;
  const size_t item_size = holoflow::core::size_of(input.dtype);
  std::vector<std::byte> output(input.bytes.size());
  for (size_t out = 0; out < element_count(input.shape); ++out) {
    auto source_coords = unravel(out, input.shape);
    for (size_t axis = 0; axis < source_coords.size(); ++axis)
      source_coords[axis] = (source_coords[axis] + shifts[axis]) % input.shape[axis];
    const size_t source = ravel(source_coords, input.shape);
    std::memcpy(output.data() + out * item_size, input.bytes.data() + source * item_size, item_size);
  }
  return {input.shape, input.dtype, std::move(output)};
}

Array dispatch(const std::string &op, const std::vector<Array> &inputs, const nlohmann::json &settings) {
  if (op == "abs") return op_abs(inputs.at(0));
  if (op == "conj") return op_conj(inputs.at(0));
  if (op == "add" || op == "subtract" || op == "multiply" || op == "divide" || op == "equal")
    return op_binary(inputs.at(0), inputs.at(1), op);
  if (op == "where") return op_where(inputs.at(0), inputs.at(1), inputs.at(2));
  if (op == "min" || op == "max" || op == "mean") return op_reduce(inputs.at(0), settings, op);
  if (op == "argmax") return op_argmax(inputs.at(0), settings);
  if (op == "ascontiguousarray" || op == "copy") return inputs.at(0);
  if (op == "reshape") {
    auto result = inputs.at(0);
    result.shape = setting_shape(settings);
    return result;
  }
  if (op == "transpose") return op_transpose(inputs.at(0), settings);
  if (op == "concatenate") return op_concatenate(inputs, settings);
  if (op == "slice") return op_slice(inputs.at(0), settings);
  if (op == "meshgrid") throw std::logic_error("meshgrid dispatch handled separately");
  if (op == "fft") return op_fft(inputs.at(0), settings);
  if (op == "fft2") return op_fft2(inputs.at(0), settings);
  if (op == "rfft") return op_rfft(inputs.at(0), settings);
  if (op == "rfft2") return op_rfft2(inputs.at(0), settings);
  if (op == "irfft2") return op_irfft2(inputs.at(0), settings);
  if (op == "fftshift") return op_fftshift(inputs.at(0), settings);
  if (op == "arange") {
    const double start = settings.at("start").get<double>();
    const double stop  = settings.at("stop").get<double>();
    const double step  = settings.at("step").get<double>();
    const auto dtype = setting_dtype(settings, "dtype", DType::F32);
    const size_t count = step > 0.0 ? (stop <= start ? 0 : static_cast<size_t>(std::ceil((stop - start) / step)))
                                    : (stop >= start ? 0 : static_cast<size_t>(std::ceil((stop - start) / step)));
    if (dtype == DType::F32) { std::vector<float> data(count); for (size_t i = 0; i < count; ++i) data[i] = static_cast<float>(start + step * i); return make_array({count}, DType::F32, data); }
    if (dtype == DType::U8) { std::vector<uint8_t> data(count); for (size_t i = 0; i < count; ++i) data[i] = static_cast<uint8_t>(start + step * i); return make_array({count}, DType::U8, data); }
    if (dtype == DType::U16) { std::vector<uint16_t> data(count); for (size_t i = 0; i < count; ++i) data[i] = static_cast<uint16_t>(start + step * i); return make_array({count}, DType::U16, data); }
    std::vector<CF32> data(count); for (size_t i = 0; i < count; ++i) data[i] = CF32{static_cast<float>(start + step * i), 0.F}; return make_array({count}, DType::CF32, data);
  }
  if (op == "asarray") {
    return make_array({1}, DType::F32, std::vector<float>{settings.at("value").get<float>()});
  }
  if (op == "zeros") {
    const auto shape = setting_shape(settings);
    const auto dtype = setting_dtype(settings, "dtype", DType::F32);
    if (dtype == DType::U8) return make_array(shape, dtype, std::vector<uint8_t>(element_count(shape)));
    if (dtype == DType::U16) return make_array(shape, dtype, std::vector<uint16_t>(element_count(shape)));
    if (dtype == DType::CF32) return make_array(shape, dtype, std::vector<CF32>(element_count(shape)));
    return make_array(shape, dtype, std::vector<float>(element_count(shape)));
  }
  throw std::invalid_argument("unknown C++ reference operation: " + op);
}

} // namespace

ReferenceOutput invoke_reference(const ReferenceInput &input) {
  std::vector<Array> arrays;
  arrays.reserve(input.input_descs.size());
  if (input.input_descs.size() != input.input_bytes.size())
    throw std::invalid_argument("reference input descriptor/payload count mismatch");
  for (size_t i = 0; i < input.input_descs.size(); ++i)
    arrays.push_back(load_array(input.input_descs[i], input.input_bytes[i]));

  std::vector<Array> outputs;
  if (input.op == "meshgrid") outputs = op_meshgrid(arrays, input.settings);
  else outputs.push_back(dispatch(input.op, arrays, input.settings));
  if (outputs.size() != input.n_outputs)
    throw std::invalid_argument("reference output count mismatch");

  ReferenceOutput result;
  result.output_bytes.reserve(outputs.size());
  for (auto &output : outputs) result.output_bytes.push_back(std::move(output.bytes));
  return result;
}

} // namespace holonp_test
