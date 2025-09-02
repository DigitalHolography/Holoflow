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

#include "holoflow/core/tensor.hh"

#include <limits>
#include <stdexcept>

#include "core/bug.hh"

namespace holoflow::core {

constexpr size_t size_of(DType dtype) noexcept {
  switch (dtype) {
  case DType::U8:
    return 1;
  case DType::U16:
    return 2;
  case DType::F32:
    return 4;
  case DType::CF32:
    return 8;
  }

  HOLOFLOW_UNREACHABLE();
}

constexpr std::string_view to_string(DType dtype) noexcept {
  switch (dtype) {
  case DType::U8:
    return "U8";
  case DType::U16:
    return "U16";
  case DType::F32:
    return "F32";
  case DType::CF32:
    return "CF32";
  }

  HOLOFLOW_UNREACHABLE();
}

constexpr std::string_view to_string(MemLoc loc) noexcept {
  switch (loc) {
  case MemLoc::Host:
    return "Host";
  case MemLoc::Device:
    return "Device";
  }

  HOLOFLOW_UNREACHABLE();
}

size_t TDesc::rank() const noexcept { return shape.size(); }

size_t TDesc::num_elements() const {
  constexpr size_t max = std::numeric_limits<size_t>::max();
  size_t           n   = 1;
  for (auto d : shape) {
    if (d == 0)
      return 0;
    if (n > max / d) {
      throw std::overflow_error("num_elements overflow");
    }
    n *= d;
  }
  return n;
}

size_t TDesc::num_bytes() const {
  size_t elems = num_elements();
  size_t s     = size_of(dtype);
  if (elems > std::numeric_limits<size_t>::max() / s) {
    throw std::overflow_error("num_bytes overflow");
  }
  return elems * s;
}

Tensor::Tensor(const TDesc &desc) : desc_(desc), data_(nullptr) {
  switch (desc_.mem_loc) {
  case MemLoc::Host:
    h_data_ = curaii::make_unique_host_ptr<std::byte>(desc_.num_bytes());
    data_   = h_data_.get();
    break;
  case MemLoc::Device:
    d_data_ = curaii::make_unique_device_ptr<std::byte>(desc_.num_bytes());
    data_   = d_data_.get();
    break;
  }
}

void *Tensor::data() noexcept { return data_; }

const void *Tensor::data() const noexcept { return data_; }

const TDesc &Tensor::desc() const noexcept { return desc_; }

TView Tensor::view() noexcept { return {data_, desc_}; }

CTView Tensor::cview() const noexcept { return {data_, desc_}; }

} // namespace holoflow::core