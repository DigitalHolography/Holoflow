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

#include "bug.hh"

namespace holoflow::core {

size_t size_of(DType dtype) noexcept {
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

std::string_view to_string(DType dtype) noexcept {
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

void to_json(nlohmann::json &j, DType dtype) { j = to_string(dtype); }

void from_json(const nlohmann::json &j, DType &dtype) {
  std::string_view s = j.get<std::string_view>();
  if (s == "U8") {
    dtype = DType::U8;
  } else if (s == "U16") {
    dtype = DType::U16;
  } else if (s == "F32") {
    dtype = DType::F32;
  } else if (s == "CF32") {
    dtype = DType::CF32;
  } else {
    throw std::invalid_argument("invalid DType string");
  }
}

std::string_view to_string(MemLoc loc) noexcept {
  switch (loc) {
  case MemLoc::Host:
    return "Host";
  case MemLoc::Device:
    return "Device";
  }

  HOLOFLOW_UNREACHABLE();
}

void to_json(nlohmann::json &j, MemLoc loc) { j = to_string(loc); }

void from_json(const nlohmann::json &j, MemLoc &loc) {
  std::string_view s = j.get<std::string_view>();
  if (s == "Host") {
    loc = MemLoc::Host;
  } else if (s == "Device") {
    loc = MemLoc::Device;
  } else {
    throw std::invalid_argument("invalid MemLoc string");
  }
}

void to_json(nlohmann::json &j, const TDesc &desc) {
  j = nlohmann::json{
      {"shape", desc.shape},
      {"dtype", desc.dtype},
      {"mem_loc", desc.mem_loc},
      {"strides", desc.strides},
  };
}

void from_json(const nlohmann::json &j, TDesc &desc) {
  j.at("shape").get_to(desc.shape);
  j.at("dtype").get_to(desc.dtype);
  j.at("mem_loc").get_to(desc.mem_loc);
  j.at("strides").get_to(desc.strides);
}

namespace {
std::vector<std::size_t> make_default_strides(const std::vector<std::size_t> &shape, DType dtype) {
  std::vector<std::size_t> strides(shape.size());

  std::size_t stride = size_of(dtype);
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }

  return strides;
}
} // namespace

TDesc::TDesc(std::vector<size_t> shape, DType dtype, MemLoc mem_loc)
    : shape(std::move(shape)), dtype(dtype), mem_loc(mem_loc),
      strides(make_default_strides(this->shape, this->dtype)) {}

TDesc::TDesc(std::vector<size_t> shape, DType dtype, MemLoc mem_loc, std::vector<size_t> strides)
    : shape(std::move(shape)), dtype(dtype), mem_loc(mem_loc), strides(std::move(strides)) {}

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
  if (shape.empty()) {
    return 0;
  }

  return strides[0] * shape[0];
}

std::byte *TView::data() {
  HOLOFLOW_CHECK(storage != nullptr, "TView has null storage");
  HOLOFLOW_CHECK(storage->ptr != nullptr, "TView has null data pointer");
  return storage->ptr;
}

bool TView::is_nullptr() { return storage == nullptr || storage->ptr == nullptr; }

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

  storage_ = std::make_unique<Storage>(Storage{desc_.mem_loc, desc_.num_bytes(), data_});
}

void *Tensor::data() noexcept { return data_; }

const void *Tensor::data() const noexcept { return data_; }

const TDesc &Tensor::desc() const noexcept { return desc_; }

TView Tensor::view() noexcept { return {desc_, storage_.get()}; }

} // namespace holoflow::core