#include "holoflow/tensor.hh"

#include <algorithm>
#include <glog/logging.h>
#include <numeric>
#include <vector>

namespace dh {

// ==========================================================================
//                     DataType Utilities
// ==========================================================================

size_t size_of(DataType data_type) {
  switch (data_type) {
  case DataType::U8:
    return 1;
  case DataType::U16:
    return 2;
  case DataType::F32:
    return 4;
  case DataType::CF32:
    return 8;
  default:
    LOG(FATAL) << "Unknown DataType";
    std::exit(EXIT_FAILURE);
  }
}

// ==========================================================================
//                     TensorMeta Implementation
// ==========================================================================

TensorMeta::TensorMeta(DataType data_type, MemoryLocation memory_location,
                       const std::vector<size_t> &shape)
    : data_type(data_type), memory_location(memory_location), shape(shape) {

  CHECK(!shape.empty()) << "Tensor shape cannot be empty";
  CHECK(std::all_of(shape.begin(), shape.end(), [](size_t val) {
    return val > 0;
  })) << "Tensor shape cannot contain zero";

  // Compute row-major strides
  strides.resize(shape.size());
  size_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }

  VLOG(2) << "TensorMeta constructed: dtype=" << static_cast<int>(data_type)
          << ", location=" << static_cast<int>(memory_location)
          << ", shape=" << shape.size() << ", stride=" << strides.size();
}

TensorMeta::TensorMeta(DataType data_type, MemoryLocation memory_location,
                       const std::vector<size_t> &shape,
                       const std::vector<size_t> &strides)
    : data_type(data_type), memory_location(memory_location), shape(shape),
      strides(strides) {

  CHECK(!shape.empty()) << "Tensor shape cannot be empty";
  CHECK(std::all_of(shape.begin(), shape.end(), [](size_t val) {
    return val > 0;
  })) << "Tensor shape cannot contain zero";

  CHECK(shape.size() == strides.size())
      << "Tensor shape and stride dimensions must match";
  CHECK(std::all_of(strides.begin(), strides.end(), [](size_t val) {
    return val > 0;
  })) << "Tensor stride cannot contain zero";

  // TODO: Check stride is large enough to store elements.

  VLOG(2) << "TensorMeta constructed: dtype=" << static_cast<int>(data_type)
          << ", location=" << static_cast<int>(memory_location)
          << ", shape=" << shape.size() << ", stride=" << strides.size();
}

size_t TensorMeta::size() const {
  return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
}

size_t TensorMeta::size_in_bytes() const {
  if (shape.empty() || strides.empty()) {
    return 0;
  }

  // Compute the last accessed element based on strides
  size_t last_offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    last_offset += (shape[i] - 1) * strides[i];
  }

  return (last_offset + 1) * size_of(data_type);
}

// ==========================================================================
//                     TensorView Implementation
// ==========================================================================

TensorView::TensorView(void *data, const TensorMeta &meta)
    : data_(data), meta_(meta) {
  CHECK(data != nullptr) << "TensorView received a null data pointer";

  VLOG(2) << "TensorView created with " << size_in_bytes() << " bytes of data";
}

void *TensorView::data() { return data_; }

const void *TensorView::data() const { return data_; }

const std::vector<size_t> &TensorView::shape() const { return meta_.shape; }

const std::vector<size_t> &TensorView::strides() const { return meta_.strides; }

DataType TensorView::data_type() const { return meta_.data_type; }

MemoryLocation TensorView::memory_location() const {
  return meta_.memory_location;
}

const TensorMeta &TensorView::meta() const { return meta_; }

size_t TensorView::size() const { return meta_.size(); }

size_t TensorView::size_in_bytes() const { return meta_.size_in_bytes(); }

} // namespace dh