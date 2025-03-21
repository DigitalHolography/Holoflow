#include "holoflow/tensor.hh"

#include <algorithm>
#include <glog/logging.h>
#include <numeric>
#include <vector>

namespace dh {

// ==========================================================================
//                     MemoryLocation Utilities
// ==========================================================================

std::ostream &operator<<(std::ostream &os, MemoryLocation loc) {
  return os << (loc == MemoryLocation::HOST ? "HOST" : "DEVICE");
}

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

std::ostream &operator<<(std::ostream &os, DataType dt) {
  switch (dt) {
  case DataType::U8:
    return os << "U8";
  case DataType::U16:
    return os << "U16";
  case DataType::F32:
    return os << "F32";
  case DataType::CF32:
    return os << "CF32";
  default:
    LOG(FATAL) << "Unknown DataType";
    std::exit(EXIT_FAILURE);
  }
}

// ==========================================================================
//                     TensorMeta Implementation
// ==========================================================================

TensorMeta::TensorMeta(DataType data_type, MemoryLocation memory_location,
                       std::vector<size_t> shape)
    : data_type_(data_type), memory_location_(memory_location),
      shape_(std::move(shape)) {
  CHECK(!shape_.empty()) << "Tensor shape cannot be empty";
  CHECK(std::all_of(shape_.begin(), shape_.end(), [](size_t val) {
    return val > 0;
  })) << "Tensor shape cannot contain zero";

  strides_.resize(shape_.size());
  size_t stride = 1;
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

TensorMeta::TensorMeta(DataType data_type, MemoryLocation memory_location,
                       std::vector<size_t> shape, std::vector<size_t> strides)
    : data_type_(data_type), memory_location_(memory_location),
      shape_(std::move(shape)), strides_(std::move(strides)) {
  CHECK(!shape_.empty()) << "Tensor shape cannot be empty";
  CHECK(shape_.size() == strides_.size())
      << "Shape and strides must match in size";
  CHECK(std::all_of(strides_.begin(), strides_.end(), [](size_t val) {
    return val > 0;
  })) << "Tensor strides cannot be zero";
}

size_t TensorMeta::size() const {
  return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies());
}

size_t TensorMeta::size_in_bytes() const {
  if (shape_.empty() || strides_.empty()) {
    return 0;
  }
  size_t last_offset = 0;
  for (size_t i = 0; i < shape_.size(); ++i) {
    last_offset += (shape_[i] - 1) * strides_[i];
  }
  return (last_offset + 1) * size_of(data_type_);
}

const std::vector<size_t> &TensorMeta::shape() const { return shape_; }
const std::vector<size_t> &TensorMeta::strides() const { return strides_; }
DataType TensorMeta::data_type() const { return data_type_; }
MemoryLocation TensorMeta::memory_location() const { return memory_location_; }

std::ostream &operator<<(std::ostream &os, const TensorMeta &meta) {
  os << "TensorMeta(dtype=" << meta.data_type_
     << ", location=" << meta.memory_location_ << ", shape={";
  for (size_t i = 0; i < meta.shape_.size(); ++i) {
    os << meta.shape_[i] << (i + 1 < meta.shape_.size() ? ", " : "");
  }
  os << "}, strides={";
  for (size_t i = 0; i < meta.strides_.size(); ++i) {
    os << meta.strides_[i] << (i + 1 < meta.strides_.size() ? ", " : "");
  }
  os << "})";
  return os;
}

// ==========================================================================
//                     TensorView Implementation
// ==========================================================================

TensorView::TensorView(void *data, const TensorMeta &meta)
    : data_(data), meta_(std::ref(meta)) {
  CHECK(data_ != nullptr) << "TensorView received a null data pointer";
  if (reinterpret_cast<uintptr_t>(data_) % alignof(std::max_align_t) != 0) {
    LOG(WARNING) << "TensorView data pointer is not properly aligned";
  }
}

void *TensorView::data() { return data_; }

const void *TensorView::data() const { return data_; }

const std::vector<size_t> &TensorView::shape() const {
  return meta_.get().shape();
}

const std::vector<size_t> &TensorView::strides() const {
  return meta_.get().strides();
}

DataType TensorView::data_type() const { return meta_.get().data_type(); }

MemoryLocation TensorView::memory_location() const {
  return meta_.get().memory_location();
}

const TensorMeta &TensorView::meta() const { return meta_; }

size_t TensorView::size() const { return meta_.get().size(); }

size_t TensorView::size_in_bytes() const { return meta_.get().size_in_bytes(); }

std::ostream &operator<<(std::ostream &os, const TensorView &view) {
  os << "TensorView(data=" << view.data() << ", metadata=" << view.meta()
     << ")";
  return os;
}

} // namespace dh