#include "holoflow/accumulator.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     AccumulatorMeta Implementation
// ==========================================================================

AccumulatorMeta::AccumulatorMeta(const TensorMeta &imeta,
                                 const TensorMeta &ometa)
    : imeta_(imeta), ometa_(ometa) {
  CHECK(imeta_.memory_location() == ometa_.memory_location())
      << "Input and output tensors must have the same memory location";
  CHECK(imeta_.data_type() == ometa_.data_type())
      << "Input and output tensors must have the same data type";
  CHECK(imeta_.shape().size() >= 2 && ometa_.shape().size() >= 2)
      << "Input and output tensors must have at least rank 2";
  CHECK(imeta_.shape().size() == ometa_.shape().size())
      << "Input and output tensors must have the same rank";
  CHECK(imeta_.strides().size() == ometa_.strides().size())
      << "Input and output tensors must have the same number of strides";
  for (size_t i = 1; i < imeta_.shape().size(); ++i) {
    CHECK(imeta_.shape()[i] == ometa_.shape()[i])
        << "Input and output tensor shapes must match except for the first "
           "dimension";
    CHECK(imeta_.strides()[i] == ometa_.strides()[i])
        << "Input and output tensor strides must match except for the first "
           "dimension";
  }
}

const TensorMeta &AccumulatorMeta::imeta() const { return imeta_; }
const TensorMeta &AccumulatorMeta::ometa() const { return ometa_; }

std::ostream &operator<<(std::ostream &os, const AccumulatorMeta &meta) {
  os << "AccumulatorMeta(input=" << meta.imeta() << ", output=" << meta.ometa()
     << ")";
  return os;
}

// ==========================================================================
//                     Accumulator Implementation
// ==========================================================================

Accumulator::Accumulator(const AccumulatorMeta &meta, cudaStream_t stream)
    : meta_(meta), stream_(stream) {}

const AccumulatorMeta &Accumulator::meta() const { return meta_; }
const TensorMeta &Accumulator::imeta() const { return meta_.imeta(); }
const TensorMeta &Accumulator::ometa() const { return meta_.ometa(); }

} // namespace dh
