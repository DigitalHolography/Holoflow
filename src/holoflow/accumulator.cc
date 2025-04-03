#pragma once

#include "holoflow/accumulator.hh"

#include <cassert>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "holoflow/holoflow.hh"

namespace dh {

// ==========================================================================
//                     AccumulatorMeta Implementation
// ==========================================================================

AccumulatorMeta::AccumulatorMeta(const TensorMeta &imeta,
                                 const TensorMeta &ometa)
    : imeta_(imeta), ometa_(ometa) {
  dh::holoflow_logger()->trace(
      "Initializing AccumulatorMeta with input shape [{}], output shape [{}]",
      fmt::join(imeta_.shape(), ", "), fmt::join(ometa_.shape(), ", "));

  DH_CHECK(imeta_.memory_location() == ometa_.memory_location() &&
           "Input and output tensors must have the same memory location");
  DH_CHECK(imeta_.data_type() == ometa_.data_type() &&
           "Input and output tensors must have the same data type");
  DH_CHECK(imeta_.shape().size() >= 2 && ometa_.shape().size() >= 2 &&
           "Input and output tensors must have at least rank 2");
  DH_CHECK(imeta_.shape().size() == ometa_.shape().size() &&
           "Input and output tensors must have the same rank");
  DH_CHECK(imeta_.strides().size() == ometa_.strides().size() &&
           "Input and output tensors must have the same number of strides");

  for (size_t i = 1; i < imeta_.shape().size(); ++i) {
    DH_CHECK(imeta_.shape()[i] == ometa_.shape()[i] &&
             "Input and output tensor shapes must match except for the first "
             "dimension");
    DH_CHECK(imeta_.strides()[i] == ometa_.strides()[i] &&
             "Input and output tensor strides must match except for the first "
             "dimension");
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
    : meta_(meta), stream_(stream) {
  dh::holoflow_logger()->trace(
      "Created Accumulator with stream={} and shape [{}]",
      reinterpret_cast<void *>(stream), fmt::join(meta.imeta().shape(), ", "));
}

const AccumulatorMeta &Accumulator::meta() const { return meta_; }
const TensorMeta &Accumulator::imeta() const { return meta_.imeta(); }
const TensorMeta &Accumulator::ometa() const { return meta_.ometa(); }

} // namespace dh