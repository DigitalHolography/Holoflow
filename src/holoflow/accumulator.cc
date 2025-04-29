#pragma once

#include "holoflow/accumulator.hh"

#include "bug_buster/bug_buster.hh"
#include "holoflow/holoflow.hh"

namespace dh {

// ==========================================================================
//                     AccumulatorMeta Implementation
// ==========================================================================

AccumulatorMeta::AccumulatorMeta(const TensorMeta &imeta,
                                 const TensorMeta &ometa)
    : imeta_(imeta), ometa_(ometa) {
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

// ==========================================================================
//                     Accumulator Implementation
// ==========================================================================

Accumulator::Accumulator(const AccumulatorMeta &meta, cudaStream_t stream,
                         EventListeners event_listeners)
    : meta_(meta), stream_(stream), event_listeners_(event_listeners) {}

void Accumulator::handle_event(const json &) {
  throw std::runtime_error("Not implemented");
}

void Accumulator::emit_event(const nlohmann::json &event) {
  for (auto &listener : event_listeners_) {
    listener(event);
  }
}

const AccumulatorMeta &Accumulator::meta() const { return meta_; }
const TensorMeta &Accumulator::imeta() const { return meta_.imeta(); }
const TensorMeta &Accumulator::ometa() const { return meta_.ometa(); }

} // namespace dh