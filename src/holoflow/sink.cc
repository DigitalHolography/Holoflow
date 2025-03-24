#include "holoflow/sink.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     SinkMeta Implementation
// ==========================================================================

SinkMeta::SinkMeta(const TensorMeta &imeta) : imeta_(imeta) {}

const TensorMeta &SinkMeta::imeta() const { return imeta_; }

std::ostream &operator<<(std::ostream &os, const SinkMeta &meta) {
  os << "SinkMeta(input=" << meta.imeta() << ")";
  return os;
}

// ==========================================================================
//                     Sink Implementation
// ==========================================================================

Sink::Sink(const SinkMeta &meta, cudaStream_t stream)
    : meta_(meta), stream_(stream) {}

const SinkMeta &Sink::meta() const { return meta_; }

const TensorMeta &Sink::imeta() const { return meta_.imeta(); }

} // namespace dh