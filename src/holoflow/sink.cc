#include "holoflow/sink.hh"

#include "holoflow/holoflow.hh"

namespace dh {

// ==========================================================================
//                     SinkMeta Implementation
// ==========================================================================

SinkMeta::SinkMeta(const TensorMeta &imeta) : imeta_(imeta) {}

const TensorMeta &SinkMeta::imeta() const { return imeta_; }

// ==========================================================================
//                     Sink Implementation
// ==========================================================================

Sink::Sink(const SinkMeta &meta, cudaStream_t stream)
    : meta_(meta), stream_(stream) {}

void Sink::handle_event(const json &) {
  throw std::runtime_error("Not implemented");
}

const SinkMeta &Sink::meta() const { return meta_; }

const TensorMeta &Sink::imeta() const { return meta_.imeta(); }

} // namespace dh