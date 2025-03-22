#include "holoflow/source.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     SourceMeta Implementation
// ==========================================================================

SourceMeta::SourceMeta(const TensorMeta &ometa) : ometa_(ometa) {}

const TensorMeta &SourceMeta::ometa() const { return ometa_; }

std::ostream &operator<<(std::ostream &os, const SourceMeta &meta) {
  os << "SourceMeta(output=" << meta.ometa() << ")";
  return os;
}

// ==========================================================================
//                     Source Implementation
// ==========================================================================

Source::Source(const SourceMeta &meta, cudaStream_t stream)
    : meta_(meta), stream_(stream) {}

const SourceMeta &Source::meta() const { return meta_; }

const TensorMeta &Source::ometa() const { return meta_.ometa(); }

} // namespace dh