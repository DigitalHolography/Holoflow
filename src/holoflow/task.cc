#include "holoflow/task.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     TaskMeta Implementation
// ==========================================================================

TaskMeta::TaskMeta(const TensorMeta &imeta, const TensorMeta &ometa,
                   bool inlined)
    : imeta_(imeta), ometa_(ometa), inlined_(inlined) {
  if (inlined_) {
    CHECK(imeta_.data_type() == ometa_.data_type())
        << "Inlined tasks must have equal input and output tensors";
    CHECK(imeta_.memory_location() == ometa_.memory_location())
        << "Inlined tasks must have equal input and output tensors";
    CHECK(imeta_.shape() == ometa_.shape())
        << "Inlined tasks must have equal input and output tensors";
    CHECK(imeta_.strides() == ometa_.strides())
        << "Inlined tasks must have equal input and output tensors";
  }
}

const TensorMeta &TaskMeta::imeta() const { return imeta_; }
const TensorMeta &TaskMeta::ometa() const { return ometa_; }
bool TaskMeta::inlined() const { return inlined_; }

std::ostream &operator<<(std::ostream &os, const TaskMeta &meta) {
  os << "TaskMeta(input=" << meta.imeta() << ", output=" << meta.ometa()
     << ", inlined=" << (meta.inlined() ? "true" : "false") << ")";
  return os;
}

// ==========================================================================
//                     Task Implementation
// ==========================================================================

Task::Task(const TaskMeta &meta, cudaStream_t stream)
    : meta_(meta), stream_(stream) {}

const TaskMeta &Task::meta() const { return meta_; }
const TensorMeta &Task::imeta() const { return meta_.imeta(); }
const TensorMeta &Task::ometa() const { return meta_.ometa(); }
bool Task::inlined() const { return meta_.inlined(); }

} // namespace dh