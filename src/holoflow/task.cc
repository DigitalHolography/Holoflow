#include "holoflow/task.hh"

#include <cassert>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "holoflow/holoflow.hh"

namespace dh {

// ==========================================================================
//                     TaskMeta Implementation
// ==========================================================================

TaskMeta::TaskMeta(const TensorMeta &imeta, const TensorMeta &ometa,
                   bool inlined)
    : imeta_(imeta), ometa_(ometa), inlined_(inlined) {
  dh::holoflow_logger()->trace("Initializing TaskMeta with input shape [{}], "
                               "output shape [{}], inlined []",
                               fmt::join(imeta_.shape(), ", "),
                               fmt::join(ometa_.shape(), ", "), inlined);

  if (inlined_) {
    DH_CHECK((imeta_.data_type() == ometa_.data_type()) &&
             "Inlined tasks must have equal input and output tensors");
    DH_CHECK((imeta_.memory_location() == ometa_.memory_location()) &&
             "Inlined tasks must have equal input and output tensors");
    DH_CHECK((imeta_.shape() == ometa_.shape()) &&
             "Inlined tasks must have equal input and output tensors");
    DH_CHECK((imeta_.strides() == ometa_.strides()) &&
             "Inlined tasks must have equal input and output tensors");
  }
}

const TensorMeta &TaskMeta::imeta() const { return imeta_; }
const TensorMeta &TaskMeta::ometa() const { return ometa_; }
bool TaskMeta::inlined() const { return inlined_; }

// ==========================================================================
//                     Task Implementation
// ==========================================================================

Task::Task(const TaskMeta &meta, CudaStreamRef stream)
    : meta_(meta), stream_(stream) {}

const TaskMeta &Task::meta() const { return meta_; }
const TensorMeta &Task::imeta() const { return meta_.imeta(); }
const TensorMeta &Task::ometa() const { return meta_.ometa(); }
bool Task::inlined() const { return meta_.inlined(); }

} // namespace dh