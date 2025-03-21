#include "holoflow/model.hh"

#include <glog/logging.h>
#include <optional>
#include <stack>

#include "curaii/curaii.hh"

namespace dh {

Model::TensorSlot::TensorSlot(TensorMeta meta, unique_host_ptr<uint8_t> h,
                              unique_device_ptr<uint8_t> d)
    : meta(meta), host_data(std::move(h)), device_data(std::move(d)) {}

// ==========================================================================
//                     Model Implementation
// ==========================================================================

tl::expected<Model, Error> Model::from_descriptor(const ModelDescriptor &) {
  LOG(FATAL) << "NOT IMPLEMENTED!";
  std::exit(EXIT_FAILURE);
}

} // namespace dh