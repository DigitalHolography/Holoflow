#include "holoflow/v3/model/model.hh"

#include "holoflow/holoflow.hh"
#include "holoflow/tensor.hh"

namespace holoflow::model {

TensorSlot::TensorSlot(dh::TensorMeta meta, dh::unique_host_ptr<uint8_t> h,
                       dh::unique_device_ptr<uint8_t> d)
    : meta(meta), host_data(std::move(h)), device_data(std::move(d)),
      data(nullptr) {
  switch (meta.memory_location()) {
  case dh::MemoryLocation::HOST:
    data = host_data.get();
    break;
  case dh::MemoryLocation::DEVICE:
    data = device_data.get();
    break;
  }
}

dh::TensorView TensorSlot::view() { return dh::TensorView(data, meta); }

} // namespace holoflow::model