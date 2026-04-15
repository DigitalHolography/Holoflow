// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensor_test_buffer.hh"

#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace holonp_test {

TensorTestBuffer::TensorTestBuffer(const holoflow::core::TDesc &desc)
    : desc_(desc), storage_bytes_(desc.offset + desc.num_bytes()) {
  if (desc_.mem_loc == holoflow::core::MemLoc::Device) {
    d_ptr_   = curaii::make_unique_device_ptr<std::byte>(storage_bytes_);
    raw_ptr_ = d_ptr_.get();
  } else {
    h_ptr_   = curaii::make_unique_host_ptr<std::byte>(storage_bytes_);
    raw_ptr_ = h_ptr_.get();
  }
  storage_ = holoflow::core::Storage{desc_.mem_loc, storage_bytes_, raw_ptr_};
}

void TensorTestBuffer::upload(std::span<const std::byte> data) {
  if (data.size() != desc_.num_bytes()) {
    throw std::invalid_argument("TensorTestBuffer::upload: data size mismatch");
  }
  if (desc_.mem_loc == holoflow::core::MemLoc::Device) {
    CUDA_CHECK(
        cudaMemcpy(raw_ptr_ + desc_.offset, data.data(), data.size(), cudaMemcpyHostToDevice));
  } else {
    std::memcpy(raw_ptr_ + desc_.offset, data.data(), data.size());
  }
}

std::vector<std::byte> TensorTestBuffer::download() const {
  std::vector<std::byte> out(desc_.num_bytes());
  if (desc_.mem_loc == holoflow::core::MemLoc::Device) {
    CUDA_CHECK(cudaMemcpy(out.data(), raw_ptr_ + desc_.offset, out.size(), cudaMemcpyDeviceToHost));
  } else {
    std::memcpy(out.data(), raw_ptr_ + desc_.offset, out.size());
  }
  return out;
}

holoflow::core::TView TensorTestBuffer::view() noexcept {
  return holoflow::core::TView{desc_, &storage_};
}

} // namespace holonp_test
