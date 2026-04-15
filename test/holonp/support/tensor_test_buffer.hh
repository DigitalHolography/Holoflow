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

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tensor.hh"

namespace holonp_test {

// ---- TensorTestBuffer: managed backing store for test tensors -----------------------------------
//
// Allocates (desc.offset + desc.num_bytes()) bytes in the tensor's declared memory location,
// so offset views have valid backing storage without clobbering adjacent bytes.
//
// The returned TView::data() lands at storage.ptr + desc.offset, matching holoflow semantics.
//
class TensorTestBuffer {
public:
  explicit TensorTestBuffer(const holoflow::core::TDesc &desc);

  // Upload `data` (exactly desc.num_bytes() bytes) to the logical region at desc.offset.
  void upload(std::span<const std::byte> data);

  // Download the logical region to a dense host buffer.
  // Returns exactly desc.num_bytes() bytes.
  [[nodiscard]] std::vector<std::byte> download() const;

  [[nodiscard]] holoflow::core::TView        view() noexcept;
  [[nodiscard]] const holoflow::core::TDesc &desc() const noexcept { return desc_; }

private:
  holoflow::core::TDesc                desc_;
  size_t                               storage_bytes_;
  curaii::unique_device_ptr<std::byte> d_ptr_;
  curaii::unique_host_ptr<std::byte>   h_ptr_;
  std::byte                           *raw_ptr_ = nullptr;
  holoflow::core::Storage              storage_;
};

} // namespace holonp_test
