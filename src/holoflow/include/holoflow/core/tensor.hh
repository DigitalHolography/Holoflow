// Copyright 2025 Digital Holography Foundation
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

#include <cstdint>
#include <string_view>
#include <vector>

#include "curaii/cuda.hh"

namespace holoflow::core {

/// Supported tensor element types.
enum class DType : uint8_t {
  U8,   ///< Unsigned 8-bit integer
  U16,  ///< Unsigned 16-bit integer
  F32,  ///< 32-bit float
  CF32, ///< 32-bit complex float
};

/// Returns the size in bytes of a given DType.
[[nodiscard]] size_t size_of(DType dtype) noexcept;

/// Returns a human-readable name for the DType.
[[nodiscard]] std::string_view to_string(DType dtype) noexcept;

/// Where memory lives: host (CPU) or device (GPU).
enum class MemLoc : uint8_t {
  Host,   ///< CPU-accessible memory
  Device, ///< GPU device memory
};

/// Returns a human-readable name for the MemLoc.
[[nodiscard]] std::string_view to_string(MemLoc loc) noexcept;

/// Describes a multi-dimensional array (tensor).
struct TDesc {
  std::vector<size_t> shape;   ///< The shape of the tensor (dimensions)
  DType               dtype;   ///< The data type of the tensor elements
  MemLoc              mem_loc; ///< The memory location of the tensor

  /// Returns the rank (number of dimensions) of the tensor.
  size_t rank() const noexcept;

  /// Returns the total number of elements in the tensor.
  size_t num_elements() const;

  /// Returns the total size in bytes of the tensor data.
  size_t num_bytes() const;
};

/// A non-owning view into tensor data.
struct TView {
  std::byte *data; ///< Pointer to the tensor data
  TDesc      desc; ///< Description of the tensor
};

/// A view into constant tensor data.
struct CTView {
  const std::byte *data; ///< Pointer to the tensor data
  const TDesc      desc; ///< Description of the tensor
};

/// A multi-dimensional array (tensor) holding data in either host or device
/// memory.
class Tensor {
public:
  /// Constructs a Tensor with the given descriptor. Allocates memory
  /// according to the descriptor's memory location.
  explicit Tensor(const TDesc &desc);

  /// Returns a mutable pointer to the tensor data.
  [[nodiscard]] void *data() noexcept;

  /// Returns a constant pointer to the tensor data.
  [[nodiscard]] const void *data() const noexcept;

  /// Returns the tensor descriptor.
  [[nodiscard]] const TDesc &desc() const noexcept;

  /// Returns a non-owning mutable view into the tensor data.
  [[nodiscard]] TView view() noexcept;

  /// Returns a non-owning constant view into the tensor data.
  [[nodiscard]] CTView cview() const noexcept;

private:
  using HData = curaii::unique_host_ptr<std::byte>;
  using DData = curaii::unique_device_ptr<std::byte>;

  TDesc      desc_;   /// Descriptor of the tensor
  HData      h_data_; ///< Host memory (if applicable)
  DData      d_data_; ///< Device memory (if applicable)
  std::byte *data_;   ///< Raw pointer to the tensor data
};

} // namespace holoflow::core