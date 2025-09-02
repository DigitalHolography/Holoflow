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

namespace holoflow::core {

/// Supported tensor element types.
enum class DType : uint8_t {
  U8,   ///< Unsigned 8-bit integer
  U16,  ///< Unsigned 16-bit integer
  F32,  ///< 32-bit float
  CF32, ///< 32-bit complex float
};

/// Returns the size in bytes of a given DataType.
size_t size_of(DType dtype) noexcept;

/// Returns a human-readable name for the DataType.
const char *to_string(DType dtype) noexcept;

/// Where memory lives: host (CPU) or device (GPU).
enum class MemLoc : uint8_t {
  Host,   ///< CPU-accessible memory
  Device, ///< GPU device memory
};

/// Returns a human-readable name for the MemLoc.
const char *to_string(MemLoc loc) noexcept;

} // namespace holoflow::core