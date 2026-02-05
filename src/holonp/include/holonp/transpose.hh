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
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holonp {

struct TransposeSettings {
  // Permutation of axes. Must have length == input.ndim.
  // Negative axes are allowed (e.g. -1).
  std::vector<int> axes;
};

void to_json(nlohmann::json &j, const TransposeSettings &s);
void from_json(const nlohmann::json &j, TransposeSettings &s);

class Transpose : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Transpose(const TransposeSettings &settings, cudaStream_t stream, size_t ndim, size_t total_elems,
            DevPtr<std::int64_t> d_in_strides, DevPtr<std::int64_t> d_out_strides,
            DevPtr<std::int64_t> d_out_shape);

  friend class TransposeFactory;

  TransposeSettings settings_;
  cudaStream_t      stream_;

  size_t ndim_;
  size_t total_elems_;

  // Device buffers for kernel configuration
  // We store strides/shape to avoid expensive index re-calculation
  // from scratch if possible, though for general N-D transpose,
  // index reconstruction is often necessary.
  DevPtr<std::int64_t> d_in_strides_;  // Input strides (permuted)
  DevPtr<std::int64_t> d_out_strides_; // Output strides (contiguous)
  DevPtr<std::int64_t> d_out_shape_;   // Output shape
};

class TransposeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp