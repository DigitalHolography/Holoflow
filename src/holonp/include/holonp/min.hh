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

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

// -----------------------------------------------------------------------------
// Settings
// -----------------------------------------------------------------------------

struct MinSettings {
  std::vector<int> axis; // Empty => reduce all
  bool             keepdims = false;
};

void to_json(nlohmann::json &j, const MinSettings &s);
void from_json(const nlohmann::json &j, MinSettings &s);

// -----------------------------------------------------------------------------
// Task Definition
// -----------------------------------------------------------------------------

class Min : public holoflow::core::ISyncTask {
public:
  Min(const MinSettings &settings, cudaStream_t stream, size_t total_out, size_t total_red,
      int out_ndim, int red_ndim, DevPtr<size_t> in_strides, DevPtr<size_t> out_strides,
      DevPtr<int> out_to_in_map, DevPtr<size_t> red_strides, DevPtr<int> red_axes_map);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  MinSettings  settings_;
  cudaStream_t stream_;

  size_t total_out_;
  size_t total_red_;
  int    out_ndim_;
  int    red_ndim_;

  // Device-resident tensors for address calculation
  DevPtr<size_t> d_in_strides_;
  DevPtr<size_t> d_out_strides_;
  DevPtr<int>    d_out_to_in_map_;
  DevPtr<size_t> d_red_strides_;
  DevPtr<int>    d_red_axes_map_;

  friend class MinFactory;
};

// -----------------------------------------------------------------------------
// Factory Definition
// -----------------------------------------------------------------------------

class MinFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp