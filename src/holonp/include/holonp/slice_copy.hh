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
//
#pragma once

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holonp {

struct SliceItem {
  // NumPy slice semantics (basic slicing only).
  // start/stop can be null (defaults applied during normalization).
  // step must be > 0 for now.
  std::optional<std::int64_t> start = std::nullopt;
  std::optional<std::int64_t> stop  = std::nullopt;
  std::int64_t                step  = 1;
};

struct SliceCopySettings {
  // Must have length == input.ndim.
  std::vector<SliceItem> slices;
};

void to_json(nlohmann::json &j, const SliceItem &s);
void from_json(const nlohmann::json &j, SliceItem &s);

void to_json(nlohmann::json &j, const SliceCopySettings &s);
void from_json(const nlohmann::json &j, SliceCopySettings &s);

class SliceCopy : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  SliceCopy(const SliceCopySettings &settings, cudaStream_t stream, size_t ndim, size_t total_out,
            HostPtr<std::int64_t> h_in_shape, DevPtr<std::int64_t> d_in_shape,
            HostPtr<std::int64_t> h_in_strides, DevPtr<std::int64_t> d_in_strides,
            HostPtr<std::int64_t> h_start, DevPtr<std::int64_t> d_start,
            HostPtr<std::int64_t> h_step, DevPtr<std::int64_t> d_step,
            HostPtr<std::int64_t> h_out_shape, DevPtr<std::int64_t> d_out_shape);

  friend class SliceCopyFactory;

  SliceCopySettings settings_;
  cudaStream_t      stream_;

  size_t ndim_;
  size_t total_out_;

  HostPtr<std::int64_t> h_in_shape_;
  DevPtr<std::int64_t>  d_in_shape_;

  HostPtr<std::int64_t> h_in_strides_;
  DevPtr<std::int64_t>  d_in_strides_;

  HostPtr<std::int64_t> h_start_;
  DevPtr<std::int64_t>  d_start_;

  HostPtr<std::int64_t> h_step_;
  DevPtr<std::int64_t>  d_step_;

  HostPtr<std::int64_t> h_out_shape_;
  DevPtr<std::int64_t>  d_out_shape_;
};

class SliceCopyFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp