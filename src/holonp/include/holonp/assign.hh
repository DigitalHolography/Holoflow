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

#include "holonp/slice.hh"

namespace holonp {

struct AssignSettings {
  // Must have length == write tensor ndim.
  std::vector<SliceItem> slices;
};

void to_json(nlohmann::json &j, const AssignSettings &s);
void from_json(const nlohmann::json &j, AssignSettings &s);

class Assign : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Assign(const AssignSettings &settings, cudaStream_t stream, size_t ndim,
              size_t total_out, HostPtr<std::int64_t> h_dst_strides,
              DevPtr<std::int64_t> d_dst_strides, HostPtr<std::int64_t> h_start,
              DevPtr<std::int64_t> d_start, HostPtr<std::int64_t> h_step,
              DevPtr<std::int64_t> d_step, HostPtr<std::int64_t> h_slice_shape,
              DevPtr<std::int64_t> d_slice_shape);

  friend class AssignFactory;

  AssignSettings settings_;
  cudaStream_t        stream_;

  size_t ndim_;
  size_t total_out_;

  HostPtr<std::int64_t> h_dst_strides_;
  DevPtr<std::int64_t>  d_dst_strides_;

  HostPtr<std::int64_t> h_start_;
  DevPtr<std::int64_t>  d_start_;

  HostPtr<std::int64_t> h_step_;
  DevPtr<std::int64_t>  d_step_;

  HostPtr<std::int64_t> h_slice_shape_;
  DevPtr<std::int64_t>  d_slice_shape_;
};

class AssignFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
