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

struct MeanSettings {
  // NumPy-like axis selection. Empty => all axes.
  std::vector<int> axis;
  bool             keepdims = false;

  bool operator==(const MeanSettings &other) const {
    return axis == other.axis && keepdims == other.keepdims;
  }
};

void to_json(nlohmann::json &j, const MeanSettings &s);
void from_json(const nlohmann::json &j, MeanSettings &s);

class Mean : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc& get_idesc() const { return idesc_; }
  const MeanSettings&          get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  Mean(const MeanSettings &settings, const holoflow::core::TDesc &idesc, cudaStream_t stream, 
       size_t out_ndim, size_t red_ndim,
       std::int64_t total_out, std::int64_t total_red, HostPtr<std::int64_t> h_in_strides,
       DevPtr<std::int64_t> d_in_strides, HostPtr<std::int64_t> h_out_strides,
       DevPtr<std::int64_t> d_out_strides, HostPtr<int> h_out_to_in, DevPtr<int> d_out_to_in,
       HostPtr<int> h_red_axes, DevPtr<int> d_red_axes, HostPtr<std::int64_t> h_red_strides,
       DevPtr<std::int64_t> d_red_strides);
  friend class MeanFactory;

  MeanSettings          settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;

  size_t       out_ndim_;
  size_t       red_ndim_;
  std::int64_t total_out_;
  std::int64_t total_red_;

  HostPtr<std::int64_t> h_in_strides_;
  DevPtr<std::int64_t>  d_in_strides_;
  HostPtr<std::int64_t> h_out_strides_;
  DevPtr<std::int64_t>  d_out_strides_;
  HostPtr<int>          h_out_to_in_;
  DevPtr<int>           d_out_to_in_;
  HostPtr<int>          h_red_axes_;
  DevPtr<int>           d_red_axes_;
  HostPtr<std::int64_t> h_red_strides_;
  DevPtr<std::int64_t>  d_red_strides_;
};

class MeanFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc>     input_descs,
         const nlohmann::json                       &jsettings,
         const holoflow::core::SyncCreateCtx        &ctx) const override;
};

} // namespace holonp