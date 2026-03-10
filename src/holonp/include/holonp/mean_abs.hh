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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holonp/mean.hh"

namespace holonp {

using MeanAbsSettings = MeanSettings;

class MeanAbs : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc& get_idesc() const { return idesc_; }
  const MeanAbsSettings&       get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  MeanAbs(const MeanAbsSettings &settings, const holoflow::core::TDesc &idesc, cudaStream_t stream, 
          size_t out_ndim, size_t red_ndim,
          std::int64_t total_out, std::int64_t total_red,
          curaii::unique_host_ptr<std::int64_t>   h_in_strides,
          curaii::unique_device_ptr<std::int64_t> d_in_strides,
          curaii::unique_host_ptr<std::int64_t>   h_out_strides,
          curaii::unique_device_ptr<std::int64_t> d_out_strides,
          curaii::unique_host_ptr<int> h_out_to_in, curaii::unique_device_ptr<int> d_out_to_in,
          curaii::unique_host_ptr<int> h_red_axes, curaii::unique_device_ptr<int> d_red_axes,
          curaii::unique_host_ptr<std::int64_t>   h_red_strides,
          curaii::unique_device_ptr<std::int64_t> d_red_strides);
  friend class MeanAbsFactory;

  MeanAbsSettings       settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;

  size_t       out_ndim_;
  size_t       red_ndim_;
  std::int64_t total_out_;
  std::int64_t total_red_;

  curaii::unique_host_ptr<std::int64_t>   h_in_strides_;
  curaii::unique_device_ptr<std::int64_t> d_in_strides_;
  curaii::unique_host_ptr<std::int64_t>   h_out_strides_;
  curaii::unique_device_ptr<std::int64_t> d_out_strides_;
  curaii::unique_host_ptr<int>            h_out_to_in_;
  curaii::unique_device_ptr<int>          d_out_to_in_;
  curaii::unique_host_ptr<int>            h_red_axes_;
  curaii::unique_device_ptr<int>          d_red_axes_;
  curaii::unique_host_ptr<std::int64_t>   h_red_strides_;
  curaii::unique_device_ptr<std::int64_t> d_red_strides_;
};

class MeanAbsFactory : public holoflow::core::ISyncTaskFactory {
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