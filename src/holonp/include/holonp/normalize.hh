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

#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

struct NormalizeSettings {
  // NumPy-like axis selection. Empty => reduce all axes.
  std::vector<int> axes;
  float            lo = 0.0f;
  float            hi = 1.0f;

  bool operator==(const NormalizeSettings &other) const {
    return axes == other.axes && lo == other.lo && hi == other.hi;
  }
};

void to_json(nlohmann::json &j, const NormalizeSettings &s);
void from_json(const nlohmann::json &j, NormalizeSettings &s);

class Normalize : public holoflow::core::ISyncTask {
public:
  Normalize(const NormalizeSettings &settings, const holoflow::core::TDesc &idesc,
            cudaStream_t stream, size_t ndim, size_t group_ndim, size_t red_ndim,
            size_t total_elems, size_t total_groups, size_t total_red, DevPtr<size_t> shape,
            DevPtr<size_t> in_strides, DevPtr<size_t> out_strides, DevPtr<int> group_axes,
            DevPtr<size_t> group_strides, DevPtr<int> red_axes, DevPtr<size_t> red_strides,
            DevPtr<float> group_mins, DevPtr<float> group_maxs);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const NormalizeSettings     &get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  NormalizeSettings     settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;

  size_t ndim_;
  size_t group_ndim_;
  size_t red_ndim_;
  size_t total_elems_;
  size_t total_groups_;
  size_t total_red_;

  DevPtr<size_t> d_shape_;
  DevPtr<size_t> d_in_strides_;
  DevPtr<size_t> d_out_strides_;
  DevPtr<int>    d_group_axes_;
  DevPtr<size_t> d_group_strides_;
  DevPtr<int>    d_red_axes_;
  DevPtr<size_t> d_red_strides_;
  DevPtr<float>  d_group_mins_;
  DevPtr<float>  d_group_maxs_;
};

class NormalizeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp