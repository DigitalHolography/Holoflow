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

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holonp {

struct FFTShiftSettings {
  // NumPy-like axes selection (empty => all axes).
  std::vector<int> axes;

  bool operator==(const FFTShiftSettings &other) const { return axes == other.axes; }
};

void to_json(nlohmann::json &j, const FFTShiftSettings &s);
void from_json(const nlohmann::json &j, FFTShiftSettings &s);

class FFTShift : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const FFTShiftSettings      &get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  FFTShift(const FFTShiftSettings &settings, const holoflow::core::TDesc &idesc,
           cudaStream_t stream, size_t ndim, size_t total_elems, HostPtr<std::int64_t> h_shape,
           DevPtr<std::int64_t> d_shape, HostPtr<std::int64_t> h_strides,
           DevPtr<std::int64_t> d_strides, HostPtr<std::int64_t> h_shifts,
           DevPtr<std::int64_t> d_shifts);

  friend class FFTShiftFactory;

  FFTShiftSettings      settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                ndim_;
  size_t                total_elems_;

  HostPtr<std::int64_t> h_shape_;
  DevPtr<std::int64_t>  d_shape_;
  HostPtr<std::int64_t> h_strides_;
  DevPtr<std::int64_t>  d_strides_;
  HostPtr<std::int64_t> h_shifts_;
  DevPtr<std::int64_t>  d_shifts_;
};

class FFTShiftFactory : public holoflow::core::ISyncTaskFactory {
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