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

#include <nlohmann/json.hpp>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holovibes::tasks::syncs {

/// @brief Settings for the percentile clipping task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "min_pct": 0.1,
///   "max_pct": 99.9,
///   "roi": {
///     "cx": 0.5,
///     "cy": 0.5,
///     "rx": 0.5,
///     "ry": 0.5,
///     "angle": 0.0
///   }
/// }
/// @endcode
struct PctClipSettings {
  /// @brief Elliptical ROI for percentile computation.
  struct Ellipse {
    float cx;    // Center x (0-1).
    float cy;    // Center y (0-1).
    float rx;    // Radius x (0-1).
    float ry;    // Radius y (0-1).
    float angle; // Rotation angle in degrees.
  };

  float   min_pct; // Percentile for minimum value (0-100).
  float   max_pct; // Percentile for maximum value (0-100).
  Ellipse roi;     // Elliptical ROI for percentile computation.
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref PctClipSettings and
/// @ref PctClipSettings::Ellipse.
/// @{
void to_json(nlohmann::json &j, const PctClipSettings::Ellipse &e);
void from_json(const nlohmann::json &j, PctClipSettings::Ellipse &e);
void to_json(nlohmann::json &j, const PctClipSettings &pcs);
void from_json(const nlohmann::json &j, PctClipSettings &pcs);
/// @}

class PctClip : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  PctClip(const PctClipSettings &settings, DevPtr<float> &&d_min, DevPtr<float> &&d_max,
          DevPtr<uint8_t> &&d_roi_mask, size_t sort_tmp_bytes, DevPtr<uint8_t> &&d_sort_tmp,
          size_t select_tmp_bytes, DevPtr<uint8_t> &&d_select_tmp, DevPtr<float> &&d_roi,
          DevPtr<int> &&d_roi_count, HostPtr<int> &&h_roi_count, cudaStream_t stream);

  friend class PctClipFactory;

  PctClipSettings settings_;
  DevPtr<float>   d_min_;
  DevPtr<float>   d_max_;
  DevPtr<uint8_t> d_roi_mask_;
  size_t          sort_tmp_bytes_;
  DevPtr<uint8_t> d_sort_tmp_;
  size_t          select_tmp_bytes_;
  DevPtr<uint8_t> d_select_tmp_;
  DevPtr<float>   d_roi_;
  DevPtr<int>     d_roi_count_;
  HostPtr<int>    h_roi_count_;
  cudaStream_t    stream_;
};

class PctClipFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks::syncs