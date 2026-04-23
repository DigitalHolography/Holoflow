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

/// @file
/// @brief Synchronous task that reads batches of frames from a HoloFile.
/// @details
/// - **Inputs:** none
/// - **Outputs:** `[batch_size, height, width]` tensor of `u8` or `u16`
/// - **Settings:** @ref holotask::sources::HolofileSettings
/// - **Failure modes:** Propagates I/O errors, JSON errors, CUDA transfer errors.

#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

/// @brief Settings for the HoloFile reader task.
struct HolofileSettings {
  /// @brief Loading strategy.
  enum class LoadKind {
    Live,      ///< Read on demand from disk.
    CPUCached, ///< Preload all frames into CPU memory.
    GPUCached  ///< Preload all frames into GPU memory.
  };

  std::string path;               ///< Path to the HoloFile.
  LoadKind    load_kind;          ///< Loading strategy.
  int         start_frame;        ///< First frame to read (inclusive).
  int         end_frame;          ///< Last frame to read (exclusive).
  int         batch_size;         ///< Number of frames per output tensor.
  std::optional<int> max_fps;     ///< Optional playback cap in frames per second.
  bool        keep_cursor = true; ///< Maintain cursor across updates.

  bool operator==(const HolofileSettings &) const = default;
};

void to_json(nlohmann::json &j, const HolofileSettings::LoadKind &lk);
void from_json(const nlohmann::json &j, HolofileSettings::LoadKind &lk);
void to_json(nlohmann::json &j, const HolofileSettings &hs);
void from_json(const nlohmann::json &j, HolofileSettings &hs);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

/// @brief Factory for HoloFile reader tasks.
class HolofileFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holotask::sources
