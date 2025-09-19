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
/// - **Settings:** @ref holovibes::tasks::HolofileSettings
/// - **Failure modes:** Propagates I/O errors, JSON errors, CUDA transfer errors.

#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "curaii/cuda.hh"
#include "holofile/holofile.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holovibes::tasks {

/// @brief Settings for the HoloFile reader task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "path": "string",
///   "load_kind": "Live|CPUCached|GPUCached",
///   "start_frame": 0,
///   "end_frame": 1024,
///   "batch_size": 8
/// }
/// @endcode
struct HolofileSettings {
  /// @brief Loading strategy.
  enum class LoadKind {
    Live,      ///< Read on demand from disk.
    CPUCached, ///< Preload all frames into CPU memory.
    GPUCached  ///< Preload all frames into GPU memory.
  };

  /// Path to the HoloFile.
  std::string path;

  /// Loading strategy.
  /// See @ref LoadKind.
  LoadKind load_kind;

  /// First frame to read (inclusive).
  /// Must satisfy `0 <= start_frame < end_frame`.
  int start_frame;

  /// Last frame to read (exclusive).
  /// Must satisfy `start_frame < end_frame <= total_frames`.
  int end_frame;

  /// Number of frames per output tensor.
  /// Must satisfy `batch_size > 0 <= (end_frame - start_frame)`.
  int batch_size;
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref HolofileSettings and @ref HolofileSettings::LoadKind.
/// @{
void to_json(nlohmann::json &j, const HolofileSettings::LoadKind &lk);
void from_json(const nlohmann::json &j, HolofileSettings::LoadKind &lk);
void to_json(nlohmann::json &j, const HolofileSettings &hs);
void from_json(const nlohmann::json &j, HolofileSettings &hs);
/// @}

/// @brief Synchronous task that reads frames from a HoloFile into a tensor.
/// @details
/// Output tensor layout: `[batch_size, height, width]`
/// with element type chosen from file bit depth.
/// On each `execute`, advances the internal frame cursor by `batch_size`.
///
/// @par CUDA
/// Uses the stream provided by the runtime in @ref SyncCtx to perform H2D/D2D copies
/// for `GPUCached` and for upload of host batches. The stream is not owned.
///
/// @par Errors
/// - `holofile::Exception` and derived types on I/O issues
/// - CUDA runtime errors from transfer operations
class Holofile : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Holofile(const HolofileSettings &settings, holofile::Reader &&reader,
           const holofile::Header &header, int frame_idx, std::byte *buf,
           HostPtr<std::byte> &&h_buf, DevPtr<std::byte> &&d_buf, cudaStream_t stream);

  friend class HolofileFactory;

  HolofileSettings settings_; //< Settings.

  holofile::Reader reader_;    //< Reader.
  holofile::Header header_;    //< Header.
  int              frame_idx_; //< Next frame to read.

  std::byte         *buf_;   // Non-owning view of the active buffer.
  HostPtr<std::byte> h_buf_; // Owned CPU buffer (if any).
  DevPtr<std::byte>  d_buf_; // Owned GPU buffer (if any).

  cudaStream_t stream_; // Stream for GPU transfers.
};

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

} // namespace holovibes::tasks