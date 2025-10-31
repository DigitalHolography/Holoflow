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

#include <cstddef>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>

#include "holofile/holofile.hh"
#include "holoflow/core/tasks.hh"

#include "pipeline/settings.hh"

namespace holovibes::tasks::sinks {

struct HolofileSettings {
  std::string    path;              // Path to the HoloFile.
  int            count;             // Number of frames to write.
  nlohmann::json pipeline_settings; // Pipeline settings to store in footer.
};

void to_json(nlohmann::json &j, const HolofileSettings &hs);
void from_json(const nlohmann::json &j, HolofileSettings &hs);

class Holofile : public holoflow::core::ISyncTask {
public:
  struct RecordingGeometry {
    uint8_t  bits_per_pixel;
    uint32_t frame_width;
    uint32_t frame_height;
  };

  Holofile(const HolofileSettings &settings, RecordingGeometry geometry);

  ~Holofile() override;

  [[nodiscard]] holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  void               handle_events(holoflow::core::SyncCtx &ctx);
  void               start_recording(const std::string &path, int count);
  void               finalize_recording(holoflow::core::SyncCtx &ctx);
  holofile::Header   make_header(int count) const;
  [[nodiscard]] bool is_recording() const noexcept;
  void               reset_recording_state();

  HolofileSettings                settings_;
  RecordingGeometry               geometry_;
  std::optional<holofile::Writer> writer_;
  int                             frames_written_ = 0;
};

class HolofileFactory : public holoflow::core::ISyncTaskFactory {
public:
  [[nodiscard]] holoflow::core::InferResult
  infer(std::span<const holoflow::core::TDesc> input_descs,
        const nlohmann::json                  &jsettings) const override;

  [[nodiscard]] std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks::sinks
