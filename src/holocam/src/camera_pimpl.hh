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
#include <nlohmann/json.hpp>
#include <string>

#ifdef HOLOCAM_HAS_EGRABBER
namespace holocam {
class CameraBase::Impl {
public:
  CameraBase::Impl(const nlohmann::json &config);

  std::size_t bank_count() const noexcept;
  std::size_t frame_width() const noexcept;
  std::size_t frame_height() const noexcept;
  std::size_t buffer_part_count() const noexcept;

  void set_recording_settings(std::optional<RecordingSettings>);

  bool start_acquisition();
  void stop_acquisition();
  bool start_raw_record();
  void stop_raw_record();

  // returns nullptr if cancelled
  // returns the next frame in the buffer queue and set it as pending frame
  // release pending frame if any before replacing it
  std::byte *get_frame(std::atomic<bool> *cancelled);
  // release pending frame
  void release_frame();

  const nlohmann::json &configuration() const noexcept;

  const CameraMetrics &metrics() const noexcept;

  // TODO add exposition for members needed for the configuration so that each camera can use it
private:
  nlohmann::json config_;
  CameraMetrics  metrics_{};

  // TODO copy S711 stuffs
};
} // namespace holocam
#endif