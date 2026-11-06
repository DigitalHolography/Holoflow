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
#include <cstddef>
#include <cstdint>
#include <memory>

namespace holocam {

struct RecordingSettings {
  std::string output_directory;
  std::string base_filename;
};

struct CameraMetrics {};

// TODO add factory
// factory throw error if egrabber is not detected
// No virtual camera
// ICamera -> CameraBase
// CameraBase -> S710, S711, S991, S980
class CameraBase {
public:
  virtual ~CameraBase() = default;

  // returns string literals for the camera name and short name
  // ex name: AmetekS710EuresysCoaxlinkOcto
  //    short: s710
  //    short name correspond to the config root node
  virtual const char *name() const noexcept       = 0;
  virtual const char *short_name() const noexcept = 0;

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

protected:
  CameraBase(const nlohmann::json &config);

  // apply configuration to the camera
  // constructor helper
  virtual void configure() = 0;

protected:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
};

} // namespace holocam

#else
namespace holocam {
struct RecordingSettings {
  std::string output_directory;
  std::string base_filename;
};

struct CameraMetrics {};

class CameraBase {
public:
  // returns string literals for the camera name and short name
  // ex name: AmetekS710EuresysCoaxlinkOcto
  //    short: s710
  //    short name correspond to the config root node
  virtual const char *name() const noexcept       = 0;
  virtual const char *short_name() const noexcept = 0;

  std::size_t bank_count() const noexcept { return 0; }
  std::size_t frame_width() const noexcept { return 0; }
  std::size_t frame_height() const noexcept { return 0; }
  std::size_t buffer_part_count() const noexcept { return 0; }

  void set_recording_settings(std::optional<RecordingSettings> s) { (void)s; }

  bool start_acquisition() { return false; }
  void stop_acquisition() {}
  bool start_raw_record() { return false; }
  void stop_raw_record() {}

  // returns nullptr if cancelled
  // returns the next frame in the buffer queue and set it as pending frame
  // release pending frame if any before replacing it
  std::byte *get_frame(std::atomic<bool> *cancelled) {
    (void)cancelled;
    return nullptr;
  }

  // release pending frame
  void release_frame() {}

  const nlohmann::json &configuration() const noexcept { return configuration_; };

  const CameraMetrics &metrics() const noexcept { return metrics_; }

protected:
  CameraBase(nlohmann::json config) : configuration_(config) {
    throw std::runtime_error("This build does not support egrabber");
  }

  // apply configuration to the camera
  // constructor helper
  void configure() {}

protected:
  nlohmann::json configuration_;
  CameraMetrics  metrics_{};
};
} // namespace holocam
#endif