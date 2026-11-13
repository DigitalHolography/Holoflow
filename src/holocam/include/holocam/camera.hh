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
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>

#include "record_settings.hh"

namespace holocam {

class CameraBase {
public:
  class Impl;
  virtual ~CameraBase();

  // returns string literals for the camera name and short name
  // ex name: AmetekS710EuresysCoaxlinkOcto
  //    short: s710
  //    model name: Phantom S710
  //    short name correspond to the config root node
  virtual const char *name() const noexcept       = 0;
  virtual const char *short_name() const noexcept = 0;
  virtual const char *model_name() const noexcept = 0;

  virtual size_t grabber_count() const noexcept = 0;

  void set_record_options(std::optional<RecordOptions> o);
  void update_record_path(const std::string &path);

  void start_acquisition();
  void stop_acquisition();
  bool start_raw_record();
  void stop_raw_record();
  void cancel_all_records();

  RecordingSessionId current_recording_session_id();
  bool               recording();

  // returns nullptr if cancelled
  // returns the next frame in the buffer queue and set it as pending frame
  // release pending frame if any before replacing it
  std::byte *get_frame(std::atomic<bool> *cancelled);
  // release pending frame
  void release_frame();

  const nlohmann::json &configuration() const noexcept;

protected:
  using ConfigurationCallback = std::function<void(Impl*, const nlohmann::json& cfg)>;
  CameraBase(const nlohmann::json &config, std::optional<RecordOptions> record_options,
             const char *short_name, const char *model_name, size_t grabber_count,
            ConfigurationCallback configure);

protected:
  std::unique_ptr<Impl> impl_;
};

} // namespace holocam