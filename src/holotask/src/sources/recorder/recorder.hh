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

#include <stdint.h>
#include <string>

#include <nlohmann/json.hpp>

#include "holofile/holofile.hh"

#include "sources/buffer_queue/buffer_queue.hh"

#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"

namespace holotask::sources {

class RecordingSession {
public:
  struct Result {
    bool        completed;
    bool        discard_file;
    std::string failure;
  };

  explicit RecordingSession(size_t to_write);

  std::stop_token token() const;
  void            request_stop();

  // Failure and completion claim the same boundary, so only one can win.
  bool fail(const std::string &message);

  Result finish(std::string error = {});

  void register_write(size_t bytes);

  size_t get_remaining_to_write() const;

private:
  mutable std::mutex mutex_;
  std::stop_source   stop_;
  bool               accepting_failure_ = true;
  std::string        failure_;
  size_t             to_write_;
  size_t             written_ = 0;
};

// read B
class Recorder {
public:
  using Clock = std::chrono::steady_clock;

  struct RecordingGeometry {
    uint8_t bits_per_pixel;
    size_t  frame_width;
    size_t  frame_height;
  };

  Recorder(const std::string &file_path, uint32_t frame_count, size_t buffer_part_count,
           CameraBufferQueue &queue, const RecordingGeometry &g,
           const nlohmann::json &pipeline_settings, RecordingSession &session);

  ~Recorder();

  size_t execute(std::stop_token cancelled, std::function<void(size_t)> poll_metrics);

  static void worker(const holotask::sources::RecordSettings &settings,
                     const Recorder::RecordingGeometry &g, size_t buffer_part_count,
                     CameraBufferQueue &queue, RecordingSession &session,
                     std::atomic<bool>                       &recording,
                     std::function<void(size_t)>              poll_metrics_callback,
                     std::function<void(size_t)>              finished_callback,
                     std::function<void(const std::string &)> failed_callback);

private:
  static std::string remove_incomplete_camera_recording(const std::string &path, std::string failure);

private:
  holofile::Writer   writer_;
  size_t             frame_to_record_;
  size_t             batch_;
  size_t             buffer_part_count_;
  size_t             current_frame_;
  CameraBufferQueue &queue_;
  RecordingSession  &session_;
};

} // namespace holotask::sources