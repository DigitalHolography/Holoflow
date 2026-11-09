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
#include "holocam/camera.hh"
#include <nlohmann/json.hpp>
#include <string>

#ifdef HOLOCAM_HAS_EGRABBER

#include "curaii/cuda.hh"
#include "grabber/grabber.hh"
#include <EGrabber.h>

namespace holocam {

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;
class CameraBase::Impl {
public:
  Impl(const char *model_name, const nlohmann::json &config, size_t expected_grabber_count,
       size_t buffer_size);

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

  const nlohmann::json &configuration() const noexcept { return config_; }

  const CameraMetrics &metrics() const noexcept { return metrics_; }

  // forwarded to the grabber
  template <typename Module, typename T> void set_required(const char *name, const T &value);
  template <typename Module, typename T> void set_opt(const char *name, const T &value);

private:
  nlohmann::json config_;
  CameraMetrics  metrics_{};

  std::unique_ptr<Euresys::EGenTL> gentl_;
  HostPtr<uint8_t>                 buffers_;
  std::unique_ptr<Grabber>         grabber_;
};
} // namespace holocam
#else
class CameraBase::Impl {};
#endif