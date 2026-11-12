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

#include "buffer_queue/buffer_queue.hh"

#include "record/recorder.hh"

namespace holocam {

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;
class CameraBase::Impl {
public:
  Impl(const char *model_name, const char *short_name, const nlohmann::json &config,
       const std::optional<RecordOptions> &record_options, size_t expected_grabber_count,
       size_t buffer_size);

  std::size_t bank_count() const noexcept;
  std::size_t frame_width() const noexcept;
  std::size_t frame_height() const noexcept;
  std::size_t buffer_part_count() const noexcept;

  void set_recording_options(std::optional<RecordOptions>);
  void update_record_path(const std::string &path);

  void start_acquisition();
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

  // forwarded to the grabber
  template <typename Module, typename T> void set_required(const char *name, const T &value);
  template <typename Module, typename T> void set_opt(const char *name, const T &value);

  void reset_grabber_config() { grabber_->reset_config(); }
  void set_offset(int64_t a, int64_t b) { grabber_->set_offsets(a, b); }
  const Euresys::EGrabberCameraInfo info() const noexcept { return info_; }

  size_t find_grabber_index_for_bank(size_t id);

private:
  void init_record_context(const char *short_name, const RecordOptions &options,
                           size_t buffer_size);

  void acquisition_loop(std::stop_token cancelled);

  void cancel_all_records(); 
  void check_for_record_failure(); 

private:
  static constexpr uint64_t pop_timeout_ = 100;
  nlohmann::json            config_;

  std::unique_ptr<Euresys::EGenTL> gentl_;
  Euresys::EGrabberCameraInfo      info_;
  HostPtr<uint8_t>                 buffers_;
  size_t                           buffers_size_;
  size_t                           buffer_part_count_;
  size_t                           nb_buffers_;
  std::unique_ptr<Grabber>         grabber_{nullptr};

  // Buffer Queue
  std::unique_ptr<CameraBufferQueue> buffer_queue_{nullptr};

  std::optional<std::jthread> acquisition_thread_{std::nullopt};
  std::atomic<bool>           acquiring_{false};

  // Record
  std::atomic<bool> recording_{false};

  std::optional<RecordingContext> recording_context_{std::nullopt};
  HostPtr<std::byte>              recording_buffers_         = nullptr;
  size_t                          recording_buffers_size_    = 0;
  size_t                          recording_buffers_count_   = 0;
  RecordingSession               *current_recording_session_ = nullptr;
};

} // namespace holocam

#include "camera_pimpl.hxx"
#else
namespace holocam {
class CameraBase::Impl {};
} // namespace holocam
#endif