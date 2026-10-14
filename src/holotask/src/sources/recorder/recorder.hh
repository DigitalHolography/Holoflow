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

#include "sources/buffer_queue/buffer_queue.hh"

namespace holotask::sources {

struct RecordSettings
{
  std::string file_path;
  int recording_count;
  nlohmann::json pipeline_config;

  bool operator==(const RecordSettings &other) const  {
      return recording_count == other.recording_count && file_path == other.file_path && pipeline_config == other.pipeline_config;
  }

  bool requires_rebuild(const RecordSettings& old) const;
};

class RecordingSession {
public:
  struct Result {
    bool        completed;
    bool        discard_file;
    std::string failure;
  };

  explicit RecordingSession(size_t to_write) : to_write_(to_write) {}

  std::stop_token token() const { return stop_.get_token(); }
  void            request_stop() { stop_.request_stop(); }

  // Failure and completion claim the same boundary, so only one can win.
  bool fail(const std::string &message) {
    const std::lock_guard lock(mutex_);
    if (!accepting_failure_ || !failure_.empty() || stop_.stop_requested())
      return false;
    failure_ = message;
    stop_.request_stop();
    return true;
  }

  Result finish(std::string error = {}) {
    const std::lock_guard lock(mutex_);
    accepting_failure_ = false;
    if (!failure_.empty())
      error = failure_;
    const bool completed = !stop_.stop_requested() && error.empty();
    return {completed, !failure_.empty(), std::move(error)};
  }

  void register_write(size_t bytes) {
    const std::lock_guard lock(mutex_);
    written_ += bytes;
  }

  size_t get_remaining_to_write() const {
    const std::lock_guard lock(mutex_);
    assert(to_write_ >= written_);
    return to_write_ - written_;
  }

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
  struct RecordingGeometry {
    uint8_t bits_per_pixel;
    size_t  frame_width;
    size_t  frame_height;
  };

  Recorder(const std::string &file_path, uint32_t frame_count, size_t buffer_part_count,
           CameraBufferQueue &queue, const RecordingGeometry &g,
           const nlohmann::json &pipeline_settings, RecordingSession &session)
      : writer_{file_path,
                holofile::Header{
                    .magic_number   = holofile::Header::MAGIC_NUMBER_LE,
                    .version        = holofile::Header::CURRENT_VERSION,
                    .bits_per_pixel = g.bits_per_pixel,
                    .frame_width    = static_cast<uint32_t>(g.frame_width),
                    .frame_height   = static_cast<uint32_t>(g.frame_height),
                    .frame_count    = frame_count,
                    .data_size_in_bytes =
                        frame_count * g.frame_height * g.frame_width * g.bits_per_pixel / 8,
                    .endianness = holofile::Header::LITTLE_ENDIAN,
                },
                holofile::Footer{pipeline_settings}},
        frame_to_record_{frame_count}, buffer_part_count_{buffer_part_count}, current_frame_{0},
        queue_{queue}, session_{session} {
    if (buffer_part_count == 0)
      throw std::invalid_argument("Cannot record with zero buffer parts");
    logger()->info(
        "Create recorder with width: {}, height: {}, bits_per_pixel: {}, frame_count: {}",
        g.frame_width, g.frame_height, g.bits_per_pixel, frame_count);
    queue_.subscribe_b();
  }

  ~Recorder() {
    queue_.unsubscribe_b();
    writer_.flush();
  }

  size_t execute(std::stop_token cancelled, std::function<void(size_t)> poll_metrics) {
    batch_ = 0;
    std::atomic<bool>  stop_requested{false};
    std::stop_callback on_stop(cancelled,
                               [&] { stop_requested.store(true, std::memory_order_release); });

    Clock::time_point last_poll     = Clock::now();
    constexpr auto    poll_interval = std::chrono::milliseconds(500);

    while (current_frame_ < frame_to_record_ && !cancelled.stop_requested()) {
      const auto *frame = queue_.read_b(&stop_requested);
      if (!frame) {
        if (cancelled.stop_requested())
          break;
        throw std::runtime_error("Camera queue closed before recording completed");
      }
      if (!frame->base)
        throw std::runtime_error("Camera recording buffer has no data");

      const auto to_write = std::min(buffer_part_count_, frame_to_record_ - current_frame_);
      writer_.write_frames(reinterpret_cast<const uint8_t *>(frame->base), to_write);
      queue_.release_b();
      current_frame_ += to_write;
      session_.register_write(to_write);
      ++batch_;

      auto now = Clock::now();
      if (now - last_poll > poll_interval) {
        poll_metrics(current_frame_);
      }
    }

    if (!cancelled.stop_requested())
      writer_.write_footer();
    return current_frame_;
  }

private:
  holofile::Writer   writer_;
  size_t             frame_to_record_;
  size_t             batch_;
  size_t             buffer_part_count_;
  size_t             current_frame_;
  CameraBufferQueue &queue_;
  RecordingSession  &session_;
};

void recorder_worker(const holotask::sources::RecordSettings &settings,
                     const Recorder::RecordingGeometry &g, size_t buffer_part_count,
                     CameraBufferQueue &queue, RecordingSession &session,
                     std::atomic<bool>                       &recording,
                     std::function<void(size_t)>              poll_metrics_callback,
                     std::function<void(size_t)>              finished_callback,
                     std::function<void(const std::string &)> failed_callback) {
  logger()->info("[Recorder] started recorder thread");
  const auto  cancelled      = session.token();
  size_t      frames_written = 0;
  std::string failure;
  try {
    Recorder rec{settings.file_path,
                 static_cast<uint32_t>(settings.recording_count),
                 buffer_part_count,
                 queue,
                 g,
                 settings.pipeline_config,
                 session};
    frames_written = rec.execute(cancelled, poll_metrics_callback);
  } catch (const std::exception &e) {
    failure = e.what();
  } catch (...) {
    failure = "Unknown camera recording error";
  }
  // The recorder has released reader B and closed the file. Claim the result
  // before acquisition can mark this session as failed.
  auto result = session.finish(std::move(failure));
  failure     = std::move(result.failure);
  if (!failure.empty()) {
    if (result.discard_file)
      failure = remove_incomplete_camera_recording(settings.file_path, std::move(failure));
    logger()->error("[Recorder] recording failed: {}", failure);
  }
  try {
    if (!failure.empty())
      failed_callback(failure);
    else if (result.completed)
      finished_callback(frames_written);
  } catch (const std::exception &e) {
    logger()->error("[Recorder] failed to report recording result: {}", e.what());
  } catch (...) {
    logger()->error("[Recorder] failed to report recording result");
  }
  recording.store(false, std::memory_order_release);
  logger()->info("[Recorder] stopped recorder thread");
}


}