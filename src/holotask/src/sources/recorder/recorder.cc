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

#include "recorder.hh"

#include "logger.hh"

#undef min
#undef max

#include "algorithm"

namespace holotask::sources {

RecordingSession::RecordingSession(size_t to_write) : to_write_(to_write) {}

std::stop_token RecordingSession::token() const { return stop_.get_token(); }
void            RecordingSession::request_stop() { stop_.request_stop(); }

// Failure and completion claim the same boundary, so only one can win.
bool RecordingSession::fail(const std::string &message) {
  const std::lock_guard lock(mutex_);
  if (!accepting_failure_ || !failure_.empty() || stop_.stop_requested())
    return false;
  failure_ = message;
  stop_.request_stop();
  return true;
}

RecordingSession::Result RecordingSession::finish(std::string error) {
  const std::lock_guard lock(mutex_);
  accepting_failure_ = false;
  if (!failure_.empty())
    error = failure_;
  const bool completed = !stop_.stop_requested() && error.empty();
  return {completed, !failure_.empty(), std::move(error)};
}

void RecordingSession::register_write(size_t bytes) {
  const std::lock_guard lock(mutex_);
  written_ += bytes;
}

size_t RecordingSession::get_remaining_to_write() const {
  const std::lock_guard lock(mutex_);
  assert(to_write_ >= written_);
  return to_write_ - written_;
}

// read B
Recorder::Recorder(const std::string &file_path, uint32_t frame_count, size_t buffer_part_count,
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
  logger()->info("Create recorder with width: {}, height: {}, bits_per_pixel: {}, frame_count: {}",
                 g.frame_width, g.frame_height, g.bits_per_pixel, frame_count);
  queue_.subscribe_b();
}

Recorder::~Recorder() {
  queue_.unsubscribe_b();
  writer_.flush();
}

size_t Recorder::execute(std::stop_token cancelled, std::function<void(size_t)> poll_metrics) {
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

std::string Recorder::remove_incomplete_camera_recording(const std::string &path, std::string failure) {
  if (!path.empty() && std::remove(path.c_str()) != 0 && errno != ENOENT)
    failure += "; failed to remove incomplete recording: " +
               std::error_code(errno, std::generic_category()).message();
  return failure;
}

void Recorder::worker(const holotask::sources::RecordSettings &settings,
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

} // namespace holotask::sources