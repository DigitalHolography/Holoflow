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

// -------------------------------------------------------------------------------------------------
// RecordingGeometry
// -------------------------------------------------------------------------------------------------

RecordingGeometry::RecordingGeometry(uint8_t bits_per_pixel_, size_t frame_width_,
                                     size_t frame_height_)
    : bits_per_pixel{bits_per_pixel_}, frame_width{frame_width_}, frame_height{frame_height_},
      frame_size_in_bytes{frame_height * frame_width * bits_per_pixel / 8} {}

// -------------------------------------------------------------------------------------------------
// RecordingSession
// -------------------------------------------------------------------------------------------------

RecordingSession::RecordingSession(SessionId id, CameraBufferQueue &buffer_queue,
                                   const RecordingGeometry &geometry, size_t buffer_size,
                                   std::byte *buffer, PollMetricsCallback metrics_callback,
                                   SuccessCallback                success_callback,
                                   FailureCallback                failure_callback,
                                   BufferFlushingFinishedCallback buffer_flushing_finished_callback)
    : buffer_queue_{buffer_queue}, id_{id}, geometry_{geometry}, buffer_(buffer),
      buffer_size_{buffer_size}, metrics_callback_{metrics_callback},
      success_callback_{success_callback}, failure_callback_{failure_callback},
      buffer_flushing_finished_callback_{buffer_flushing_finished_callback} {}

RecordingSession::RecordingSession(RecordingSession &&other)
    : buffer_queue_(other.buffer_queue_), state_(other.state_.load(std::memory_order_relaxed)),
      id_(other.id_), geometry_(std::move(other.geometry_)), buffer_(std::move(other.buffer_)),
      worker_thread_(std::move(other.worker_thread_)),
      metrics_callback_(std::move(other.metrics_callback_)),
      success_callback_(std::move(other.success_callback_)),
      failure_callback_(std::move(other.failure_callback_)),
      buffer_flushing_finished_callback_(std::move(other.buffer_flushing_finished_callback_)),
      remaining_to_record_(other.remaining_to_record_.load(std::memory_order_relaxed)) {}

RecordingSession::~RecordingSession() {
  if (cancel_worker_thread())
    logger()->warn("[RecordingSession::~RecordingSession] session {} destroyed before record end "
                   "(record discarded)",
                   id_);
}

size_t RecordingSession::remaining_to_record() const {
  return remaining_to_record_.load(std::memory_order_acquire);
}

namespace {

const char *state_to_string(RecordingSession::State state) {
  switch (state) {
  case RecordingSession::State::Free:
    return "Free";

  case RecordingSession::State::WritingToDisk:
    return "WritingToDisk";

  case RecordingSession::State::Recording:
    return "Recording";
  default:
    return "unknown";
  }
}
} // namespace

// start recording workers
// recording is set to true when starting and then is set to false when all the camera buffers
// required for the record are copied to the session buffer
void RecordingSession::start(const RecordSettings &settings, size_t buffer_part_count,
                             std::atomic<bool>   &recording,
                             std::atomic<size_t> &frame_recorded_since_start) {

  auto s = state();
  if (s != State::Free) {
    logger()->error("[RecordingSession::start] cannot start recording session {} because it is "
                    "already running (state: {})",
                    id_, state_to_string(s));
    throw std::logic_error("[RecordingSession::start] cannot start a recording session because "
                           "it is already running");
  }

  cancel_worker_thread(); // ensure the worker thread is joined (state == Free => no work is
                          // discarded)

  state_.store(State::Recording, std::memory_order_release);
  worker(settings, buffer_part_count, recording, frame_recorded_since_start);
}

// send stop request to the workers and join
void RecordingSession::stop() {
  if (cancel_worker_thread()) {
    state_.store(State::Free, std::memory_order_release);
    logger()->info("[RecordingSession::stop] stopped recording session {}", id_);
  } else
    logger()->info("[RecordingSession::stop] session {} was not active", id_);
}

bool RecordingSession::cancel_worker_thread() {

  if (worker_thread_.joinable()) {
    worker_thread_.request_stop();
    worker_thread_.join();
    return true;
  }
  return false;
}
void RecordingSession::worker(const RecordSettings &settings, size_t buffer_part_count,
                              std::atomic<bool>   &recording,
                              std::atomic<size_t> &frame_recorded_since_start) {

  cancel_worker_thread();
  worker_thread_ = std::jthread([this, settings, buffer_part_count, &recording,
                                 &frame_recorded_since_start](std::stop_token cancelled) {
    std::atomic<uint32_t> finished = 0;

    logger()->info("[RecordingSession::worker::worker_thread] worker thread started");
    {

      auto writer = RecordWriter(id_, std::span(buffer_, buffer_size_), settings.file_path,
                                 settings.recording_count, static_cast<uint32_t>(buffer_part_count),
                                 geometry_, settings.pipeline_config, metrics_callback_);

      remaining_to_record_.store(settings.recording_count, std::memory_order_release);

      // start rec and writers on other threads
      auto rec_thread = std::jthread([this, &settings, &recording, &frame_recorded_since_start,
                                      &finished, cancelled, buffer_part_count, &writer] {
        logger()->info("[RecordingSession::worker::rec_thread] recording thread started");
        recording.store(true, std::memory_order_release);
        frame_recorded_since_start.store(0, std::memory_order_release);

        {
          auto rec = Recorder(settings.recording_count, buffer_part_count, buffer_queue_, writer);
          rec.execute(cancelled, remaining_to_record_);
        }

        if (!cancelled.stop_requested()) {
          state_.store(State::WritingToDisk, std::memory_order_release);
          finished.fetch_add(1);
          success_callback_(id_, settings.recording_count);
        } else {
          logger()->info("[RecordingSession::worker] record cancelled");
          failure_callback_(id_, "record cancelled");
        }

        recording.store(false, std::memory_order_release);
        logger()->info(
            "[RecordingSession::worker::rec_thread] recording thread finished (cancelled: {})",
            cancelled.stop_requested());
      });

      auto writer_thread = std::jthread([this, &writer, &finished, cancelled, &settings] {
        logger()->info("[RecordingSession::worker::writer_thread] writer thread started");
        writer.execute(cancelled);
        if(!cancelled.stop_requested()) {
          finished.fetch_add(1);
          buffer_flushing_finished_callback_(id_);
        }
        logger()->info(
            "[RecordingSession::worker::writer_thread] writer thread finished (cancelled: {})",
            cancelled.stop_requested());
      });

      constexpr size_t thread_count = 2;
      while (finished.load(std::memory_order_acquire) < thread_count && !cancelled.stop_requested())
        ;

      rec_thread.join();
      writer_thread.join();
    }

    if (cancelled.stop_requested()) {
      remove_incomplete_camera_recording(settings.file_path);
    }
    logger()->info(
        "[RecordingSession::worker::worker_thread] worker thread finished (cancelled: {})",
        cancelled.stop_requested());
    state_.store(State::Free, std::memory_order_release);
  });
}

void RecordingSession::remove_incomplete_camera_recording(const std::string &path) {
  if (!path.empty() && std::remove(path.c_str()) != 0 && errno != ENOENT) {
    logger()->warn("[RecordingSession::remove_incomplete_camera_recording] failed to remove "
                   "incomplete file at: {}",
                   path);
  } else {
    logger()->warn(
        "[RecordingSession::remove_incomplete_camera_recording] removed incomplete record at: {}",
        path);
  }
}

// -------------------------------------------------------------------------------------------------
// RecordWriter
// -------------------------------------------------------------------------------------------------

RecordWriter::RecordWriter(RecordingSession::SessionId id, std::span<std::byte> buffer,
                           const std::string &file_path, uint32_t frame_count,
                           uint32_t buffer_part_count, const RecordingGeometry &g,
                           const nlohmann::json                 &pipeline_settings,
                           RecordingSession::PollMetricsCallback poll_metrics)
    : id_{id}, writer_{file_path,
                       holofile::Header{
                           .magic_number       = holofile::Header::MAGIC_NUMBER_LE,
                           .version            = holofile::Header::CURRENT_VERSION,
                           .bits_per_pixel     = g.bits_per_pixel,
                           .frame_width        = static_cast<uint32_t>(g.frame_width),
                           .frame_height       = static_cast<uint32_t>(g.frame_height),
                           .frame_count        = frame_count,
                           .data_size_in_bytes = frame_count * g.frame_size_in_bytes,
                           .endianness         = holofile::Header::LITTLE_ENDIAN,
                       },
                       holofile::Footer{pipeline_settings}},
      buffer_{buffer}, frame_to_write_{frame_count}, buffer_part_count_{buffer_part_count},
      batch_size_in_bytes{buffer_part_count * g.frame_size_in_bytes}, poll_metrics_{poll_metrics} {}

RecordWriter::~RecordWriter() { writer_.flush(); }

// writes the frame buffered to the disk
void RecordWriter::execute(std::stop_token cancelled) {
  logger()->info("[RecordWriter::execute] started writing of {} frames", frame_to_write_);
  std::atomic<bool>  stop_requested{false};
  std::stop_callback on_stop(cancelled,
                             [&] { stop_requested.store(true, std::memory_order_release); });

  using Clock                     = std::chrono::steady_clock;
  Clock::time_point start         = Clock::now();
  Clock::time_point last_poll     = Clock::now();
  constexpr auto    poll_interval = std::chrono::milliseconds(200);

  size_t written_last_interval = 0;

  size_t current_frame = 0;

  while (current_frame < frame_to_write_ && !cancelled.stop_requested()) {
    const auto frame = wait_next_frame();

    const auto to_write = std::min(buffer_part_count_, frame_to_write_ - current_frame);
    writer_.write_frames(reinterpret_cast<const uint8_t *>(frame), to_write);
    current_frame += to_write;
    written_last_interval += to_write;

    auto now     = Clock::now();
    auto elapsed = now - last_poll;
    if (elapsed > poll_interval) {
      size_t writing_speed = written_last_interval * 1000 /
                             std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                                 .count(); // speed in frames / s

      auto elapsed_time_since_start =
          std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count();
      poll_metrics_(id_, current_frame, writing_speed, elapsed_time_since_start);
      logger()->info("[RecordWriter {}] {} total frames written to disk ({} during the last {} ms)",
                     id_, current_frame, written_last_interval,
                     std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      written_last_interval = 0;
      last_poll             = Clock::now();
    }
  }

  if (!cancelled.stop_requested())
    writer_.write_footer();
}

// called from the recorder
// frames are copied
void RecordWriter::push_frames(const std::byte *frames) {
  auto offset = buffer_offset_.load(std::memory_order_acquire);

  memcpy(buffer_.data() + offset, frames, batch_size_in_bytes);

  buffer_offset_.store(offset + batch_size_in_bytes, std::memory_order_release);
}

std::byte *RecordWriter::wait_next_frame() {
  while (current_offset_ >= buffer_offset_.load(std::memory_order_acquire))
    ;

  auto offset = current_offset_;
  current_offset_ += batch_size_in_bytes;
  return buffer_.data() + offset;
}

// -------------------------------------------------------------------------------------------------
// Recorder
// -------------------------------------------------------------------------------------------------

// read B
Recorder::Recorder(uint32_t frame_count, size_t buffer_part_count, CameraBufferQueue &queue,
                   RecordWriter &writer)
    : frame_to_record_{frame_count}, buffer_part_count_{buffer_part_count}, queue_{queue},
      writer_{writer} {
  if (buffer_part_count == 0)
    throw std::invalid_argument("Cannot record with zero buffer parts");
  queue_.subscribe_b();
}

Recorder::~Recorder() {
  queue_.unsubscribe_b();
  queue_.clear_b();
}

void Recorder::execute(std::stop_token cancelled, std::atomic<size_t> &remaining_to_record) {
  logger()->info("[Recorder::execute] started recording of {} frames", frame_to_record_);
  size_t             current_frame = 0;
  std::atomic<bool>  stop_requested{false};
  std::stop_callback on_stop(cancelled,
                             [&] { stop_requested.store(true, std::memory_order_release); });

  Clock::time_point last_poll     = Clock::now();
  constexpr auto    poll_interval = std::chrono::milliseconds(200);

  while (current_frame < frame_to_record_ && !cancelled.stop_requested()) {
    const auto *frame = queue_.read_b(&stop_requested);
    if (!frame) {
      if (cancelled.stop_requested())
        break;
      throw std::runtime_error("Camera queue closed before recording completed");
    }
    if (!frame->base)
      throw std::runtime_error("Camera recording buffer has no data");

    writer_.push_frames(frame->base);
    queue_.release_b();

    current_frame += buffer_part_count_;
    auto rem = remaining_to_record.fetch_sub(buffer_part_count_);
    if (rem < buffer_part_count_)
      remaining_to_record.store(0, std::memory_order_release);
  }
}

// -------------------------------------------------------------------------------------------------
// RecordingContext
// -------------------------------------------------------------------------------------------------

void RecordingContext::invalidate(
    size_t session_count, CameraBufferQueue &buffer_queue, size_t buffer_size, std::byte *buffer,
    const RecordingGeometry &g, RecordingSession::PollMetricsCallback metrics_callback,
    RecordingSession::SuccessCallback                success_callback,
    RecordingSession::FailureCallback                failure_callback,
    RecordingSession::BufferFlushingFinishedCallback buffer_flushing_finished_callback) {

  sessions_.clear();

  recording_.store(false, std::memory_order_release);
  acquired_since_record_start_.store(0, std::memory_order_release);

  logger()->info("[RecordingContext::reset] creating {} recording session with buffer of {} bytes",
                 session_count, buffer_size);

  sessions_.reserve(session_count);
  size_t offset = 0;
  for (size_t i = 0; i < session_count; ++i) {
    sessions_.emplace_back(static_cast<RecordingSession::SessionId>(i), buffer_queue, g,
                           buffer_size, buffer + offset, metrics_callback, success_callback,
                           failure_callback, buffer_flushing_finished_callback);
    offset += buffer_size;
  }
}

} // namespace holotask::sources
