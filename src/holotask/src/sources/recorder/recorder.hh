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

#include <span>
#include <stdint.h>
#include <string>

#include <nlohmann/json.hpp>

#include "holofile/holofile.hh"

#include "sources/buffer_queue/buffer_queue.hh"

#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"

namespace holotask::sources {

struct RecordingGeometry {
  uint8_t bits_per_pixel;
  size_t  frame_width;
  size_t  frame_height;
  size_t  frame_size_in_bytes;

  RecordingGeometry(uint8_t bits_per_pixel_, size_t frame_width_, size_t frame_height_);
};

class RecordingSession final {
public:
  enum class State { Free, Recording, WritingToDisk };
  using SessionId = uint32_t;

  using BufferRecordingFinishedCallback = std::function<void(SessionId)>;
  using PollMetricsCallback             = std::function<void(SessionId, size_t, size_t)>;
  using SuccessCallback                 = std::function<void(SessionId, size_t)>;
  using FailureCallback                 = std::function<void(SessionId, const std::string &)>;

  RecordingSession(SessionId id, CameraBufferQueue &buffer_queue, const RecordingGeometry &geometry,
                   size_t buffer_size, PollMetricsCallback metrics_callback,
                   SuccessCallback success_callback, FailureCallback failure_callback,
                   BufferRecordingFinishedCallback buffer_recording_finished_callback);

  RecordingSession(const RecordingSession &)            = delete;
  RecordingSession &operator=(const RecordingSession &) = delete;

  RecordingSession(RecordingSession &&other);
  RecordingSession &operator=(RecordingSession &&other) = delete;

  ~RecordingSession();
  // start recording workers
  // recording is set to true when starting and then is set to false when all the camera buffers
  // required for the record are copied to the session buffer
  void start(const RecordSettings &settings, size_t buffer_part_count, std::atomic<bool> &recording,
             std::atomic<size_t> &frame_recorded_since_start);

  State     state() const { return state_.load(std::memory_order_acquire); }
  bool      is_free() const { return state() == State::Free; }
  size_t    remaining_to_record() const;
  SessionId id() const { return id_; }

  // send stop request to the workers and join
  void stop();

private:
  void worker(const RecordSettings &settings, size_t buffer_part_count,
              std::atomic<bool> &recording, std::atomic<size_t> &frame_recorded_since_start);

  void remove_incomplete_camera_recording(const std::string &path);

  bool cancel_worker_thread();

private:
  CameraBufferQueue &buffer_queue_;

  std::atomic<State> state_ = State::Free;
  SessionId          id_    = 0;

  RecordingGeometry geometry_;

  std::vector<std::byte> buffer_; // buffer where the frame are stored before being flushed to disk
  std::jthread worker_thread_;    // the recorder and writer are created and destroyed for each
                                  // recording session, so it is local to the worker thread;

  PollMetricsCallback             metrics_callback_;
  SuccessCallback                 success_callback_;
  FailureCallback                 failure_callback_;
  BufferRecordingFinishedCallback buffer_recording_finished_callback_;

  std::atomic<size_t> remaining_to_record_ = 0;
};

class RecordWriter final {
public:
  RecordWriter(RecordingSession::SessionId id, std::span<std::byte> buffer,
               const std::string &file_path, uint32_t frame_count, uint32_t buffer_part_count,
               const RecordingGeometry &g, const nlohmann::json &pipeline_settings,
               RecordingSession::PollMetricsCallback poll_metrics);

  ~RecordWriter();

  RecordWriter(const RecordWriter &)            = delete;
  RecordWriter &operator=(const RecordWriter &) = delete;
  RecordWriter(RecordWriter &&)                 = delete;
  RecordWriter &operator=(RecordWriter &&)      = delete;

  // writes the frame buffered to the disk
  void execute(std::stop_token cancelled);

  // called from the recorder
  // frames are copied
  void push_frames(const std::byte *frames);

private:
  std::byte *wait_next_frame();

private:
  RecordingSession::SessionId id_;
  holofile::Writer            writer_;

  std::span<std::byte> &buffer_;

  size_t       frame_to_write_ = 0;
  const size_t buffer_part_count_;
  const size_t batch_size_in_bytes;

  // sync
  std::atomic<size_t> buffer_offset_  = 0;
  size_t              current_offset_ = 0;

  // callback
  RecordingSession::PollMetricsCallback poll_metrics_;
};

// read B
class Recorder final {
public:
  using Clock = std::chrono::steady_clock;

  Recorder(uint32_t frame_count, size_t buffer_part_count, CameraBufferQueue &queue,
           RecordWriter &writer);

  ~Recorder();

  Recorder(const Recorder &)            = delete;
  Recorder &operator=(const Recorder &) = delete;
  Recorder(Recorder &&)                 = delete;
  Recorder &operator=(Recorder &&)      = delete;

  void execute(std::stop_token cancelled, std::atomic<size_t> &remaining_to_record);

  // set frame_recorded_since_start to 0
  // set recording to true before execute
  // set recording to false when execute end
  static void worker(CameraBufferQueue &queue, size_t frame_to_record, size_t buffer_part_count,
                     std::atomic<bool> &recording, std::atomic<size_t> &frame_recorded_since_start);

private:
  RecordWriter      &writer_;
  size_t             frame_to_record_;
  size_t             buffer_part_count_;
  CameraBufferQueue &queue_;
};

struct RecordingContext {
  std::atomic<bool>   recording_                   = false;
  std::atomic<size_t> acquired_since_record_start_ = 0;
  std::atomic<bool>   acquiring_for_record_        = false;

  std::vector<RecordingSession> sessions_;

  void
  invalidate(size_t session_count, CameraBufferQueue &buffer_queue, size_t buffer_size,
             const RecordingGeometry &g, RecordingSession::PollMetricsCallback metrics_callback,
             RecordingSession::SuccessCallback                 success_callback,
             RecordingSession::FailureCallback                 failure_callback,
             RecordingSession::BufferRecordingFinishedCallback buffer_recording_finished_callback);
};
} // namespace holotask::sources
