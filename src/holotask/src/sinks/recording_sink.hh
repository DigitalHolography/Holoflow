// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow_event/router.hh"
#include "logger.hh"

namespace holotask::sinks::detail {

class RecordingMonitor {
public:
  void start() { started_ = last_update_ = Clock::now(); }
  void update(holoflow::core::SyncCtx &ctx, int acquired, int written, bool force = false) {
    const auto now = Clock::now();
    if (!force && now - last_update_ < std::chrono::milliseconds(200))
      return;
    last_update_       = now;
    const auto seconds = std::chrono::duration<double>(now - started_).count();
    send(ctx, {{"type", "acquisition_update"}, {"id", 0}, {"frame_acquired", acquired}});
    send(ctx, {{"type", "recording_update"},
               {"id", 0},
               {"frame_recorded", written},
               {"writing_speed", seconds > 0 ? static_cast<size_t>(written / seconds) : 0},
               {"elapsed_time", static_cast<size_t>(seconds)}});
  }
  void finish(holoflow::core::SyncCtx &ctx) {
    send(ctx, {{"type", "recording_finished"}, {"id", 0}});
    send(ctx, {{"type", "record_flushing_finished"}, {"id", 0}});
  }

private:
  static void send(holoflow::core::SyncCtx &ctx, nlohmann::json data) {
    if (ctx.event_writer)
      (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                        .node_id   = "",
                                        .data      = std::move(data),
                                        .ts        = Clock::now()});
  }
  using Clock                    = std::chrono::steady_clock;
  Clock::time_point started_     = Clock::now();
  Clock::time_point last_update_ = started_;
};

class RecordingSink : public holoflow::core::ISyncTask {
protected:
  using EventHandler = std::function<void(RecordingSink &, holoflow::core::SyncCtx &,
                                          const holoflow_event::Event &)>;
  void handle_events(holoflow::core::SyncCtx &ctx);
  void emit_finished_event(holoflow::core::SyncCtx &ctx);
  void emit_failed_event(holoflow::core::SyncCtx &ctx, const std::string &message);

  void emit_recording_update_event(holoflow::core::SyncCtx &ctx, size_t frames_written,
                                   size_t writing_speed, size_t elapsed_time);
  void reset();
  void update_recording_path(std::string &settings_path);
  bool is_recording() const { return recording_; }
  const std::string &recording_path() const { return recording_path_; }
  std::string       &recording_path() { return recording_path_; }
  virtual void       recording_stopped_callback() {}
  virtual void       recording_reset_callback() {}
  virtual void       recording_path_update_callback() {}
  void on_start_recording_event(holoflow::core::SyncCtx &, const holoflow_event::Event &);
  void on_stop_recording_event(holoflow::core::SyncCtx &, const holoflow_event::Event &);
  void on_update_record_path_event(holoflow::core::SyncCtx &, const holoflow_event::Event &);
  void emit_event(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data);

protected:
  RecordingMonitor monitor_;
  bool             recording_       = false;
  int              frames_buffered_ = 0;

private:
  std::string recording_path_;
};

template <typename W>
concept WriterConcept = requires(W &writer, const std::uint8_t *data, std::size_t count) {
  writer.write_frames(data, count);
  writer.write_footer();
  writer.flush();
};

template <typename W>
  requires WriterConcept<W>
class Sink : public RecordingSink {
public:
  Sink() = default;
  ~Sink() override { shutdown_async_writer(); }
  Sink(const Sink &)                          = delete;
  Sink &operator=(const Sink &)               = delete;
  Sink(Sink &&)                               = delete;
  Sink                    &operator=(Sink &&) = delete;
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

protected:
  template <typename... Args> W &emplace_writer(Args &&...args) {
    writer_.emplace(std::forward<Args>(args)...);
    return *writer_;
  }
  bool has_writer() const noexcept { return writer_.has_value(); }
  void reset_writer() noexcept { writer_.reset(); }
  W   &writer() { return writer_.value(); }
  void write_frames(const std::uint8_t *data, std::size_t count) {
    writer().write_frames(data, count);
  }
  void write_footer() { writer().write_footer(); }
  void flush() { writer().flush(); }
  void set_recording_count(int count);
  int  recording_count() const noexcept { return recording_count_; }
  int  frames_written() const noexcept { return frames_written_; }

  virtual int                 buffer_input(holoflow::core::SyncCtx &, int max_frames) = 0;
  virtual const std::uint8_t *buffered_data(std::size_t frame_index) const            = 0;
  virtual std::size_t         buffered_frame_size() const                             = 0;
  virtual void                make_writer()                                           = 0;
  virtual void                writer_start() {}
  virtual void                on_flush_started() {}
  virtual void                on_flush_finished() {}
  virtual void                on_sink_reset() {}
  virtual void                on_execute_error(holoflow::core::SyncCtx &, const std::exception &);
  void                        flush_to_disk(holoflow::core::SyncCtx &ctx);
  void                        shutdown_async_writer();
  void                        recording_reset_callback() override;

private:
  void start_writer_thread(holoflow::core::SyncCtx &ctx);
  void poll_writer(holoflow::core::SyncCtx &ctx);

private:
  std::optional<W>  writer_;
  int               recording_count_ = 0;
  int               frames_written_  = 0;
  bool              start_flush_     = true;
  std::jthread      writer_thread_;
  std::atomic<bool> writer_done_{false};
};

} // namespace holotask::sinks::detail

namespace holotask::sinks {
template <typename W>
concept WriterConcept = detail::WriterConcept<W>;
template <typename W>
  requires detail::WriterConcept<W>
using Sink = detail::Sink<W>;
} // namespace holotask::sinks

#include "recording_sink.hxx"
