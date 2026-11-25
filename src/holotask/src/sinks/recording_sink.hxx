// Copyright 2025 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0
#pragma once

#undef min
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <system_error>

#include "bug.hh"

namespace holotask::sinks::detail {

template <typename W>
  requires WriterConcept<W>
void Sink<W>::set_recording_count(int count) {
  if (count < 0)
    throw std::invalid_argument("Sink recording count cannot be negative");
  recording_count_ = count;
}

template <typename W>
  requires WriterConcept<W>
holoflow::core::OpResult Sink<W>::execute(holoflow::core::SyncCtx &ctx) {
  handle_events(ctx);
  if (!is_recording())
    return holoflow::core::OpResult::Ok;
  try {
    if (frames_buffered_ < recording_count_) {
      const int copied = buffer_input(ctx, recording_count_ - frames_buffered_);
      if (copied < 0 || copied > recording_count_ - frames_buffered_)
        throw std::runtime_error("Sink buffer hook returned an invalid frame count");
      frames_buffered_ += copied;
      if (copied > 0)
        monitor_.update(ctx, frames_buffered_, frames_written_);
    }
    if (frames_buffered_ >= recording_count_)
      flush_to_disk(ctx);
  } catch (const std::exception &error) {
    on_execute_error(ctx, error);
  }
  return holoflow::core::OpResult::Ok;
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::handle_events(holoflow::core::SyncCtx &ctx) {
  if (!ctx.event_reader)
    return;

  static std::map<std::string, EventHandler> event_handlers = {
      {"start_recording",
       [](Sink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
         self.on_start_recording_event(ctx, event);
       }},
      {"stop_recording",
       [](Sink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
         self.on_stop_recording_event(ctx, event);
       }},
      {"update_recording_path",
       [](Sink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
         self.on_update_record_path_event(ctx, event);
       }},
  };

  while (true) {
    auto event = ctx.event_reader->try_pop();
    if (!event.has_value())
      break;

    HOLOVIBES_CHECK(event->direction == holoflow_event::EventDirection::ToNode,
                    "Unexpected event direction");

    const auto type = event->data.at("type").get<std::string>();

    logger()->debug("[Sink::handle_events] received event: {}", type);

    auto handler_it = event_handlers.find(type);
    if (handler_it == event_handlers.end()) {
      HOLOVIBES_BUG("Unknown event type: {}", type);
    } else {
      handler_it->second(*this, ctx, *event);
    }
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::on_start_recording_event(holoflow::core::SyncCtx     &ctx,
                                       const holoflow_event::Event &event) {
  if (recording_) {
    logger()->error("[Sink] Ignoring duplicate start_recording event");
    emit_failed_event(ctx, "Recording already in progress");
    return;
  }

  const auto record_path = event.data.value("record_path", std::string{});
  if (record_path.empty()) {
    logger()->error("[Sink] Rejecting start_recording event with empty path");
    emit_failed_event(ctx, "Cannot start recording: empty path");
    return;
  }

  recording_path_  = record_path;
  frames_buffered_ = 0;
  recording_       = true;
  monitor_.start();
  monitor_.update(ctx, 0, 0, true);
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::on_stop_recording_event(holoflow::core::SyncCtx     &ctx,
                                      const holoflow_event::Event &event) {
  (void)event;
  if (!recording_)
    return;

  std::string stop_error;
  try {
    recording_stopped_callback();
  } catch (const std::exception &error) {
    stop_error = error.what();
  }

  emit_failed_event(ctx, !stop_error.empty() ? stop_error : "recording cancelled");
  reset();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::remove_recording_file() {
  if (!recording_path_.empty() && std::remove(recording_path_.c_str()) != 0 && errno != ENOENT) {
    std::error_code ec(errno, std::generic_category());
    logger()->error("Sink::remove_recording_file] Failed to remove incomplete recording at {}: {}",
                    recording_path_, ec.message());
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::on_update_record_path_event(holoflow::core::SyncCtx     &ctx,
                                          const holoflow_event::Event &event) {
  (void)ctx;
  assert(event.data.contains("record_path"));
  // The asynchronous writer reads the active path while constructing/finalizing its
  // format writer. Keep active-recording state immutable; path changes apply only
  // while idle and can be picked up by the next recording.
  if (recording_) {
    logger()->warn("[Sink] Ignoring recording-path update while recording is active");
    return;
  }
  recording_path_ = event.data.value("record_path", std::string{});
  recording_path_update_callback();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::cancel(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  if (recording_)
    recording_stopped_callback();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::emit_finished_event(holoflow::core::SyncCtx &ctx) {
  monitor_.update(ctx, frames_buffered_, frames_buffered_, true);
  monitor_.finish(ctx);
  recording_ = false;
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::emit_failed_event(holoflow::core::SyncCtx &ctx, const std::string &message) {
  emit_event(ctx, "recording_failed", {{"message", message}});
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::emit_recording_update_event(holoflow::core::SyncCtx &ctx, std::size_t frames_written,
                                          std::size_t writing_speed, std::size_t elapsed_time) {
  auto event = holoflow_event::Event{
      .direction = holoflow_event::EventDirection::ToUi,
      .node_id   = "",
      .data      = nlohmann::json{{"type", "recording_update"},
                                  {"frame_recorded", frames_written},
                                  {"writing_speed", writing_speed},
                                  {"elapsed_time", elapsed_time}},
      .ts        = std::chrono::steady_clock::now(),
  };
  HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                  "Failed to emit recording_update event");
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::reset() {
  recording_reset_callback();
  recording_       = false;
  frames_buffered_ = 0;
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::update_recording_path(std::string &settings_path) {
  if (recording_)
    settings_path = recording_path_;
  else {
    recording_path_ = settings_path;
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::emit_event(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data) {
  if (!ctx.event_writer)
    return;

  data["type"] = type;
  data["path"] = recording_path_;
  (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                    .node_id   = "",
                                    .data      = std::move(data),
                                    .ts        = std::chrono::steady_clock::now()});
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::recording_stopped_callback() {
  logger()->info("[Sink::recording_stopped_callback] stop requested");
  shutdown_async_writer();
  reset();
  remove_recording_file();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::start_writer_thread(holoflow::core::SyncCtx &ctx) {
  writer_done_.store(false, std::memory_order_relaxed);
  writer_thread_ = std::jthread([this, &ctx](std::stop_token cancelled) {
    try {
      make_writer();
      writer_start();

      monitor_.update(ctx, frames_buffered_, 0, true);

      using Clock                     = std::chrono::steady_clock;
      Clock::time_point start         = Clock::now();
      Clock::time_point last_poll     = Clock::now();
      constexpr auto    poll_interval = std::chrono::milliseconds(200);

      std::size_t written_last_interval = 0;

      while (frames_written_ < recording_count_ && !cancelled.stop_requested()) {
        const auto *frame = buffered_data(frames_written_);

        constexpr int batch_size = 64;
        const auto    to_write   = std::min(batch_size, recording_count_ - frames_written_);
        writer().write_frames(reinterpret_cast<const std::uint8_t *>(frame), to_write);
        frames_written_ += to_write;
        written_last_interval += to_write;

        auto now     = Clock::now();
        auto elapsed = now - last_poll;
        if (elapsed > poll_interval) {
          monitor_.update(ctx, frames_buffered_, frames_written_);
          logger()->info("[Sink::start_writer_thread] {} total frames written to disk ({} during "
                         "the last {} ms)",
                         frames_written_, written_last_interval,
                         std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
          written_last_interval = 0;
          last_poll             = Clock::now();
        }
      }

      if (!cancelled.stop_requested())
        writer().write_footer();

      // The writer is owned and closed by the same thread that performed its IO.
      reset_writer();
      writer_done_.store(true, std::memory_order_release);
    } catch (const std::exception &e) {
      logger()->error("[Sink::start_writer_thread] an error occurred: {}", e.what());
      emit_failed_event(ctx, e.what());
    }
  });
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::poll_writer(holoflow::core::SyncCtx &ctx) {
  if (writer_done_.load(std::memory_order_acquire)) {
    writer_thread_.join();
    on_flush_finished();
    emit_finished_event(ctx);
    frames_written_ = 0;
    start_flush_    = true;
    reset();
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::shutdown_async_writer() {
  if (!writer_thread_.joinable())
    return;
  writer_thread_.request_stop();
  writer_thread_.join();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::flush_to_disk(holoflow::core::SyncCtx &ctx) {
  if (start_flush_) {
    on_flush_started();
    start_flush_    = false;
    frames_written_ = 0;
    // The worker consumes the live recording buffer. Subsequent execute() calls
    // return without waiting or reusing it until the worker signals completion.
    start_writer_thread(ctx);
  }

  if (writer_thread_.joinable()) {
    poll_writer(ctx);
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::recording_reset_callback() {
  shutdown_async_writer();
  frames_written_ = 0;
  start_flush_    = true;
  reset_writer();
  on_sink_reset();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::on_execute_error(holoflow::core::SyncCtx &ctx, const std::exception &error) {
  emit_failed_event(ctx, error.what());
  reset();
}

} // namespace holotask::sinks::detail
