// Copyright 2025 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0
#pragma once

#undef min
#include <algorithm>

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

      size_t written_last_interval = 0;

      while (frames_written_ < recording_count_ && !cancelled.stop_requested()) {
        const auto *frame = buffered_data(frames_written_);

        constexpr int batch_size = 64;
        const auto       to_write   = std::min(batch_size, recording_count_ - frames_written_);
        writer().write_frames(reinterpret_cast<const uint8_t *>(frame), to_write);
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
