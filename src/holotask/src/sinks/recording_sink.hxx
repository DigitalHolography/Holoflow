// Copyright 2025 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0
#pragma once

namespace holotask::sinks::detail {

template <typename W>
  requires WriterConcept<W>
void Sink<W>::set_recording_count(int count) {
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
        monitor_.update(ctx, frames_buffered_, 0);
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
void Sink<W>::flush_to_disk(holoflow::core::SyncCtx &ctx) {
  if (start_flush_) {
    on_flush_started();
    start_flush_ = false;
    frames_written_ = 0;
    make_writer();
  }
  if (!has_writer())
    throw std::logic_error("Sink flush started without a writer");

  const int batch_size = static_cast<int>(ctx.inputs[0].desc.shape[0]);
  const int remaining = recording_count_ - frames_written_;
  const int to_write = remaining > 0 ? std::min(remaining, batch_size) : 0;
  if (to_write > 0) {
    write_frames(buffered_data(static_cast<std::size_t>(frames_written_)),
                 static_cast<std::size_t>(to_write));
    frames_written_ += to_write;
    monitor_.update(ctx, frames_buffered_, frames_written_);
  }

  if (to_write == 0 || (flush_on_final_batch() && frames_written_ >= recording_count_)) {
    write_footer();
    flush();
    on_flush_finished();
    emit_finished_event(ctx);
    reset_writer();
    frames_written_ = 0;
    start_flush_ = true;
  }
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::recording_reset_callback() {
  frames_written_ = 0;
  start_flush_ = true;
  reset_writer();
  on_sink_reset();
}

template <typename W>
  requires WriterConcept<W>
void Sink<W>::on_execute_error(holoflow::core::SyncCtx &ctx, const std::exception &error) {
  emit_failed_event(ctx, error.what());
}

} // namespace holotask::sinks::detail
