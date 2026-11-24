// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "recording_sink.hh"

namespace holotask::sinks::detail {

RecordingMonitor::RecordingMonitor() = default;

void RecordingMonitor::start() { started_ = last_update_ = Clock::now(); }

void RecordingMonitor::update(holoflow::core::SyncCtx &ctx, int acquired, int written, bool force) {
  const auto now = Clock::now();
  if (!force && now - last_update_ < std::chrono::milliseconds(200))
    return;
  last_update_       = now;
  const auto seconds = std::chrono::duration<double>(now - started_).count();
  send(ctx, {{"type", "acquisition_update"}, {"id", 0}, {"frame_acquired", acquired}});
  send(ctx, {{"type", "recording_update"},
             {"id", 0},
             {"frame_recorded", written},
             {"writing_speed", seconds > 0 ? static_cast<std::size_t>(written / seconds) : 0},
             {"elapsed_time", static_cast<std::size_t>(seconds)}});
}

void RecordingMonitor::finish(holoflow::core::SyncCtx &ctx) {
  send(ctx, {{"type", "recording_finished"}, {"id", 0}});
  send(ctx, {{"type", "record_flushing_finished"}, {"id", 0}});
}

void RecordingMonitor::send(holoflow::core::SyncCtx &ctx, nlohmann::json data) {
  if (ctx.event_writer)
    (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                      .node_id   = "",
                                      .data      = std::move(data),
                                      .ts        = Clock::now()});
}

} // namespace holotask::sinks::detail
