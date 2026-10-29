// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <chrono>
#include <string>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow_event/router.hh"
#include "logger.hh"

namespace holotask::sinks::detail {

// -------------------------------------------------------------------------------------------------
// Common recording sink lifecycle
// -------------------------------------------------------------------------------------------------

// Single-queue sinks use queue zero, matching the recording monitor's event contract.
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

  void reset();

  void update_recording_path(std::string &settings_path);

  bool               is_recording() const { return recording_; }
  const std::string &recording_path() const { return recording_path_; }
  std::string       &recording_path() { return recording_path_; }

  virtual void recording_stopped_callback() {}
  virtual void recording_reset_callback() {}
  virtual void recording_path_update_callback() {}

  void on_start_recording_event(holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event);
  void on_stop_recording_event(holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event);
  void on_update_record_path_event(holoflow::core::SyncCtx     &ctx,
                                   const holoflow_event::Event &event);

  void emit_event(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data);

protected:
  RecordingMonitor monitor_;
  bool             recording_       = false;
  int              frames_buffered_ = 0;

private:
  std::string recording_path_;
};

} // namespace holotask::sinks::detail
