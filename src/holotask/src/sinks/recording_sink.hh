// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <string>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "bug.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow_event/router.hh"
#include "logger.hh"

namespace holotask::sinks::detail {

// -------------------------------------------------------------------------------------------------
// Common recording sink lifecycle
// -------------------------------------------------------------------------------------------------

class RecordingSink : public holoflow::core::ISyncTask {
protected:
  void handle_events(holoflow::core::SyncCtx &ctx, std::string &settings_path,
                     int configured_count) {
    if (!recording_)
      recording_path_ = settings_path;
    if (!ctx.event_reader)
      return;

    while (auto event = ctx.event_reader->try_pop()) {
      HOLOVIBES_CHECK(event->direction == holoflow_event::EventDirection::ToNode,
                      "Unexpected event direction");

      const auto type = event->data.at("type").get<std::string>();
      if (type == "start_recording") {
        if (recording_) {
          logger()->error("[RecordingSink] Ignoring duplicate start_recording event");
          emit_failed_event(ctx, "Recording already in progress");
          continue;
        }

        const auto record_path = event->data.value("record_path", std::string{});
        if (record_path.empty()) {
          logger()->error("[RecordingSink] Rejecting start_recording event with empty path");
          emit_failed_event(ctx, "Cannot start recording: empty path");
          continue;
        }
        if (configured_count <= 0) {
          const auto message = "Cannot start recording: invalid frame count (" +
                               std::to_string(configured_count) + ")";
          logger()->error("[RecordingSink] {}", message);
          emit_failed_event(ctx, message);
          continue;
        }

        settings_path    = record_path;
        recording_path_  = record_path;
        frames_buffered_ = 0;
        recording_       = true;
      } else if (type == "stop_recording") {
        if (!recording_) {
          logger()->warn("[RecordingSink] Ignoring stop_recording event while idle");
          continue;
        }

        std::string stop_error;
        try {
          on_recording_stopped();
        } catch (const std::exception &error) {
          stop_error = error.what();
        }

        if (!recording_path_.empty() && std::remove(recording_path_.c_str()) != 0 &&
            errno != ENOENT) {
          std::error_code ec(errno, std::generic_category());
          logger()->error("[RecordingSink] Failed to remove incomplete recording at {}: {}",
                          recording_path_, ec.message());
          stop_error =
              "Failed to remove incomplete recording at " + recording_path_ + ": " + ec.message();
        }
        if (!stop_error.empty())
          emit_failed_event(ctx, stop_error);
        reset();
      } else {
        HOLOVIBES_BUG("Unknown event type: {}", type);
      }
    }
  }

  void emit_finished_event(holoflow::core::SyncCtx &ctx) {
    emit_event(ctx, "recording_finished", {{"frames_written", frames_buffered_}});
  }

  void emit_failed_event(holoflow::core::SyncCtx &ctx, const std::string &message) {
    emit_event(ctx, "recording_failed", {{"message", message}});
  }

  void reset() {
    on_recording_reset();
    recording_       = false;
    frames_buffered_ = 0;
  }

  void update_recording_path(std::string &settings_path) {
    if (recording_)
      settings_path = recording_path_;
    else {
      recording_path_ = settings_path;
    }
  }

  bool               is_recording() const { return recording_; }
  const std::string &recording_path() const { return recording_path_; }

  virtual void on_recording_stopped() {}
  virtual void on_recording_reset() {}

  void emit_event(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data) {
    if (!ctx.event_writer)
      return;

    data["type"] = type;
    data["path"] = recording_path_;
    (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                      .node_id   = "",
                                      .data      = std::move(data),
                                      .ts        = std::chrono::steady_clock::now()});
  }

  bool recording_       = false;
  int  frames_buffered_ = 0;

private:
  std::string recording_path_;
};

} // namespace holotask::sinks::detail
