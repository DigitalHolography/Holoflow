// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include "recording_sink.hh"

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

void RecordingSink::on_start_recording_event(holoflow::core::SyncCtx     &ctx,
                                             const holoflow_event::Event &event) {
  if (recording_) {
    logger()->error("[RecordingSink] Ignoring duplicate start_recording event");
    emit_failed_event(ctx, "Recording already in progress");
    return;
  }

  const auto record_path = event.data.value("record_path", std::string{});
  if (record_path.empty()) {
    logger()->error("[RecordingSink] Rejecting start_recording event with empty path");
    emit_failed_event(ctx, "Cannot start recording: empty path");
    return;
  }

  recording_path_  = record_path;
  frames_buffered_ = 0;
  recording_       = true;
  monitor_.start();
  monitor_.update(ctx, 0, 0, true);
}

void RecordingSink::on_stop_recording_event(holoflow::core::SyncCtx     &ctx,
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

  if (!recording_path_.empty() && std::remove(recording_path_.c_str()) != 0 && errno != ENOENT) {
    std::error_code ec(errno, std::generic_category());
    logger()->error(
        "RecordingSink::on_stop_recording_event] Failed to remove incomplete recording at {}: {}",
        recording_path_, ec.message());
    stop_error =
        "Failed to remove incomplete recording at " + recording_path_ + ": " + ec.message();
  }
  if (!stop_error.empty())
    emit_failed_event(ctx, stop_error);
  reset();
}

void RecordingSink::on_update_record_path_event(holoflow::core::SyncCtx     &ctx,
                                                const holoflow_event::Event &event) {
  (void)ctx;
  assert(event.data.contains("record_path"));
  recording_path_ = event.data.value("record_path", std::string{});
  recording_path_update_callback();
}

void RecordingSink::handle_events(holoflow::core::SyncCtx &ctx) {
  if (!ctx.event_reader)
    return;

  static std::map<std::string, EventHandler> event_handlers = {
      {"start_recording",
       [](RecordingSink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
         self.on_start_recording_event(ctx, event);
       }},
      {"stop_recording",
       [](RecordingSink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
         self.on_stop_recording_event(ctx, event);
       }},
      {"update_recording_path",
       [](RecordingSink &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
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

    logger()->debug("[RecordingSink::handle_events] received event: {}", type);

    auto handler_it = event_handlers.find(type);
    if (handler_it == event_handlers.end()) {
      HOLOVIBES_BUG("Unknown event type: {}", type);
    } else {
      handler_it->second(*this, ctx, *event);
    }
  }
}

void RecordingSink::emit_finished_event(holoflow::core::SyncCtx &ctx) {
  emit_event(ctx, "recording_finished", {{"frames_written", frames_buffered_}});
  monitor_.update(ctx, frames_buffered_, frames_buffered_, true);
  monitor_.finish(ctx);
  recording_ = false;
}

void RecordingSink::emit_failed_event(holoflow::core::SyncCtx &ctx, const std::string &message) {
  emit_event(ctx, "recording_failed", {{"message", message}});
}

void RecordingSink::reset() {
  recording_reset_callback();
  recording_       = false;
  frames_buffered_ = 0;
}

void RecordingSink::update_recording_path(std::string &settings_path) {
  if (recording_)
    settings_path = recording_path_;
  else {
    recording_path_ = settings_path;
  }
}

void RecordingSink::emit_event(holoflow::core::SyncCtx &ctx, const char *type,
                               nlohmann::json data) {
  if (!ctx.event_writer)
    return;

  data["type"] = type;
  data["path"] = recording_path_;
  (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                    .node_id   = "",
                                    .data      = std::move(data),
                                    .ts        = std::chrono::steady_clock::now()});
}

} // namespace holotask::sinks::detail