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

#include "holotask/sources/camera.hh"

#include "holocam/camera.hh"

#include <format>
#include <functional>
#include <optional>
#include <utility>

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const CameraSettings &s) {
  j = nlohmann::json{{"cfg_path", s.cfg_path}, {"record_settings", nullptr}};
  if (s.record_settings)
    j["record_settings"] = *s.record_settings;
}

void from_json(const nlohmann::json &j, CameraSettings &s) {
  j.at("cfg_path").get_to(s.cfg_path);
  s.record_settings.reset();
  if (j.contains("record_settings") && !j.at("record_settings").is_null())
    s.record_settings = j.at("record_settings").get<holocam::RecordSettings>();
}

} // namespace holotask::sources

#ifdef HOLOTASK_HAS_EGRABBER

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <functional>
#undef max
#undef min
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// Private implementation types
// -------------------------------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[AmetekS711EuresysCoaxlinkQSFPFactory] error: {}", msg);
    throw std::invalid_argument("AmetekS711EuresysCoaxlinkQSFPFactory error: " + msg);
  }
}

nlohmann::json load_cfg(const std::string &cfg_path) {
  check(!cfg_path.empty(), "cfg_path is empty");

  std::ifstream cfg_file(cfg_path);
  check(cfg_file.is_open(), std::format("could not open config file: {}", cfg_path));

  return nlohmann::json::parse(cfg_file);
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Task implementation (private)
// -------------------------------------------------------------------------------------------------

/**
 * Camera source task.
 */
class CameraTask : public holoflow::core::ISyncTask {
public:
  CameraTask(const CameraSettings &settings, const nlohmann::json &config)
      : settings_{settings}, cfg_{config} {
    std::unique_ptr<holocam::CameraBase> base = nullptr;

    if (settings.record_settings.has_value()) {
      auto options = holocam::RecordOptions{
          *settings.record_settings,
          {
              .flushing_finished =
                  [this](holocam::RecordingSessionId id) {
                    if (ctx_)
                      emit_camera_record_flushing_finished_event(*ctx_, id);
                  },
              .poll_metrics =
                  [this](holocam::RecordingSessionId id, size_t written, size_t speed,
                         size_t elapsed_time) {
                    if (ctx_)
                      emit_recording_update_event(*ctx_, id, written, speed, elapsed_time);
                  },

              .acquisition_metrics =
                  [this](holocam::RecordingSessionId id, size_t frames) {
                    if (ctx_)
                      emit_acquisition_update_event(*ctx_, id, frames);
                  },

              .success_callback =
                  [this](holocam::RecordingSessionId id, size_t frames) {
                    if (ctx_)
                      emit_recording_finished_event(*ctx_, id, frames);
                  },
              .failure_callback =
                  [this](holocam::RecordingSessionId id, const std::string &msg) {
                    if (ctx_)
                      emit_recording_failed_event(*ctx_, id, msg);
                  },
          }};
      base = factory_.create(settings.name, config, options);
    } else {
      base = factory_.create(settings.name, config, std::nullopt);
    }
  }

  ~CameraTask() override {
    camera_->stop_raw_record();
    camera_->stop_acquisition();
  }

  void handle_events(holoflow::core::SyncCtx &ctx) {
    if (!ctx.event_reader) {
      if (log_due(last_event_log_))
        logger()->warn("[CameraTask::handle_events] the given ctx has no event receiver");
      return;
    }

    using EventHandler =
        std::function<void(CameraTask &, holoflow::core::SyncCtx &, const holoflow_event::Event &)>;
    static std::map<std::string, EventHandler> event_handlers = {
        {"start_recording",
         [](auto &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
           self.on_start_recording_event(ctx, event);
         }},
        {"stop_recording",
         [](auto &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
           self.on_stop_recording_event(ctx, event);
         }},
        {"update_recording_path",
         [](auto &self, holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
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

      logger()->debug("[CameraTask::handle_events] received event: {}", type);

      auto handler_it = event_handlers.find(type);
      if (handler_it == event_handlers.end()) {
        HOLOVIBES_BUG("Unknown event type: {}", type);
      } else {
        handler_it->second(*this, ctx, *event);
      }
    }
  }

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    (void)index;
    throw std::out_of_range("AmetekS711EuresysCoaxlinkQSFP task has no inputs");
  }

  /**
   * Re-queue both bank buffers for the frame currently exposed as output 0.
   */
  void release_output(int index) override {
    if (index != 0) {
      throw std::out_of_range("Camera task has only one output at index 0");
    }

    camera_->release_frame();
    storage_access().owned_output_storage(0).ptr = nullptr;
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_events(ctx);

    if (!running_.load(std::memory_order_acquire)) {
      ctx_ = &ctx;
      camera_->start_acquisition();
    }

    while (!ctx.cancelled->load()) {
      auto *next = camera_->get_frame(ctx.cancelled); // return null if cancelled
      if (next == nullptr) {
        break;
      }
      if (next) {
        auto &storage = storage_access().owned_output_storage(0);
        storage.ptr   = next;

        ctx.outputs[0] = holoflow::core::TView{
            .desc    = ctx.outputs[0].desc,
            .storage = &storage,
        };
      }
      return holoflow::core::OpResult::Ok;
    }

    return holoflow::core::OpResult::Cancelled;
  }

  void cancel(holoflow::core::SyncCtx &ctx) override {
    logger()->info("[CameraTask::execute] cancelled");
    camera_->stop_acquisition();
    running_.store(false, std::memory_order_release);
    ctx_ = nullptr;
  }

  const nlohmann::json &get_cfg() const { return cfg_; }

  const CameraSettings &get_settings() const { return settings_; }

  void stop_acquisition() {
    if (!running_.load(std::memory_order_acquire)) {
      logger()->warn("[CameraTask::stop_acquisition] acquisition is not running");
    } else {
      camera_->stop_acquisition();
      running_.store(false);
    }
  }

  void update_settings(const CameraSettings &settings) {
    // TODO
  }

  void log_update_lifecycle(bool updated) {
    if (updated) {
      logger()->info("[CameraTask::log_update_lifecycle] update camera settings in place");

    } else {
      logger()->info("[CameraTask::log_update_lifecycle] update camera by replacing it");
    }
  }

private:
  void emit_camera_record_flushing_finished_event(holoflow::core::SyncCtx    &ctx,
                                                  holocam::RecordingSessionId id) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "record_flushing_finished"},
                {"id", id},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit record_flushing_finished event");
  }

  void emit_recording_finished_event(holoflow::core::SyncCtx &ctx, holocam::RecordingSessionId id,
                                     size_t frames_written) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_finished"},
                {"id", id},
                {"frames_written", frames_written},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_finished event");
  }

  void emit_recording_update_event(holoflow::core::SyncCtx &ctx, holocam::RecordingSessionId id,
                                   size_t frames_written, size_t writing_speed,
                                   size_t elapsed_time) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_update"},
                {"id", id},
                {"frame_recorded", frames_written},
                {"writing_speed", writing_speed},
                {"elapsed_time", elapsed_time},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_update event");
  }

  void emit_acquisition_update_event(holoflow::core::SyncCtx &ctx, holocam::RecordingSessionId id,
                                     size_t frames) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "acquisition_update"},
                {"id", id},
                {"frame_acquired", frames},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit acquisition_update event");
  }

  void emit_recording_failed_event(holoflow::core::SyncCtx &ctx, holocam::RecordingSessionId id,
                                   const std::string &message) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_failed"},
                {"id", id},
                {"message", message},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_failed event");
  }

  void on_update_record_path_event(holoflow::core::SyncCtx     &ctx,
                                   const holoflow_event::Event &event) {

    const auto record_path = event.data.value("record_path", std::string{});
    if (record_path.empty()) {
      logger()->error("[CameraTask::on_update_record_path] Rejecting "
                      "update_record_path event with empty path");
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(),
                                  "Cannot update recording path to: empty path");
      return;
    }

    settings_.record_settings->file_path = record_path;
    camera_->update_record_path(record_path);
  }

  void on_start_recording_event(holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {

    if (camera_->recording()) {
      logger()->error("[CameraTask::on_start_recording_event] Ignoring duplicate "
                      "start_recording event");
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(),
                                  "Recording already in progress");
      return;
    }

    const auto record_path = event.data.value("record_path", std::string{});
    if (record_path.empty()) {
      logger()->error("[CameraTask::on_start_recording_event] Rejecting "
                      "start_recording event with empty path");
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(),
                                  "Cannot start recording: empty path");
      return;
    }

    if (settings_.record_settings->recording_count <= 0) {
      const auto message = "Cannot start recording: invalid frame count (" +
                           std::to_string(settings_.record_settings->recording_count) + ")";
      logger()->error("[CameraTask::on_start_recording_event] {}", message);
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(), message);
      return;
    }

    // A finished worker may still be delivering its completion event.
    camera_->stop_raw_record();
    settings_.record_settings->file_path = record_path;
    try {
      camera_->start_raw_record();
    } catch (const std::exception &e) {
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(), e.what());
    }
  }

  void on_stop_recording_event(holoflow::core::SyncCtx &ctx, const holoflow_event::Event &event) {
    (void)event;

    if (!camera_->recording()) {
      logger()->warn("[CameraTask::on_stop_recording_event] Ignoring stop_recording "
                     "event while idle");
      return;
    }

    camera_->stop_raw_record();

    if (!settings_.record_settings->file_path.empty() &&
        std::remove(settings_.record_settings->file_path.c_str()) != 0 && errno != ENOENT) {
      std::error_code ec(errno, std::generic_category());
      logger()->error("[CameraTask::on_stop_recording_event] Failed to remove "
                      "incomplete recording at {}: {}",
                      settings_.record_settings->file_path, ec.message());
      emit_recording_failed_event(ctx, camera_->current_recording_session_id(),
                                  "Failed to remove incomplete recording at " +
                                      settings_.record_settings->file_path + ": " + ec.message());
    }
  }

private:
  inline static holocam::CameraFactory factory_{};
  CameraSettings                       settings_;
  nlohmann::json                       cfg_;
  std::unique_ptr<holocam::CameraBase> camera_;
  std::atomic<bool>                    running_{false};

  holoflow::core::SyncCtx *ctx_{nullptr};

  using Clock = std::chrono::steady_clock;
  Clock::time_point last_event_log_;
};

// -------------------------------------------------------------------------------------------------
// Factory implementation
// -------------------------------------------------------------------------------------------------

namespace {

nlohmann::json get_root(const nlohmann::json &raw_cfg, const CameraSettings &settings) {

  static const std::map<std::string_view, std::string_view> short_name_map = {
      {"AmetekS710EuresysCoaxlinkOcto", "s710"},
      {"AmetekS711EuresysCoaxlinkQSFP+", "s711"},
      {"AmetekS991EuresysCoaxlinkQSFP+", "s991"}};

  return raw_cfg.at(short_name_map.at(settings.name));
}

holoflow::core::TDesc odesc_from_config(const nlohmann::json &raw_cfg,
                                        const CameraSettings &settings) {
  auto       cfg               = get_root(raw_cfg, settings);
  const auto width             = cfg.at("Width").get<uint64_t>();
  const auto pixel_format      = cfg.at("PixelFormat").get<std::string>();
  const auto final_height      = cfg.value("FinalHeight", cfg.at("Height").get<std::uint64_t>());
  const auto buffer_part_count = cfg.at("BufferPartCount");

  static const std::map<std::string_view, holoflow::core::DType> dtype_from_pixel_format = {
      {"Mono8", holoflow::core::DType::U8}, {"Mono16", holoflow::core::DType::U16}};

  holoflow::core::TDesc odesc({buffer_part_count, final_height, width},
                              dtype_from_pixel_format.at(pixel_format),
                              holoflow::core::MemLoc::Host);

  return odesc;
}
} // namespace

holoflow::core::InferResult CameraFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  check(input_descs.empty(), "expected zero input tensors");

  const auto settings = jsettings.get<CameraSettings>();
  const auto raw_cfg  = load_cfg(settings.cfg_path);

  auto odesc = odesc_from_config(raw_cfg, settings);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {true},
      .kind          = holoflow::core::TaskKind::Sync,
      .constness     = holoflow::core::ConstInferenceState::Mutable,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CameraFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;
  check(input_descs.empty(), "expected zero input tensors");

  const auto settings = jsettings.get<CameraSettings>();
  const auto raw_cfg  = load_cfg(settings.cfg_path);

  return std::make_unique<CameraTask>(settings, raw_cfg);
}

std::unique_ptr<holoflow::core::ISyncTask>
CameraFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)ctx;

  auto *old = dynamic_cast<CameraTask *>(old_task.get());
  if (old == nullptr || !input_descs.empty()) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<CameraSettings>();
  const auto raw_cfg  = load_cfg(settings.cfg_path);

  // if record was enabled and is disabled in new settings, we don't need to rebuild
  auto record_settings_changed = false;
  if (old->get_settings().record_settings.has_value() && settings.record_settings.has_value()) {
    record_settings_changed =
        settings.record_settings->requires_rebuild(*old->get_settings().record_settings);
  } else
    record_settings_changed = settings.record_settings.has_value();

  auto config_changed = raw_cfg != old->get_cfg();
  auto need_rebuild   = record_settings_changed || config_changed;

  if (!need_rebuild) {
    old->update_settings(settings);
    old->stop_acquisition();
    old->log_update_lifecycle(false);
    return old_task;
  }

  old->stop_acquisition();
  old->log_update_lifecycle(true);
  old_task.reset(); // destroy old task and release all the buffers
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources

#else

#include <stdexcept>

namespace holotask::sources {

holoflow::core::InferResult CameraFactory::infer(std::span<const holoflow::core::TDesc>,
                                                 const nlohmann::json &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

std::unique_ptr<holoflow::core::ISyncTask>
CameraFactory::create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
                      const holoflow::core::SyncCreateCtx &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

std::unique_ptr<holoflow::core::ISyncTask>
CameraFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                      std::span<const holoflow::core::TDesc>, const nlohmann::json &,
                      const holoflow::core::SyncCreateCtx &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

} // namespace holotask::sources

#endif
