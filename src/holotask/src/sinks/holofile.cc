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

#include "holotask/sinks/holofile.hh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "bug.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::sinks {

void to_json(nlohmann::json &j, const HolofileSettings &hs) {
  j = nlohmann::json{
      {"path", hs.path},
      {"count", hs.count},
      {"pipeline_settings", hs.pipeline_settings},
  };
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("count").get_to(hs.count);
  j.at("pipeline_settings").get_to(hs.pipeline_settings);
}

Holofile::Holofile(const HolofileSettings &settings, RecordingGeometry geometry)
    : settings_(settings), geometry_(geometry) {}

Holofile::~Holofile() {
  if (!is_recording()) {
    return;
  }

  if (frames_written_ >= settings_.count) {
    return;
  }

  logger()->warn("[Holofile] Recording stopped before completion ({}/{}), deleting {}",
                 frames_written_, settings_.count, settings_.path);

  writer_.reset();
  bool removed = std::remove(settings_.path.c_str()) == 0;
  HOLOVIBES_CHECK(removed, "Failed to delete incomplete recording at {}", settings_.path);
}

[[nodiscard]] holoflow::core::OpResult Holofile::execute(holoflow::core::SyncCtx &ctx) {
  handle_events(ctx);

  if (is_recording() && frames_written_ >= settings_.count) {
    finalize_recording(ctx);
    return holoflow::core::OpResult::Ok;
  }

  if (!is_recording()) {
    return holoflow::core::OpResult::Ok;
  }

  auto  remaining  = settings_.count - frames_written_;
  auto  batch_size = static_cast<int>(ctx.inputs[0].desc.shape[0]);
  auto  to_write   = std::min(remaining, batch_size);
  auto *idata      = reinterpret_cast<const uint8_t *>(ctx.inputs[0].data());
  writer_->write_frames(idata, to_write);
  frames_written_ += to_write;

  if (frames_written_ >= settings_.count) {
    finalize_recording(ctx);
  }

  return holoflow::core::OpResult::Ok;
}

void Holofile::handle_events(holoflow::core::SyncCtx &ctx) {
  if (ctx.event_reader == nullptr) {
    return;
  }

  while (true) {
    auto event = ctx.event_reader->try_pop();
    if (!event.has_value()) {
      break;
    }

    constexpr auto EXPECTED_DIRECTION = holoflow_event::EventDirection::ToNode;
    HOLOVIBES_CHECK(event->direction == EXPECTED_DIRECTION, "Unexpected event direction");
    auto type      = event->data["type"].get<std::string>();
    auto new_path  = event->data["record_path"].get<std::string>();
    settings_.path = new_path;

    if (type == "start_recording") {
      HOLOVIBES_CHECK(!is_recording(), "Received start_recording event while already recording");
      start_recording(settings_.path, settings_.count);
    }

    else if (type == "stop_recording") {
      HOLOVIBES_CHECK(is_recording(), "Received stop_recording event while not recording");
      writer_.reset();
      bool removed = std::remove(settings_.path.c_str()) == 0;
      HOLOVIBES_CHECK(removed, "Failed to delete incomplete recording at {}", settings_.path);
      frames_written_ = 0;
    }

    else {
      HOLOVIBES_BUG("Unknown event type: {}", type);
    }
  }
}

void Holofile::start_recording(const std::string &path, int count) {
  HOLOVIBES_CHECK(!path.empty(), "Cannot start recording: empty path");
  HOLOVIBES_CHECK(count > 0, "Cannot start recording: count must be positive (got {})", count);
  HOLOVIBES_CHECK(!is_recording(), "Cannot start recording: already recording");

  frames_written_ = 0;
  auto footer     = holofile::Footer{.pipeline_settings = settings_.pipeline_settings};
  writer_.emplace(path, make_header(count), footer);
}

void Holofile::finalize_recording(holoflow::core::SyncCtx &ctx) {
  HOLOVIBES_CHECK(is_recording(), "Cannot finalize recording: not recording");
  HOLOVIBES_CHECK(frames_written_ == settings_.count,
                  "Cannot finalize recording: incomplete ({} / {})", frames_written_,
                  settings_.count);

  writer_->write_footer();

  auto event_data = nlohmann::json{
      {"type", "recording_finished"},
      {"path", settings_.path},
      {"frames_written", frames_written_},
  };

  auto event = holoflow_event::Event{
      .direction = holoflow_event::EventDirection::ToUi,
      .node_id   = "",
      .data      = std::move(event_data),
      .ts        = std::chrono::steady_clock::now(),
  };

  bool pushed = ctx.event_writer->try_push(std::move(event));
  HOLOVIBES_CHECK(pushed, "Failed to emit recording_finished event");
  reset_recording_state();
}

holofile::Header Holofile::make_header(int count) const {
  const auto frame_count = static_cast<uint32_t>(count);
  const auto data_size   = static_cast<uint64_t>(frame_count) * geometry_.frame_height *
                         geometry_.frame_width * geometry_.bits_per_pixel / 8;
  return holofile::Header{
      .magic_number       = holofile::Header::MAGIC_NUMBER_LE,
      .version            = holofile::Header::CURRENT_VERSION,
      .bits_per_pixel     = geometry_.bits_per_pixel,
      .frame_width        = geometry_.frame_width,
      .frame_height       = geometry_.frame_height,
      .frame_count        = frame_count,
      .data_size_in_bytes = data_size,
      .endianness         = holofile::Header::LITTLE_ENDIAN,
  };
}

[[nodiscard]] bool Holofile::is_recording() const noexcept { return writer_.has_value(); }

void Holofile::reset_recording_state() {
  writer_.reset();
  frames_written_ = 0;
}

holoflow::core::InferResult
HolofileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[HolofileFactory::infer] error: {}", msg);
      throw std::invalid_argument("HolofileFactory inference error: " + msg);
    }
  };
  std::set<holoflow::core::DType> supported_dtypes = {
      holoflow::core::DType::U8,
      holoflow::core::DType::U16,
  };

  auto settings = jsettings.get<HolofileSettings>();

  // Validate
  check(input_descs.size() == 1, "Holofile task must have exactly one output");
  auto &idesc = input_descs[0];
  check(idesc.shape.size() == 3, "Output tensor must have rank 3");
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Output tensor must be in Host memory");
  check(supported_dtypes.contains(idesc.dtype),
        "Unsupported input dtype: " + std::to_string(static_cast<int>(idesc.dtype)));

  // Success
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx &) const {
  std::map<holoflow::core::DType, uint8_t> dtype_to_bpp = {
      {holoflow::core::DType::U8, static_cast<uint8_t>(8)},
      {holoflow::core::DType::U16, static_cast<uint8_t>(16)},
  };

  // Validate
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<HolofileSettings>();
  auto &idesc    = input_descs[0];

  // Setup writer
  auto                        dtype        = idesc.dtype;
  auto                        bpp          = dtype_to_bpp.at(dtype);
  auto                        frame_width  = static_cast<uint32_t>(idesc.shape[2]);
  auto                        frame_height = static_cast<uint32_t>(idesc.shape[1]);
  Holofile::RecordingGeometry geometry{
      .bits_per_pixel = bpp,
      .frame_width    = frame_width,
      .frame_height   = frame_height,
  };

  // Success
  auto *task = new Holofile(settings, geometry);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::sinks
