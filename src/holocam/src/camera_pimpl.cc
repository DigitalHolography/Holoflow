// Copyright 2026 Digital Holography Foundation
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

#pragma once

#include "camera_pimpl.hh"

#include "logger.hh"

#include <cassert>
#include <format>

#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("error: {}", msg);
    throw std::invalid_argument("error: " + msg);
  }
}

std::optional<Euresys::EGrabberCameraInfo> find_camera(Euresys::EGenTL   &gentl,
                                                       const std::string &model_name) {
  using namespace Euresys;

  EGrabberDiscovery discovery(gentl);
  discovery.discover();

  for (int i = 0; i < discovery.cameraCount(); ++i) {
    auto info = discovery.cameras(i);
    auto g    = EGrabber<>(info);

    try {
      if (g.getString<RemoteModule>("DeviceModelName") == model_name) {
        return info;
      }
    } catch (const Euresys::genapi_error &) {
      try {
        if (g.getString<DeviceModule>("DeviceModelName") == model_name) {
          return info;
        }
      } catch (const Euresys::genapi_error &) {
      }
    }
  }

  return std::nullopt;
}

} // namespace

CameraBase::Impl::Impl(const char *model_name, const char *short_name, const nlohmann::json &config,
                       const std::optional<RecordOptions> &record_options,
                       size_t expected_grabber_count, size_t buffer_size, size_t buffer_count)
    : config_{config}, gentl_{std::make_unique<Euresys::EGenTL>()},
      buffers_{curaii::make_unique_host_ptr<uint8_t>(buffer_size *
                                                     (buffer_count + safety_buffer_count))},
      total_buffer_size_{buffer_size * buffer_count}, buffer_size_{buffer_size},
      nb_buffers_{buffer_count} {
  auto camera_info = find_camera(*gentl_, model_name);
  check(camera_info.has_value(), std::format("could not find {} camera", model_name));
  info_ = *camera_info;

  assert(expected_grabber_count > 0);
  assert(expected_grabber_count <= 2);

  auto root          = config.at(short_name);
  buffer_part_count_ = root.at("BufferPartCount").get<std::size_t>();

  const auto bank_a_index = find_grabber_index_for_bank(0);
  if (expected_grabber_count == 1)
    grabber_ = std::make_unique<Grabber>(buffer_part_count_, camera_info->grabbers[bank_a_index]);
  else {
    const auto bank_b_index = find_grabber_index_for_bank(1);
    grabber_ = std::make_unique<Grabber>(buffer_part_count_, camera_info->grabbers[bank_a_index],
                                         camera_info->grabbers[bank_b_index]);
  }

  buffer_queue_ = std::make_unique<CameraBufferQueue>(
      nb_buffers_, [this](const CameraFrame &frame) { grabber_->requeue(frame.banks); });

  if (record_options.has_value()) {
    init_record_context(short_name, *record_options, buffer_size * buffer_count, config);
  }
}

void CameraBase::Impl::init_record_context(const char *short_name, const RecordOptions &options,
                                           size_t total_buffer_size, const nlohmann::json &config) {
  recording_context_.reset();
  recording_context_.emplace();

  auto       root         = config.at(short_name);
  const auto width        = root.at("Width").get<int64_t>();
  const auto final_height = root.value("FinalHeight", root.at("Height").get<std::int64_t>());
  const auto pixel_format = root.at("PixelFormat").get<std::string>();
  static const std::map<std::string, std::size_t> pixel_format_map_bits = {
      {"Mono8", 8},
      {"Mono16", 16},
  };

  auto g = RecordingGeometry(static_cast<uint8_t>(pixel_format_map_bits.at(pixel_format)), width,
                             final_height);

  const auto &record_settings  = std::get<RecordSettings>(options);
  const auto &record_callbacks = std::get<RecordCallbacks>(options);
  recording_buffers_           = curaii::make_unique_host_ptr<std::byte>(total_buffer_size *
                                                                         record_settings.record_queue_count);
  recording_context_->invalidate(
      record_settings, *buffer_queue_.get(), total_buffer_size, recording_buffers_.get(), g,
      record_callbacks.poll_metrics, record_callbacks.acquisition_metrics,
      record_callbacks.success_callback, record_callbacks.failure_callback,
      record_callbacks.flushing_finished);
}

void CameraBase::Impl::cancel_all_records() {
  if (recording_context_.has_value()) {
    recording_context_->stop_all_record();
    current_recording_session_ = nullptr;
  }
}
RecordingSessionId CameraBase::Impl::current_recording_session_id() {
  if (current_recording_session_ != nullptr)
    return current_recording_session_->id();

  return 0;
}

bool CameraBase::Impl::recording() { return recording_.load(std::memory_order_acquire); }

size_t CameraBase::Impl::find_grabber_index_for_bank(size_t id) {
  using namespace Euresys;

  for (std::size_t i = 0; i < info_.grabbers.size(); ++i) {
    EGrabber<> g(info_.grabbers[i]);
    if (g.getInteger<RemoteModule>("ConnectedBankID") == static_cast<int64_t>(id)) {
      return i;
    }
  }

  throw std::runtime_error(std::format("could not find grabber for ConnectedBankID={}", id));
}

void CameraBase::Impl::stop_acquisition() {
  if (acquisition_thread_.has_value() && acquisition_thread_->joinable()) {
    cancel_all_records();
    acquisition_thread_->request_stop();
    acquisition_thread_->join();
    acquisition_thread_.reset();
    grabber_->stop_if_running();
    acquiring_.store(false, std::memory_order_release);
    logger()->info("[CameraBase::Impl::stop_acquisition] acquisition stopped");
  }
}

void CameraBase::Impl::start_acquisition() {
  stop_acquisition();
  grabber_->start_if_not_running();
  acquisition_thread_.emplace([this](std::stop_token cancelled) { acquisition_loop(cancelled); });
  acquiring_.store(true, std::memory_order_release);
  logger()->info("[CameraBase::Impl::start_acquisition] acquisition started");
}

std::byte *CameraBase::Impl::get_frame(std::atomic<bool> *cancelled) {
  const auto *frame = buffer_queue_->read_a(cancelled);
  return frame != nullptr ? frame->base : nullptr;
}

void CameraBase::Impl::release_frame() { buffer_queue_->release_a(); }

void CameraBase::Impl::acquisition_loop(std::stop_token cancelled) {

  logger()->info("[CameraBase::Impl::acquisition_loop] acquisition loop started");
  try {
    buffer_queue_->open();
    FrameBufferPair frame{};
    while (!cancelled.stop_requested()) {
      try {
        frame = grabber_->pop(pop_timeout_);
      } catch (const Euresys::gentl_error &e) {
        if (e.gc_err == GenTL::GC_ERR_TIMEOUT) {
          logger()->debug("[CameraBase::Impl::acquisition_loop] grabber pop timed out");
          continue;
        }
        logger()->error("[CameraBase::Impl::acquisition_loop] error occurred while poping frame "
                        "from grabber: {}",
                        e.what());
        throw;
      }

      if (cancelled.stop_requested()) {
        grabber_->requeue(frame);
        break;
      }

      auto base = grabber_->validate_buffer_data(frame);
      if (!base) {
        grabber_->requeue(frame);
        check_for_record_failure();
        continue;
      }

      CameraFrame camera_frame = {.banks = frame, .base = base};
      auto        result       = CameraBufferQueue::PushResult::Closed;
      result                   = buffer_queue_->try_push(camera_frame);

      if (result != CameraBufferQueue::PushResult::Accepted) {
        check_for_record_failure();
        if (log_due(last_requeue_log_)) {
          logger()->warn("[CameraBase::Impl::acquisition_loop] buffer_queue is full or closed "
                         "(size={}/{}): dropping frame",
                         buffer_queue_->size(), buffer_queue_->capacity());
        }
        grabber_->requeue(frame);
      }
    }
    logger()->info("[CameraBase::Impl::acquisition_loop] acquisition loop stopped");
  } catch (const std::exception &e) {
    logger()->error("[CameraBase::Impl::acquisition_loop] acquisition loop stopped due to: {}",
                    e.what());
  } catch (...) {
    logger()->error(
        "[CameraBase::Impl::acquisition_loop] acquisition loop stopped due to unknown error");
  }
  cancel_all_records();
  buffer_queue_->close();
}

void CameraBase::Impl::check_for_record_failure() {
  if (recording_context_.has_value() &&
      recording_context_->recording.load(std::memory_order_acquire)) {
    auto rem = current_recording_session_->remaining_to_record() / buffer_part_count_;
    if (rem > buffer_queue_->size()) {
      current_recording_session_->stop();
      logger()->error("[CameraBase::Impl::check_for_record_failure] record on queue {} failed "
                      "because a frame has been rejected while still acquiring for record",
                      current_recording_session_->id());
    }
  }
}

bool CameraBase::Impl::start_raw_record() {
  assert(recording_context_.has_value());

  if (recording_context_->session_still_recording()) {
    logger()->error("[Camera::Impl::start_raw_record] a session is still "
                    "recording frames from camera");
    return false;
  }

  auto session_it = recording_context_->find_next_free_session();
  if (session_it == recording_context_->sessions.end()) {
    logger()->error(
        "[AmetekS711EuresysCoaxlinkQSFP:start_raw_record] cannot find a free recording session");
    return false;
  }

  auto &session              = *session_it;
  current_recording_session_ = &session;
  session.start(recording_context_->settings, buffer_part_count_, recording_context_->recording);
  logger()->info("[CameraBase::Impl::start_raw_record] started new record on queue {} to {}",
                 session.id(), recording_context_->settings.file_path);
  return true;
}

void CameraBase::Impl::stop_raw_record() {
  auto session = current_recording_session_;
  if (session) {
    session->stop();
    logger()->info("[CameraBase::Impl::stop_raw_record] stopped record on queue {}",
                   session->id());
    current_recording_session_ = nullptr;
    assert(!recording_context_->recording.load(std::memory_order_acquire));
  }
}

void CameraBase::Impl::set_recording_options(std::optional<RecordOptions> o,
                                             const char                  *short_name) {
  cancel_all_records();
  if (o.has_value()) {
    init_record_context(short_name, *o, total_buffer_size_, config_);
  } else {
    recording_context_.reset();
    buffers_.reset();
    recording_buffers_count_ = 0;
    recording_buffers_size_  = 0;
  }
}

void CameraBase::Impl::update_record_path(const std::string &path) {
  assert(recording_context_.has_value());

  recording_context_->settings.file_path = path;
}
} // namespace holocam

#endif