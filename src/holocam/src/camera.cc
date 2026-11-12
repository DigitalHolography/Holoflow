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

#include "holocam/camera.hh"
#include "camera_pimpl.hh"

#undef max
#include <utility>

#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

namespace {
void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("error: {}", msg);
    throw std::invalid_argument("error: " + msg);
  }
}

} // namespace
CameraBase::CameraBase(const nlohmann::json &config, std::optional<RecordOptions> record_options)
    : impl_{nullptr} {
  static const std::map<std::string, std::size_t> pixel_format_map = {
      {"Mono8", 1},
      {"Mono16", 2},
  };

  auto root              = config.at(short_name());
  auto nb_buffers        = root.at("NbBuffers").get<std::size_t>();
  auto buffer_part_count = root.at("BufferPartCount").get<std::size_t>();
  auto final_height      = root.value("FinalHeight", root.at("Height").get<std::size_t>());
  auto width             = root.at("Width").get<std::size_t>();
  auto pixel_format      = root.at("PixelFormat").get<std::string>();
  check(pixel_format_map.contains(pixel_format), "unsupported PixelFormat: " + pixel_format);
  auto bytes_per_pixel = pixel_format_map.at(pixel_format);

  auto   frame_size           = final_height * width * bytes_per_pixel;
  auto   buffer_size          = buffer_part_count * frame_size;
  size_t record_buffer_needed = 0;
  if (record_options.has_value()) {
    const auto &record_settings = std::get<RecordSettings>(*record_options);
    record_buffer_needed        = record_settings.recording_count / buffer_part_count;
    if (record_buffer_needed % nb_buffers != 0) {
      record_buffer_needed += nb_buffers - (record_buffer_needed % nb_buffers);
    }
  }

  constexpr size_t safety_buffer_count = 16;
  auto             buffer_count = std::max(record_buffer_needed, nb_buffers) + safety_buffer_count;
  auto             total_size   = buffer_count * buffer_size;
  impl_ = std::make_unique<Impl>(model_name(), short_name(), config, record_options,
                                 grabber_count(), total_size);
  configure();
}

CameraBase::~CameraBase() {}

std::size_t CameraBase::bank_count() const noexcept { return impl_->bank_count(); }
std::size_t CameraBase::frame_width() const noexcept { return impl_->frame_width(); }
std::size_t CameraBase::frame_height() const noexcept { return impl_->frame_height(); }
std::size_t CameraBase::buffer_part_count() const noexcept { return impl_->buffer_part_count(); }

void CameraBase::set_record_options(std::optional<RecordOptions> s) {
  impl_->set_recording_options(s);
}

void CameraBase::update_record_path(const std::string &path) { impl_->update_record_path(path); }

void CameraBase::start_acquisition() { impl_->start_acquisition(); }
void CameraBase::stop_acquisition() { impl_->stop_acquisition(); }
bool CameraBase::start_raw_record() { return impl_->start_raw_record(); }
void CameraBase::stop_raw_record() { impl_->stop_raw_record(); }

std::byte *CameraBase::get_frame(std::atomic<bool> *cancelled) {
  return impl_->get_frame(cancelled);
}

void CameraBase::release_frame() { impl_->release_frame(); }

const nlohmann::json &CameraBase::configuration() const noexcept { return impl_->configuration(); }

} // namespace holocam

#else

namespace holocam {
CameraBase::CameraBase(nlohmann::json config, std::optional<RecordingSettings> record_settings) {
  (void)config;
  (void)record_settings;
  throw std::runtime_error("This build does not support EGrabber");
}

std::size_t CameraBase::bank_count() const noexcept { return 0; }
std::size_t CameraBase::frame_width() const noexcept { return 0; }
std::size_t CameraBase::frame_height() const noexcept { return 0; }
std::size_t CameraBase::buffer_part_count() const noexcept { return 0; }

void CameraBase::set_recording_settings(std::optional<RecordingSettings> s) { (void)s; }
void CameraBase::update_record_path(const std : string &path) { (void)path; }

vod  CameraBase::start_acquisition() {}
void CameraBase::stop_acquisition() {}
bool CameraBase::start_raw_record() { return false; }
void CameraBase::stop_raw_record() {}

std::byte *CameraBase::get_frame(std::atomic<bool> *cancelled) {
  (void)cancelled;
  return nullptr;
}

void CameraBase::release_frame() {}

const nlohmann::json &CameraBase::configuration() const noexcept { return {}; }

} // namespace holocam
#endif