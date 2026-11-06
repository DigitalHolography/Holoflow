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

#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

CameraBase::CameraBase(const nlohmann::json &config) { std::make_unique<CameraBase::Impl>(config); }

std::size_t CameraBase::bank_count() const noexcept { return impl_->bank_count(); }
std::size_t CameraBase::frame_width() const noexcept { return impl_->frame_width(); }
std::size_t CameraBase::frame_height() const noexcept { return impl_->frame_height(); }
std::size_t CameraBase::buffer_part_count() const noexcept { return impl_->buffer_part_count(); }

void CameraBase::set_recording_settings(std::optional<RecordingSettings> s) {
  impl_->set_recording_settings(s);
}

bool CameraBase::start_acquisition() { return impl_->start_acquisition(); }
void CameraBase::stop_acquisition() { impl_->stop_acquisition(); }
bool CameraBase::start_raw_record() { return impl_->start_raw_record(); }
void CameraBase::stop_raw_record() { impl_->stop_raw_record(); }

std::byte *CameraBase::get_frame(std::atomic<bool> *cancelled) {
  return impl_->get_frame(cancelled);
}

void CameraBase::release_frame() { impl_->release_frame(); }

const nlohmann::json &CameraBase::configuration() const noexcept { return impl_->configuration(); }

const CameraMetrics &CameraBase::metrics() const noexcept { return impl_->metrics(); }

} // namespace holocam

#endif