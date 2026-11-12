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

#include "s991.hh"

#include "camera_pimpl.hh"

#include "logger.hh"

#ifdef HOLOCAM_HAS_EGRABBER

#include <format>
namespace holocam {
namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[S711] error: {}", msg);
    throw std::invalid_argument("S711 error: " + msg);
  }
}

} // namespace

void CameraS991::configure() {
  using namespace Euresys;
  static const std::map<std::string, std::size_t> pixel_format_map = {
      {"Mono8", 1},
      {"Mono16", 2},
  };

  check(impl_->info().grabbers.size() == grabber_count(),
        std::format("expected {} grabber(s), got {}", grabber_count(),
                    impl_->info().grabbers.size()));

  impl_->reset_grabber_config();

  const auto &cfg = configuration().at(short_name());

  impl_->set_required<RemoteModule>("Banks", cfg.value("Banks", std::string("Bank_AB")));
  impl_->set_required<RemoteModule>("Width", cfg.at("Width").get<int64_t>());

  const auto pixel_format   = cfg.at("PixelFormat").get<std::string>();
  const auto final_height   = cfg.value("FinalHeight", cfg.at("Height").get<std::int64_t>());
  const auto nb_grabbers    = impl_->info().grabbers.size();
  const auto trigger_source = cfg.at("TriggerSource").get<std::string>();
  impl_->set_required<RemoteModule>("Height", final_height / nb_grabbers);
  impl_->set_required<RemoteModule>("PixelFormat", pixel_format);
  impl_->set_required<RemoteModule>("TriggerSelector",
                                    cfg.at("TriggerSelector").get<std::string>());
  impl_->set_required<RemoteModule>("TriggerMode", cfg.at("TriggerMode").get<std::string>());
  impl_->set_required<RemoteModule>("TriggerSource", trigger_source);
  impl_->set_required<RemoteModule>("ExposureTime", cfg.at("ExposureTime").get<float>());
  impl_->set_required<RemoteModule>("GainSelector", cfg.at("GainSelector").get<std::string>());
  impl_->set_required<RemoteModule>("Gain", cfg.at("Gain").get<float>());
  impl_->set_required<RemoteModule>("ShutterMode",
                                    cfg.at("ShutterMode").get<std::string>());

  impl_->set_required<RemoteModule>("AcquisitionFrameRate",
                                    cfg.at("AcquisitionFrameRate").get<std::string>());

  if (cfg.contains("BalanceWhiteMarker")) {
    impl_->set_opt<RemoteModule>("BalanceWhiteMarker",
                                 cfg.at("BalanceWhitemarker").get<std::string>());
  }

  const auto camera_control_method =
      trigger_source == "SWTRIGGER" ? std::string("RC") : std::string("EXTERNAL");
  impl_->set_opt<DeviceModule>("CameraControlMethod", camera_control_method);

  if (trigger_source == "SWTRIGGER") {
    impl_->set_opt<DeviceModule>("ErrorSelector", "All");
    impl_->set_opt<DeviceModule>("ExposureReadoutOverlap", "True");
    if (cfg.contains("CycleMinimumPeriod"))
      impl_->set_opt<DeviceModule>("CycleMinimumPeriod",
                                   cfg.at("CycleMinimumPeriod").get<int64_t>());
  }

  auto line_width =
      cfg.value("LineWidth", cfg.at("Width").get<int64_t>() * pixel_format_map.at(pixel_format));
  auto line_pitch =
      cfg.value("LinePitch", cfg.at("Width").get<int64_t>() * pixel_format_map.at(pixel_format));
  auto stripe_height      = cfg.value("StripeHeight", 8);
  auto stripe_pitch       = cfg.value("StripePitch", 16);
  auto block_height       = cfg.value("BlockHeight", 8);
  auto stripe_arrangement = cfg.value("StripeArrangement", std::string("Geometry_1X_2YM"));

  impl_->set_required<StreamModule>("BufferPartCount", cfg.at("BufferPartCount").get<int64_t>());
  impl_->set_required<StreamModule>("LineWidth", line_width);
  impl_->set_required<StreamModule>("LinePitch", line_pitch);
  impl_->set_required<StreamModule>("StripeHeight", stripe_height);
  impl_->set_required<StreamModule>("StripePitch", stripe_pitch);
  impl_->set_required<StreamModule>("BlockHeight", block_height);
  impl_->set_required<StreamModule>("StripeArrangement", stripe_arrangement);

  auto offsets = cfg.at("Offsets").get<std::vector<int64_t>>();
  impl_->set_offset(offsets[0], offsets[1]);
}
} // namespace holocam

#else
namespace holocam {
void CameraS711::configure() {}
} // namespace holocam
#endif
