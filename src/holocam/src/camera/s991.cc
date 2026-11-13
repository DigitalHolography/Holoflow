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
    logger()->error("[S991] error: {}", msg);
    throw std::invalid_argument("S991 error: " + msg);
  }
}

} // namespace

void CameraS991::configure(Impl *impl, const nlohmann::json &cfg) {
  using namespace Euresys;
  static const std::map<std::string, std::size_t> pixel_format_map = {
      {"Mono8", 1},
      {"Mono16", 2},
  };

  constexpr size_t grabber_count = 2;
  check(impl->info().grabbers.size() == grabber_count,
        std::format("expected {} grabber(s), got {}", grabber_count, impl->info().grabbers.size()));

  impl->reset_grabber_config();

  impl->set_required<RemoteModule>("Banks", cfg.value("Banks", std::string("Banks_AB")));
  impl->set_required<RemoteModule>("Width", cfg.at("Width").get<int64_t>());

  const auto pixel_format   = cfg.at("PixelFormat").get<std::string>();
  const auto final_height   = cfg.value("FinalHeight", cfg.at("Height").get<std::int64_t>());
  const auto nb_grabbers    = impl->info().grabbers.size();
  const auto trigger_source = cfg.at("TriggerSource").get<std::string>();
  impl->set_required<RemoteModule>("Height", final_height / static_cast<int64_t>(nb_grabbers));
  impl->set_required<RemoteModule>("PixelFormat", pixel_format);
  impl->set_required<RemoteModule>("TriggerSelector", cfg.at("TriggerSelector").get<std::string>());
  impl->set_required<RemoteModule>("TriggerMode", cfg.at("TriggerMode").get<std::string>());
  impl->set_required<RemoteModule>("TriggerSource", trigger_source);
  impl->set_required<RemoteModule>("ExposureTime", cfg.at("ExposureTime").get<float>());
  impl->set_required<RemoteModule>("GainSelector", cfg.at("GainSelector").get<std::string>());
  impl->set_required<RemoteModule>("Gain", cfg.at("Gain").get<float>());
  impl->set_required<RemoteModule>("ShutterMode", cfg.at("ShutterMode").get<std::string>());

  const auto acquisition_frame_rate = cfg.at("AcquisitionFrameRate").get<size_t>();
  impl->set_required<RemoteModule>("AcquisitionFrameRate",
                                   std::format("{}", acquisition_frame_rate));

  if (cfg.contains("BalanceWhiteMarker")) {
    impl->set_opt<RemoteModule>("BalanceWhiteMarker",
                                cfg.at("BalanceWhiteMarker").get<std::string>());
  }

  const auto camera_control_method =
      trigger_source == "SWTRIGGER" ? std::string("RC") : std::string("EXTERNAL");
  impl->set_opt<DeviceModule>("CameraControlMethod", camera_control_method);

  if (trigger_source == "SWTRIGGER") {
    impl->set_opt<DeviceModule>("ErrorSelector", "All");
    impl->set_opt<DeviceModule>("ExposureReadoutOverlap", "True");
    if (cfg.contains("CycleMinimumPeriod"))
      impl->set_opt<DeviceModule>("CycleMinimumPeriod",
                                  cfg.at("CycleMinimumPeriod").get<int64_t>());
  }

  auto line_width =
      cfg.value("LineWidth", cfg.at("Width").get<int64_t>() * pixel_format_map.at(pixel_format));
  auto line_pitch =
      cfg.value("LinePitch", cfg.at("Width").get<int64_t>() * pixel_format_map.at(pixel_format));
  auto stripe_height      = cfg.value("StripeHeight", 4);
  auto stripe_pitch       = cfg.value("StripePitch", 8);
  auto block_height       = cfg.value("BlockHeight", 4);
  auto stripe_arrangement = cfg.value("StripeArrangement", std::string("Geometry_1X_1Y"));

  impl->set_required<StreamModule>("BufferPartCount", cfg.at("BufferPartCount").get<int64_t>());
  impl->set_required<StreamModule>("LineWidth", line_width);
  impl->set_required<StreamModule>("LinePitch", line_pitch);
  impl->set_required<StreamModule>("StripeHeight", stripe_height);
  impl->set_required<StreamModule>("StripePitch", stripe_pitch);
  impl->set_required<StreamModule>("BlockHeight", block_height);
  impl->set_required<StreamModule>("StripeArrangement", stripe_arrangement);

  auto offsets = cfg.at("Offsets").get<std::vector<int64_t>>();
  impl->set_offset(offsets[0], offsets[1]);
}
} // namespace holocam

#else
namespace holocam {
void CameraS711::configure(Impl *impl, const nlohmann::json &cfg) {
  (void)impl;
  (void)cfg;
}
} // namespace holocam
#endif