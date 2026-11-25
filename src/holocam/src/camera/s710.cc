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

#include "s710.hh"

#include "camera_pimpl.hh"

#ifdef HOLOCAM_HAS_EGRABBER

namespace holocam {

void CameraS710::configure(Impl *impl, const nlohmann::json &cfg) {
  const std::map<std::size_t, std::string> banks_map = {
      {2, "Banks_AB"},
      {4, "Banks_ABCD"},
  };

  // Extract configuration parameters
  const auto width                 = cfg.at("Width").get<std::size_t>();
  const auto height                = cfg.at("Height").get<std::size_t>();
  const auto nb_grabbers           = impl->info().grabbers.size();
  const auto pixel_format          = cfg.at("PixelFormat").get<std::string>();
  const auto trigger_source        = cfg.at("TriggerSource").get<std::string>();
  const auto trigger_mode          = cfg.at("TriggerMode").get<std::string>();
  const auto exposure_time         = cfg.at("ExposureTime").get<std::size_t>();
  const auto cycle_min_period      = cfg.at("CycleMinimumPeriod").get<std::size_t>();
  const auto gain_selector         = cfg.at("GainSelector").get<std::string>();
  const auto gain                  = cfg.at("Gain").get<float>();
  const auto balance_white_marker  = cfg.at("BalanceWhiteMarker").get<std::string>();
  const auto flat_field_correction = cfg.at("FlatFieldCorrection").get<std::string>();
  const auto buffer_part_count     = cfg.at("BufferPartCount").get<std::size_t>();

  // Computed parameters
  const auto stripe_height            = height / nb_grabbers;
  const auto stripe_pitch             = height;
  const auto block_height             = stripe_height;
  const auto camera_control_method    = "RC";
  const auto exposure_readout_overlap = "True";
  const auto error_selector           = "All";
  const auto stripe_arrangement       = "Geometry_1X_2YM";
  const auto statistics_sampling_sel  = "LastSecond";
  const auto lut_configuration        = "M_10x8";
  const auto banks                    = banks_map.at(nb_grabbers);

  try {
    // Device-level master configuration
    impl->reset_grabber_config();
    impl->set_required<Euresys::RemoteModule>("Banks", banks);
    impl->set_required<Euresys::DeviceModule>("CameraControlMethod", camera_control_method);

    if (trigger_source == "SWTRIGGER") {
      impl->set_required<Euresys::DeviceModule>("ErrorSelector", error_selector);
      impl->set_required<Euresys::DeviceModule>("CycleMinimumPeriod", cycle_min_period);
      impl->set_required<Euresys::DeviceModule>("ExposureReadoutOverlap", exposure_readout_overlap);
    }

    // Remote configuration
    impl->set_required<Euresys::RemoteModule>("Width", width);
    impl->set_required<Euresys::RemoteModule>("Height", height / nb_grabbers);
    impl->set_required<Euresys::RemoteModule>("PixelFormat", pixel_format);
    impl->set_required<Euresys::RemoteModule>("TriggerMode", trigger_mode);
    impl->set_required<Euresys::RemoteModule>("TriggerSource", trigger_source);
    impl->set_required<Euresys::RemoteModule>("ExposureTime", exposure_time);
    impl->set_required<Euresys::RemoteModule>("BalanceWhiteMarker", balance_white_marker);
    impl->set_required<Euresys::RemoteModule>("GainSelector", gain_selector);
    impl->set_required<Euresys::RemoteModule>("Gain", gain);
    impl->set_required<Euresys::RemoteModule>("FlatFieldCorrection", flat_field_correction);

    // Stream configuration
    impl->set_required<Euresys::StreamModule>("StripeArrangement", stripe_arrangement);
    impl->set_required<Euresys::StreamModule>("LinePitch", width);
    impl->set_required<Euresys::StreamModule>("LineWidth", width);
    impl->set_required<Euresys::StreamModule>("StripeHeight", stripe_height);
    impl->set_required<Euresys::StreamModule>("StripePitch", stripe_pitch);
    impl->set_required<Euresys::StreamModule>("BlockHeight", block_height);
    impl->set_required<Euresys::StreamModule>("LUTConfiguration", lut_configuration);
    impl->set_required<Euresys::StreamModule>("BufferPartCount", buffer_part_count);
    impl->set_required<Euresys::StreamModule>("StatisticsSamplingSelector",
                                              statistics_sampling_sel);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(format_genapi_error(e));
  }
}
} // namespace holocam

#else
namespace holocam {
void CameraS710::configure(Impl *impl, const nlohmann::json &cfg) {
  (void)impl;
  (void)cfg;
}
} // namespace holocam
#endif
