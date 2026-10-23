// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holotask/syncs/focus_stack.hh"
#include <filesystem>
#include <optional>
#include <string>

namespace holotask::pipelines {
struct HoloFocusSettings {
  int                       frame = 0;
  syncs::FocusStackSettings optics;
  std::string               method           = "holovibes_stft";
  int                       window           = 512;
  double                    temporal_focus_m = 0.002;
  double                    temporal_pitch_m = 12e-6;
  bool                      temporal_filter  = true;
  int                       filter_inner = 20, filter_outer = 362;
  bool                      playback     = true;
  int                       playback_fps = 20; ///< Maximum reconstructed windows per second.
  bool                      operator==(const HoloFocusSettings &) const = default;
};
void to_json(nlohmann::json &, const HoloFocusSettings &);
void from_json(const nlohmann::json &, HoloFocusSettings &);

struct HoloCalibration {
  int                   width = 0, height = 0, frames = 0;
  std::optional<double> wavelength_m, pixel_pitch_x_m, pixel_pitch_y_m, propagation_m;
  std::string           summary;
};

/// Read only the header/footer; saved optics are estimates, not a verified calibration.
HoloCalibration           inspect_holo_calibration(const std::filesystem::path &path);
holoflow::core::GraphSpec build_focus_stack_pipeline(const std::filesystem::path &path,
                                                     const HoloFocusSettings     &settings);
void validate_holo_focus_settings(const HoloFocusSettings &settings, int width, int height,
                                  int frames);
void register_focus_stack_tasks(holoflow::core::Registry &registry);
} // namespace holotask::pipelines
