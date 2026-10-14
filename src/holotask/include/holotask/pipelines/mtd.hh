// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holotask/syncs/gps_refinement3d.hh"
#include "holotask/syncs/mtd_field_recovery.hh"
#include "holotask/syncs/mtd_linear3d.hh"
#include "holotask/syncs/rytov_correction.hh"
#include "holotask/syncs/volume_slice.hh"
#include <filesystem>
#include <optional>

namespace holotask::pipelines {

struct MTDDatasetSettings {
  std::filesystem::path                    acquisition_directory;
  int                                      frame_count = 600;
  syncs::MTDFieldRecoverySettings          recovery;
  syncs::RytovCorrectionSettings           rytov;
  syncs::MTDLinear3DReconstructionSettings linear;
  syncs::GPSRefinement3DSettings           gps;
  bool                                     medium_index_assumed = false;
};

/// Saved configuration files take precedence over the active copies.
/// Missing NM uses N0 only when no explicit medium-index override is supplied.
MTDDatasetSettings load_mtd_dataset(const std::filesystem::path &directory,
                                    std::optional<double>        medium_index = std::nullopt);

struct MTDOutputSettings {
  std::filesystem::path                volume_path;
  std::optional<std::filesystem::path> slice_path;
  syncs::VolumeSliceSettings           slice;
  bool                                 overwrite = false;
  std::optional<std::filesystem::path>
      graph_path; ///< Optional runner graph export, reserved against data outputs.
};

/// Reusable offline Holoflow graph: complete scan -> fields -> Rytov -> 3D -> GPS -> NPY.
/// All computation tasks can also be registered and composed separately.
holoflow::core::GraphSpec build_mtd_pipeline(const MTDDatasetSettings &dataset,
                                             const MTDOutputSettings  &output);
void                      register_mtd_tasks(holoflow::core::Registry &registry);

} // namespace holotask::pipelines
