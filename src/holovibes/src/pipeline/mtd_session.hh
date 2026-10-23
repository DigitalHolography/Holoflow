// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/tensor.hh"
#include "pipeline/settings.hh"
#include "pipeline/volume_session.hh"
#include <QByteArray>
#include <filesystem>
#include <string>

namespace holovibes::pipeline {

/// Worker-thread-owned reconstruction cache. Only 2D planes are transferred for display.
class MTDSession : public VolumeSession {
public:
  bool               matches(const std::filesystem::path     &directory,
                             const MTDReconstructionSettings &settings) const;
  void               reconstruct(const std::filesystem::path     &directory,
                                 const MTDReconstructionSettings &settings);
  int                dimension() const { return dimension_; }
  double             voxel_pitch_m() const { return voxel_pitch_m_; }
  const std::string &graph_dot() const { return graph_dot_; }

private:
  std::filesystem::path     directory_;
  MTDReconstructionSettings settings_;
  int                       dimension_     = 0;
  double                    voxel_pitch_m_ = 0;
  std::string               graph_dot_;
};
} // namespace holovibes::pipeline
