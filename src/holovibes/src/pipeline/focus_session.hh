// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "holoflow/runtime/compiler.hh"
#include "holotask/pipelines/focus_stack.hh"
#include "pipeline/volume_session.hh"

namespace holovibes::pipeline {
class FocusSession : public VolumeSession {
public:
  FocusSession();
  void               reconstruct(const std::filesystem::path                  &path,
                                 const holotask::pipelines::HoloFocusSettings &settings);
  const std::string &graph_dot() const { return graph_dot_; }
  void               advance();
  int                current_frame() const { return current_frame_; }
  int                frame_count() const { return frame_count_; }

private:
  void run_window(const std::filesystem::path                  &path,
                  const holotask::pipelines::HoloFocusSettings &settings, int frame);
  holoflow::core::Registry                           registry_;
  std::unique_ptr<holoflow::runtime::CompilerOutput> compiled_;
  std::filesystem::path                              path_;
  holotask::pipelines::HoloFocusSettings             settings_;
  std::string                                        graph_dot_;
  int                                                current_frame_ = 0, frame_count_ = 0;
};
} // namespace holovibes::pipeline
