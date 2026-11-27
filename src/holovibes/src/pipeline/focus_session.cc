// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline/focus_session.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_display.hh"
#include <boost/range/iterator_range.hpp>
#include <stdexcept>

namespace holovibes::pipeline {
FocusSession::FocusSession() { holotask::pipelines::register_focus_stack_tasks(registry_); }

void FocusSession::reconstruct(const std::filesystem::path                  &path,
                               const holotask::pipelines::HoloFocusSettings &settings) {
  auto key         = settings;
  key.playback     = settings_.playback;
  key.playback_fps = settings_.playback_fps;
  if (compiled_ && width() > 0 && path_ == path && settings_ == key) {
    settings_ = settings;
    return;
  }
  const auto calibration = holotask::pipelines::inspect_holo_calibration(path);
  run_window(path, settings, settings.frame);
  path_        = path;
  settings_    = settings;
  frame_count_ = calibration.frames;
}

void FocusSession::advance() {
  if (settings_.method != "holovibes_stft" || !compiled_)
    throw std::logic_error("STFT playback requires a compiled temporal volume");
  const int next = frame_count_ - current_frame_ >= 2 * settings_.window
                       ? current_frame_ + settings_.window
                       : settings_.frame;
  run_window(path_, settings_, next);
}

void FocusSession::run_window(const std::filesystem::path                  &path,
                              const holotask::pipelines::HoloFocusSettings &settings, int frame) {
  auto window                      = settings;
  window.frame                     = frame;
  const auto                  spec = holotask::pipelines::build_focus_stack_pipeline(path, window);
  holoflow::runtime::Compiler compiler(registry_,
                                       {.verbose_tracing = false, .enable_profiling = false});
  compiled_ = compiler.compile(spec, std::move(compiled_));
  holoflow::runtime::Scheduler scheduler(compiled_->graph, compiled_->sections,
                                         compiled_->resources);
  scheduler.start();
  scheduler.wait();
  auto &resources = compiled_->resources;
  for (auto &[id, stream] : resources.streams)
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  for (const auto vertex : boost::make_iterator_range(boost::vertices(compiled_->graph))) {
    const auto &node = compiled_->graph[vertex];
    if (node.spec.name !=
        (settings.method == "holovibes_stft" ? "holo_temporal_volume" : "focus_stack"))
      continue;
    const auto  tid     = node.out_tids.at(0);
    const auto &desc    = resources.tensor_descs.at(tid);
    const auto *storage = resources.storages.at(resources.tid_to_sid.at(tid)).get();
    retain(desc, storage->ptr + desc.offset);
    graph_dot_     = holoflow::runtime::to_dot(*compiled_, {}, "holographic_focus_stack");
    current_frame_ = frame;
    return;
  }
  throw std::runtime_error("Focus reconstruction did not produce a volume");
}
} // namespace holovibes::pipeline
