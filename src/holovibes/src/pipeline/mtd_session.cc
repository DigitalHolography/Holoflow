// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline/mtd_session.hh"

#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_display.hh"
#include "holotask/pipelines/mtd.hh"
#include "holotask/syncs/volume_slice.hh"
#include <atomic>
#include <boost/range/iterator_range.hpp>
#include <stdexcept>

namespace holovibes::pipeline {
using namespace holoflow::core;

bool MTDSession::matches(const std::filesystem::path     &directory,
                         const MTDReconstructionSettings &settings) const {
  return dimension_ > 0 && directory_ == directory && settings_ == settings;
}

void MTDSession::reconstruct(const std::filesystem::path     &directory,
                             const MTDReconstructionSettings &settings) {
  if (matches(directory, settings))
    return;
  auto dataset = holotask::pipelines::load_mtd_dataset(directory, settings.medium_index);
  dataset.linear.dimension = settings.dimension;
  dataset.gps.iterations   = settings.iterations;
  const auto spec          = holotask::pipelines::build_mtd_pipeline(dataset, {});
  Registry   registry;
  holotask::pipelines::register_mtd_tasks(registry);
  holoflow::runtime::Compiler  compiler(registry,
                                        {.verbose_tracing = false, .enable_profiling = false});
  auto                         compiled = compiler.compile(spec);
  holoflow::runtime::Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  scheduler.start();
  scheduler.wait();
  auto &resources = compiled->resources;
  for (auto &[id, stream] : resources.streams)
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  for (const auto vertex : boost::make_iterator_range(boost::vertices(compiled->graph))) {
    const auto &node = compiled->graph[vertex];
    if (node.spec.name != "mtd_gps3d")
      continue;
    const auto  tid        = node.out_tids.at(0);
    const auto  descriptor = resources.tensor_descs.at(tid);
    const auto *storage    = resources.storages.at(resources.tid_to_sid.at(tid)).get();
    retain(descriptor, storage->ptr + descriptor.offset);
    graph_dot_     = holoflow::runtime::to_dot(*compiled, {}, "mtd_reconstruction");
    dimension_     = settings.dimension;
    voxel_pitch_m_ = dataset.linear.hologram_size * dataset.linear.pixel_pitch_object / dimension_;
    directory_     = directory;
    settings_      = settings;
    return;
  }
  throw std::runtime_error("MTD reconstruction did not produce a GPS volume");
}

} // namespace holovibes::pipeline
