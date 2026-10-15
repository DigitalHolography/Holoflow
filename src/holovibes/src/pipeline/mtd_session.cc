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
    auto volume = curaii::make_unique_device_ptr<std::byte>(descriptor.num_bytes(), stream_.get());
    CUDA_CHECK(cudaMemcpyAsync(volume.get(), storage->ptr + descriptor.offset,
                               descriptor.num_bytes(), cudaMemcpyDeviceToDevice, stream_.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    auto plane = curaii::make_unique_device_ptr<float>(
        static_cast<size_t>(settings.dimension) * settings.dimension, stream_.get());
    graph_dot_     = holoflow::runtime::to_dot(*compiled, {}, "mtd_reconstruction");
    volume_        = std::move(volume);
    plane_         = std::move(plane);
    desc_          = TDesc(descriptor.shape, descriptor.dtype, descriptor.mem_loc);
    dimension_     = settings.dimension;
    voxel_pitch_m_ = dataset.linear.hologram_size * dataset.linear.pixel_pitch_object / dimension_;
    directory_     = directory;
    settings_      = settings;
    return;
  }
  throw std::runtime_error("MTD reconstruction did not produce a GPS volume");
}

QByteArray MTDSession::slice(int axis, int index, const std::string &component) {
  if (dimension_ == 0)
    throw std::logic_error("MTD volume has not been reconstructed");
  holotask::syncs::VolumeSliceFactory        factory;
  const std::vector<TDesc>                   inputs{desc_};
  const holotask::syncs::VolumeSliceSettings settings{
      .axis = axis, .index = index, .component = component};
  const auto         inferred = factory.infer(inputs, settings);
  auto               task     = factory.create(inputs, settings, {stream_.get()});
  Storage            input_storage{MemLoc::Device, desc_.num_bytes(), volume_.get()};
  const auto        &out_desc = inferred.output_descs.at(0);
  Storage            output_storage{MemLoc::Device, out_desc.num_bytes(),
                                    reinterpret_cast<std::byte *>(plane_.get())};
  std::vector<TView> input_views{{desc_, &input_storage}},
      output_views{{out_desc, &output_storage}};
  std::atomic<bool> cancelled{false};
  SyncCtx context{.inputs = input_views, .outputs = output_views, .cancelled = &cancelled};
  if (task->execute(context) != OpResult::Ok)
    throw std::runtime_error("MTD slice extraction did not complete");
  QByteArray data;
  data.resize(static_cast<qsizetype>(out_desc.num_bytes()));
  CUDA_CHECK(cudaMemcpyAsync(data.data(), plane_.get(), out_desc.num_bytes(),
                             cudaMemcpyDeviceToHost, stream_.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  return data;
}
} // namespace holovibes::pipeline
