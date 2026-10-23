// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline/volume_session.hh"
#include "holotask/syncs/volume_slice.hh"
#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace holovibes::pipeline {
using namespace holoflow::core;
void VolumeSession::retain(const TDesc &desc, const std::byte *data) {
  if (volume_ && desc_.shape == desc.shape && desc_.dtype == desc.dtype &&
      desc_.mem_loc == desc.mem_loc) {
    CUDA_CHECK(cudaMemcpyAsync(volume_.get(), data, desc.num_bytes(), cudaMemcpyDeviceToDevice,
                               stream_.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return;
  }
  auto volume = curaii::make_unique_device_ptr<std::byte>(desc.num_bytes(), stream_.get());
  CUDA_CHECK(cudaMemcpyAsync(volume.get(), data, desc.num_bytes(), cudaMemcpyDeviceToDevice,
                             stream_.get()));
  const auto  &shape = desc.shape;
  const size_t count = std::max({shape[0] * shape[1], shape[0] * shape[2], shape[1] * shape[2]});
  auto         plane = curaii::make_unique_device_ptr<float>(count, stream_.get());
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  volume_ = std::move(volume);
  plane_  = std::move(plane);
  desc_   = TDesc(desc.shape, desc.dtype, desc.mem_loc);
}

QByteArray VolumeSession::slice(int axis, int index, const std::string &component, int count,
                                bool transpose) {
  if (width() == 0)
    throw std::logic_error("Volume has not been reconstructed");
  holotask::syncs::VolumeSliceFactory        factory;
  const std::vector<TDesc>                   inputs{desc_};
  const holotask::syncs::VolumeSliceSettings settings{.axis             = axis,
                                                      .index            = index,
                                                      .component        = component,
                                                      .count            = count,
                                                      .transpose_output = transpose};
  const auto                                 inferred = factory.infer(inputs, settings);
  auto               task = factory.create(inputs, settings, {stream_.get()});
  Storage            input_storage{MemLoc::Device, desc_.num_bytes(), volume_.get()};
  const auto        &out_desc = inferred.output_descs.at(0);
  Storage            output_storage{MemLoc::Device, out_desc.num_bytes(),
                                    reinterpret_cast<std::byte *>(plane_.get())};
  std::vector<TView> input_views{{desc_, &input_storage}},
      output_views{{out_desc, &output_storage}};
  std::atomic<bool> cancelled{false};
  SyncCtx context{.inputs = input_views, .outputs = output_views, .cancelled = &cancelled};
  if (task->execute(context) != OpResult::Ok)
    throw std::runtime_error("Volume slice extraction did not complete");
  QByteArray data;
  data.resize(static_cast<qsizetype>(out_desc.num_bytes()));
  CUDA_CHECK(cudaMemcpyAsync(data.data(), plane_.get(), out_desc.num_bytes(),
                             cudaMemcpyDeviceToHost, stream_.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  return data;
}
} // namespace holovibes::pipeline
