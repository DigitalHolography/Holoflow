// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/volume_slice.hh"
#include "syncs/mtd_common.cuh"
#include <type_traits>

namespace holotask::syncs {
void to_json(nlohmann::json &j, const VolumeSliceSettings &s) {
  j = {{"axis", s.axis}, {"index", s.index}, {"component", s.component}};
}
void from_json(const nlohmann::json &j, VolumeSliceSettings &s) {
  s.axis      = j.value("axis", 0);
  s.index     = j.value("index", 280);
  s.component = j.value("component", std::string{"real"});
}

namespace {
using namespace mtd_detail;
template <typename T>
__global__ void select_slice(const T *volume, float *output, size_t height, size_t width, size_t sy,
                             size_t sx, int axis, int index, int component) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= height * width)
    return;
  size_t x = i % width, y = i / width, z = static_cast<size_t>(index);
  if (axis == 1) {
    z = y;
    y = static_cast<size_t>(index);
  } else if (axis == 2) {
    z = y;
    y = x;
    x = static_cast<size_t>(index);
  }
  const size_t source = (z * sy + y) * sx + x;
  if constexpr (std::is_same_v<T, float>)
    output[i] = component == 1 ? 0.0f : (component == 2 ? fabsf(volume[source]) : volume[source]);
  else {
    const auto v = volume[source];
    output[i]    = component == 0 ? v.x : (component == 1 ? v.y : hypotf(v.x, v.y));
  }
}
class VolumeSlice : public ISyncTask {
public:
  VolumeSlice(TDesc input, VolumeSliceSettings settings, cudaStream_t stream)
      : input_(std::move(input)), s_(std::move(settings)), stream_(stream) {}
  OpResult execute(SyncCtx &ctx) override {
    if (cancelled(ctx))
      return OpResult::Cancelled;
    const auto  &out = ctx.outputs[0].desc;
    const size_t h = out.shape[0], w = out.shape[1];
    auto        *output    = reinterpret_cast<float *>(ctx.outputs[0].data());
    const int    component = s_.component == "real" ? 0 : (s_.component == "imaginary" ? 1 : 2);
    if (input_.dtype == DType::CF32)
      select_slice<<<blocks(h * w), block_size, 0, stream_>>>(
          reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data()), output, h, w,
          input_.shape[1], input_.shape[2], s_.axis, s_.index, component);
    else
      select_slice<<<blocks(h * w), block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(ctx.inputs[0].data()), output, h, w, input_.shape[1],
          input_.shape[2], s_.axis, s_.index, component);
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  TDesc               input_;
  VolumeSliceSettings s_;
  cudaStream_t        stream_;
};
} // namespace
holoflow::core::InferResult VolumeSliceFactory::infer(std::span<const TDesc> inputs,
                                                      const nlohmann::json  &settings) const {
  const auto s = settings.get<VolumeSliceSettings>();
  require(inputs.size() == 1, "volume slice expects one volume");
  const auto &in = inputs[0];
  require(in.rank() == 3 && contiguous(in) && in.mem_loc == MemLoc::Device &&
              (in.dtype == DType::CF32 || in.dtype == DType::F32),
          "slice input must be contiguous device F32/CF32 [z,y,x]");
  require(s.axis >= 0 && s.axis < 3 && s.index >= 0 &&
              static_cast<size_t>(s.index) < in.shape[static_cast<size_t>(s.axis)],
          "slice axis or index out of range");
  require(s.component == "real" || s.component == "imaginary" || s.component == "magnitude",
          "unknown slice component");
  std::vector<size_t> shape;
  for (int axis = 0; axis < 3; ++axis)
    if (axis != s.axis)
      shape.push_back(in.shape[static_cast<size_t>(axis)]);
  return {.input_descs   = {in},
          .output_descs  = {TDesc(shape, DType::F32, MemLoc::Device)},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> VolumeSliceFactory::create(std::span<const TDesc> inputs,
                                                      const nlohmann::json  &settings,
                                                      const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<VolumeSlice>(inputs[0], settings.get<VolumeSliceSettings>(), ctx.stream);
}
} // namespace holotask::syncs
