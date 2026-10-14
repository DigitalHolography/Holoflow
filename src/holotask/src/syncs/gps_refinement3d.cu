// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/gps_refinement3d.hh"
#include "syncs/mtd_common.cuh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const GPSRefinement3DSettings &s) {
  j = {{"delta_n_min", s.delta_n_min},
       {"delta_n_max", s.delta_n_max},
       {"kappa_min", s.kappa_min},
       {"kappa_max", s.kappa_max},
       {"iterations", s.iterations}};
}
void from_json(const nlohmann::json &j, GPSRefinement3DSettings &s) {
  s.delta_n_min = j.value("delta_n_min", -0.5f);
  s.delta_n_max = j.value("delta_n_max", 0.5f);
  s.kappa_min   = j.value("kappa_min", 0.0f);
  s.kappa_max   = j.value("kappa_max", 100000.0f);
  s.iterations  = j.value("iterations", 10);
}

namespace {
using namespace mtd_detail;

__global__ void constrain_volume(cuFloatComplex *volume, size_t count, float rmin, float rmax,
                                 float imin, float imax) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    volume[i].x = fminf(rmax, fmaxf(rmin, volume[i].x));
    volume[i].y = fminf(imax, fmaxf(imin, volume[i].y));
  }
}
__global__ void restore_measurements(cuFloatComplex *spectrum, const cuFloatComplex *measured,
                                     const float *coverage, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && coverage[i] > 0)
    spectrum[i] = measured[i];
}

class GPSRefinement3D : public ISyncTask {
public:
  GPSRefinement3D(TDesc input, GPSRefinement3DSettings settings, cudaStream_t stream)
      : input_(std::move(input)), s_(settings), stream_(stream),
        measured_(curaii::make_unique_device_ptr<cuFloatComplex>(input_.num_elements(), stream)) {
    plan3(plan_, static_cast<int>(input_.shape[0]), stream_);
  }
  OpResult execute(SyncCtx &ctx) override {
    if (cancelled(ctx))
      return OpResult::Cancelled;
    const size_t count    = input_.num_elements();
    const int    dim      = static_cast<int>(input_.shape[0]);
    auto        *input    = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data());
    auto        *output   = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
    auto        *coverage = reinterpret_cast<const float *>(ctx.inputs[1].data());
    CUDA_CHECK(cudaMemcpyAsync(measured_.get(), input, input_.num_bytes(), cudaMemcpyDeviceToDevice,
                               stream_));
    if (input != output)
      CUDA_CHECK(
          cudaMemcpyAsync(output, input, input_.num_bytes(), cudaMemcpyDeviceToDevice, stream_));
    centered_transform(plan_, measured_.get(), dim, count, CUFFT_FORWARD, stream_);
    for (int iteration = 0; iteration < s_.iterations; ++iteration) {
      if (cancelled(ctx)) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return OpResult::Cancelled;
      }
      constrain_volume<<<blocks(count), block_size, 0, stream_>>>(
          output, count, s_.delta_n_min, s_.delta_n_max, s_.kappa_min, s_.kappa_max);
      centered_transform(plan_, output, dim, count, CUFFT_FORWARD, stream_);
      restore_measurements<<<blocks(count), block_size, 0, stream_>>>(output, measured_.get(),
                                                                      coverage, count);
      centered_transform(plan_, output, dim, count, CUFFT_INVERSE, stream_);
      scale<<<blocks(count), block_size, 0, stream_>>>(output, count,
                                                       1.0f / static_cast<float>(count));
    }
    constrain_volume<<<blocks(count), block_size, 0, stream_>>>(
        output, count, s_.delta_n_min, s_.delta_n_max, s_.kappa_min, s_.kappa_max);
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  TDesc                     input_;
  GPSRefinement3DSettings   s_;
  cudaStream_t              stream_;
  DevicePtr<cuFloatComplex> measured_;
  curaii::CufftHandle       plan_;
};
} // namespace

holoflow::core::InferResult GPSRefinement3DFactory::infer(std::span<const TDesc> inputs,
                                                          const nlohmann::json  &settings) const {
  const auto s = settings.get<GPSRefinement3DSettings>();
  require(inputs.size() == 2, "GPS requires a linear volume and Fourier coverage");
  const auto &volume = inputs[0], &coverage = inputs[1];
  require(volume.rank() == 3 && contiguous(volume) && volume.dtype == DType::CF32 &&
              volume.mem_loc == MemLoc::Device,
          "GPS volume must be contiguous device CF32 [z,y,x]");
  require(volume.shape[0] == volume.shape[1] && volume.shape[1] == volume.shape[2] &&
              volume.shape[0] % 2 == 0 && volume.shape[0] <= 1024,
          "GPS requires an even cubic volume of supported size");
  require(coverage.shape == volume.shape && coverage.dtype == DType::F32 &&
              coverage.mem_loc == MemLoc::Device && contiguous(coverage),
          "GPS coverage must be device F32 matching the volume");
  require(s.iterations > 0 && s.iterations <= 1000 && std::isfinite(s.delta_n_min) &&
              std::isfinite(s.delta_n_max) && s.delta_n_min <= s.delta_n_max &&
              std::isfinite(s.kappa_min) && std::isfinite(s.kappa_max) &&
              s.kappa_min <= s.kappa_max,
          "invalid GPS iteration count or real/imaginary bounds");
  return {.input_descs   = {volume, coverage},
          .output_descs  = {TDesc(volume.shape, volume.dtype, volume.mem_loc)},
          .owned_inputs  = {false, false},
          .owned_outputs = {false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> GPSRefinement3DFactory::create(std::span<const TDesc> inputs,
                                                          const nlohmann::json  &settings,
                                                          const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<GPSRefinement3D>(inputs[0], settings.get<GPSRefinement3DSettings>(),
                                           ctx.stream);
}
} // namespace holotask::syncs
