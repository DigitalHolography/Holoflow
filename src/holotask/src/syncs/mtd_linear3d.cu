// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/mtd_linear3d.hh"
#include "syncs/mtd_common.cuh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const MTDLinear3DReconstructionSettings &s) {
  j = {{"dimension", s.dimension},
       {"hologram_size", s.hologram_size},
       {"wavelength", s.wavelength},
       {"pixel_pitch_object", s.pixel_pitch_object},
       {"background_index", s.background_index},
       {"medium_index", s.medium_index},
       {"tukey_alpha", s.tukey_alpha}};
}
void from_json(const nlohmann::json &j, MTDLinear3DReconstructionSettings &s) {
  s.dimension          = j.value("dimension", 512);
  s.hologram_size      = j.value("hologram_size", 1024);
  s.wavelength         = j.value("wavelength", 633e-9);
  s.pixel_pitch_object = j.value("pixel_pitch_object", 55e-9);
  s.background_index   = j.value("background_index", 1.49);
  s.medium_index       = j.value("medium_index", 1.49);
  s.tukey_alpha        = j.value("tukey_alpha", 0.1f);
}

namespace {
using namespace mtd_detail;

__global__ void prepare_spectrum(const cuFloatComplex *field, const float *window,
                                 cuFloatComplex *output, int side) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  const int   source = ((i / side + side / 2) % side) * side + (i % side + side / 2) % side;
  const auto  v      = field[source];
  const float w      = window[source];
  output[i]          = make_cuFloatComplex(v.x * w, v.y * w);
}

__global__ void accumulate_ewald(const cuFloatComplex *spectrum, const float *position,
                                 cuFloatComplex *volume, float *coverage, int side, int dim,
                                 double radius, double k0, double dx_squared) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  const int half = side / 2, fx = i % side - half, fy = i / side - half;
  if (fx * fx + fy * fy >= half * half)
    return;
  if (!isfinite(position[0]) || !isfinite(position[1]) || position[0] < 0 || position[1] < 0 ||
      position[0] >= side || position[1] >= side || truncf(position[0]) != position[0] ||
      truncf(position[1]) != position[1])
    return;
  const int    px = static_cast<int>(position[0]), py = static_cast<int>(position[1]);
  const double r2    = radius * radius - fx * fx - fy * fy;
  const double base2 = radius * radius - (px - half) * (px - half) - (py - half) * (py - half);
  if (r2 < 0 || base2 <= 0 || px < 0 || py < 0 || px >= side || py >= side)
    return;
  const double rz = sqrt(r2);
  const int    x = dim / 2 + fx - (px - half), y = dim / 2 + fy - (py - half);
  const int    z = dim / 2 + __double2int_rn(rz - sqrt(base2));
  if (x < 0 || y < 0 || z < 0 || x >= dim || y >= dim || z >= dim)
    return;
  const int    sx = (i % side - px + side) % side, sy = (i / side - py + side) % side;
  const auto   value       = spectrum[sy * side + sx];
  const double factor      = 2 * k0 * (rz / radius) * dx_squared;
  const size_t destination = (static_cast<size_t>(z) * dim + y) * dim + x;
  // At fixed illumination, integer X/Y coordinates are unique; sequential
  // launches give deterministic accumulation across illumination directions.
  atomicAdd(&volume[destination].x, static_cast<float>(factor * value.y));
  atomicAdd(&volume[destination].y, static_cast<float>(-factor * value.x));
  atomicAdd(&coverage[destination], 1.0f);
}

__global__ void average_coverage(cuFloatComplex *volume, const float *coverage, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && coverage[i] > 0) {
    volume[i].x /= coverage[i];
    volume[i].y /= coverage[i];
  }
}

class MTDLinear3DReconstruction : public ISyncTask {
public:
  MTDLinear3DReconstruction(TDesc input, MTDLinear3DReconstructionSettings s, cudaStream_t stream)
      : input_(std::move(input)), s_(s), stream_(stream), side_(static_cast<int>(input_.shape[1])),
        spectrum_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(side_) * side_,
                                                                 stream)),
        window_(curaii::make_unique_device_ptr<float>(static_cast<size_t>(side_) * side_, stream)) {
    plan2(plan2_, side_, stream_);
    plan3(plan3_, s_.dimension, stream_);
    const auto values = window(side_, s_.tukey_alpha);
    CUDA_CHECK(cudaMemcpyAsync(window_.get(), values.data(), values.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  OpResult execute(SyncCtx &ctx) override {
    if (cancelled(ctx))
      return OpResult::Cancelled;
    auto        *volume   = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
    auto        *coverage = reinterpret_cast<float *>(ctx.outputs[1].data());
    const size_t count    = ctx.outputs[0].desc.num_elements(),
                 pixels   = static_cast<size_t>(side_) * side_;
    CUDA_CHECK(cudaMemsetAsync(volume, 0, count * sizeof(cuFloatComplex), stream_));
    CUDA_CHECK(cudaMemsetAsync(coverage, 0, count * sizeof(float), stream_));
    const double radius =
        s_.hologram_size * s_.pixel_pitch_object * s_.medium_index / s_.wavelength;
    const double dx        = s_.pixel_pitch_object * s_.hologram_size / side_,
                 k0        = 2 * pi / s_.wavelength * s_.medium_index;
    const auto  *fields    = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data());
    const auto  *positions = reinterpret_cast<const float *>(ctx.inputs[1].data());
    for (size_t frame = 0; frame < input_.shape[0]; ++frame) {
      if (cancelled(ctx)) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return OpResult::Cancelled;
      }
      prepare_spectrum<<<blocks(pixels), block_size, 0, stream_>>>(
          fields + frame * pixels, window_.get(), spectrum_.get(), side_);
      CUFFT_CHECK(cufftExecC2C(plan2_.get(), spectrum_.get(), spectrum_.get(), CUFFT_FORWARD));
      accumulate_ewald<<<blocks(pixels), block_size, 0, stream_>>>(
          spectrum_.get(), positions + 2 * frame, volume, coverage, side_, s_.dimension, radius, k0,
          dx * dx);
    }
    average_coverage<<<blocks(count), block_size, 0, stream_>>>(volume, coverage, count);
    centered_transform(plan3_, volume, s_.dimension, count, CUFFT_INVERSE, stream_);
    const double df = 1.0 / (dx * side_);
    const double factor =
        -df * df * df / (2 * s_.background_index * std::pow(2 * pi / s_.wavelength, 2));
    scale<<<blocks(count), block_size, 0, stream_>>>(volume, count, static_cast<float>(factor));
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  TDesc                             input_;
  MTDLinear3DReconstructionSettings s_;
  cudaStream_t                      stream_;
  int                               side_;
  DevicePtr<cuFloatComplex>         spectrum_;
  DevicePtr<float>                  window_;
  curaii::CufftHandle               plan2_, plan3_;
};
} // namespace

holoflow::core::InferResult
MTDLinear3DReconstructionFactory::infer(std::span<const TDesc> inputs,
                                        const nlohmann::json  &settings) const {
  const auto s = settings.get<MTDLinear3DReconstructionSettings>();
  require(inputs.size() == 2, "linear reconstruction requires fields and carrier positions");
  const auto &fields = inputs[0], &positions = inputs[1];
  require(fields.rank() == 3 && fields.dtype == DType::CF32 && fields.mem_loc == MemLoc::Device &&
              contiguous(fields),
          "linear reconstruction requires contiguous device complex fields");
  require(fields.shape[1] == fields.shape[2] && fields.shape[1] >= 2 && fields.shape[1] % 2 == 0 &&
              fields.shape[1] <= 32768,
          "linear reconstruction requires even square fields");
  require(positions.rank() == 2 && positions.shape[0] == fields.shape[0] &&
              positions.shape[1] == 2 && positions.dtype == DType::F32 &&
              positions.mem_loc == MemLoc::Device && contiguous(positions),
          "carrier positions must be device F32 [frames,2]");
  require(s.dimension >= 2 && s.dimension <= 1024 && s.dimension % 2 == 0,
          "volume dimension must be even and between 2 and 1024");
  require(s.hologram_size > 0 && std::isfinite(s.wavelength) && s.wavelength > 0 &&
              std::isfinite(s.pixel_pitch_object) && s.pixel_pitch_object > 0 &&
              std::isfinite(s.background_index) && s.background_index > 0 &&
              std::isfinite(s.medium_index) && s.medium_index > 0,
          "optical parameters must be finite and positive");
  require(std::isfinite(s.tukey_alpha) && s.tukey_alpha >= 0 && s.tukey_alpha <= 1,
          "invalid Tukey alpha");
  const double radius = s.hologram_size * s.pixel_pitch_object * s.medium_index / s.wavelength;
  const double k0     = 2 * pi * s.medium_index / s.wavelength;
  const double extent = s.hologram_size * s.pixel_pitch_object;
  require(std::isfinite(radius) && radius > 0 && std::isfinite(radius * radius) &&
              std::isfinite(k0) && std::isfinite(extent) && extent > 0,
          "derived optical geometry exceeds the supported numeric range");
  const size_t dim = static_cast<size_t>(s.dimension);
  return {.input_descs   = {fields, positions},
          .output_descs  = {TDesc({dim, dim, dim}, DType::CF32, MemLoc::Device),
                            TDesc({dim, dim, dim}, DType::F32, MemLoc::Device)},
          .owned_inputs  = {false, false},
          .owned_outputs = {false, false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> MTDLinear3DReconstructionFactory::create(
    std::span<const TDesc> inputs, const nlohmann::json &settings, const SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<MTDLinear3DReconstruction>(
      inputs[0], settings.get<MTDLinear3DReconstructionSettings>(), ctx.stream);
}
} // namespace holotask::syncs
