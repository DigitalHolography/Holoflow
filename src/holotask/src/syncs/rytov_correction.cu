// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/rytov_correction.hh"
#include "syncs/mtd_background.hh"
#include "syncs/mtd_common.cuh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const RytovCorrectionSettings &s) {
  j = {{"polynomial_degree", s.polynomial_degree}, {"sample_step", s.sample_step},
       {"background_margin", s.background_margin}, {"mask_threshold", s.mask_threshold},
       {"unwrap_phase", s.unwrap_phase},           {"amplitude_floor", s.amplitude_floor},
       {"damping_global", s.damping_global},       {"damping_local", s.damping_local},
       {"gradient_margin", s.gradient_margin}};
}
void from_json(const nlohmann::json &j, RytovCorrectionSettings &s) {
  s.polynomial_degree = j.value("polynomial_degree", 5);
  s.sample_step       = j.value("sample_step", 5);
  s.background_margin = j.value("background_margin", 5);
  s.mask_threshold    = j.value("mask_threshold", 45);
  s.unwrap_phase      = j.value("unwrap_phase", true);
  s.amplitude_floor   = j.value("amplitude_floor", 0.01f);
  s.damping_global    = j.value("damping_global", 3e-4f);
  s.damping_local     = j.value("damping_local", 0.1f);
  s.gradient_margin   = j.value("gradient_margin", 2);
}

namespace {
using namespace mtd_detail;

__global__ void maximum_norm(const cuFloatComplex *field, float *result, int count) {
  __shared__ float partial[block_size];
  float            maximum = 0;
  for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(block_size)) {
    const auto v = field[i];
    maximum      = fmaxf(maximum, v.x * v.x + v.y * v.y);
  }
  partial[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    *result = partial[0];
}

__global__ void mirror_field(const cuFloatComplex *field, cuFloatComplex *mirror, int side) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x), size = 2 * side;
  if (i >= size * size)
    return;
  const int x = i % size, y = i / size;
  const int source_x = x < side ? side - 1 - x : x - side,
            source_y = y < side ? side - 1 - y : y - side;
  mirror[i]          = field[source_y * side + source_x];
}

__device__ float frequency(int i, int size) {
  return static_cast<float>(i < size / 2 ? i : i - size) / static_cast<float>(size);
}

__global__ void spectral_gradient(const cuFloatComplex *spectrum, cuFloatComplex *gx,
                                  cuFloatComplex *gy, int size) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= size * size)
    return;
  const float kx = static_cast<float>(2 * pi) * frequency(i % size, size);
  const float ky = static_cast<float>(2 * pi) * frequency(i / size, size);
  const auto  v  = spectrum[i];
  gx[i]          = make_cuFloatComplex(-kx * v.y, kx * v.x);
  gy[i]          = make_cuFloatComplex(-ky * v.y, ky * v.x);
}

__global__ void phase_gradient(const cuFloatComplex *field, cuFloatComplex *gx, cuFloatComplex *gy,
                               const float *max_norm, int size, float alpha, float beta,
                               int margin) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= size * size)
    return;
  const int x = i % size, y = i / size;
  if (x < margin || y < margin || x >= size - margin || y >= size - margin ||
      abs(x - size / 2) < margin || abs(y - size / 2) < margin) {
    gx[i] = gy[i] = make_cuFloatComplex(0, 0);
    return;
  }
  const auto  f = field[i];
  const float denominator =
      fmaxf((f.x * f.x + f.y * f.y) * (1 + beta) + alpha * (*max_norm), 1e-30f);
  const float normalization = 1.0f / (static_cast<float>(size) * size);
  const auto  dx = gx[i], dy = gy[i];
  gx[i] = make_cuFloatComplex((dx.y * f.x - dx.x * f.y) * normalization / denominator, 0);
  gy[i] = make_cuFloatComplex((dy.y * f.x - dy.x * f.y) * normalization / denominator, 0);
}

__global__ void integrate_gradient(const cuFloatComplex *gx, const cuFloatComplex *gy,
                                   cuFloatComplex *spectrum, int size) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= size * size)
    return;
  const float kx = frequency(i % size, size), ky = frequency(i / size, size);
  const float denominator = static_cast<float>(2 * pi) * (kx * kx + ky * ky);
  if (i == 0)
    spectrum[i] = make_cuFloatComplex(0, 0);
  else
    spectrum[i] = make_cuFloatComplex((kx * gx[i].y + ky * gy[i].y) / denominator,
                                      -(kx * gx[i].x + ky * gy[i].x) / denominator);
}

__global__ void extract_phase(const cuFloatComplex *field, const cuFloatComplex *integrated,
                              float *phase, int side, bool unwrap) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  phase[i] =
      unwrap ? integrated[(i / side + side) * 2 * side + i % side + side].x / (4.0f * side * side)
             : atan2f(field[i].y, field[i].x);
}

__global__ void fit_background(const cuFloatComplex *field, const float *phase, const int *indexes,
                               const double *weights, double *coefficients, int samples) {
  __shared__ double  amplitude_sum[block_size], phase_sum[block_size];
  const unsigned int tid = threadIdx.x;
  double             amp = 0, phi = 0;
  for (int i = static_cast<int>(tid); i < samples; i += static_cast<int>(block_size)) {
    const auto   v      = field[indexes[i]];
    const double weight = weights[static_cast<size_t>(blockIdx.x) * samples + i];
    amp += weight * std::sqrt(static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y);
    phi += weight * phase[indexes[i]];
  }
  amplitude_sum[tid] = amp;
  phase_sum[tid]     = phi;
  __syncthreads();
  for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      amplitude_sum[tid] += amplitude_sum[tid + stride];
      phase_sum[tid] += phase_sum[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    coefficients[2 * blockIdx.x]     = amplitude_sum[0];
    coefficients[2 * blockIdx.x + 1] = phase_sum[0];
  }
}

__global__ void correct_rytov(const cuFloatComplex *field, const float *phase, const double *coeff,
                              const float *max_norm, cuFloatComplex *output, int side, int degree,
                              float amplitude_floor) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  const double x = 2.0 * (i % side) / (side - 1) - 1.0, y = 2.0 * (i / side) / (side - 1) - 1.0;
  double       xp[9], yp[9];
  xp[0] = yp[0] = 1.0;
  for (int p = 1; p <= degree; ++p) {
    xp[p] = xp[p - 1] * x;
    yp[p] = yp[p - 1] * y;
  }
  double amplitude_bg = 0, phase_bg = 0;
  int    term = 0;
  for (int px = 0; px <= degree; ++px)
    for (int py = 0; py <= degree - px; ++py, ++term) {
      const double basis = xp[px] * yp[py];
      amplitude_bg += basis * coeff[2 * term];
      phase_bg += basis * coeff[2 * term + 1];
    }
  const auto   v   = field[i];
  const double amp = sqrt(static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y);
  amplitude_bg =
      fmax(amplitude_bg, fmax(amplitude_floor * sqrt(static_cast<double>(*max_norm)), 1e-30));
  output[i] = make_cuFloatComplex(static_cast<float>(log(fmax(amp / amplitude_bg, 1e-8))),
                                  static_cast<float>(phase[i] - phase_bg));
}

class RytovCorrection : public ISyncTask {
public:
  RytovCorrection(TDesc input, RytovCorrectionSettings settings, cudaStream_t stream)
      : input_(std::move(input)), s_(settings), stream_(stream),
        side_(static_cast<int>(input_.shape[1])),
        mirror_(curaii::make_unique_device_ptr<cuFloatComplex>(
            static_cast<size_t>(4) * side_ * side_, stream)),
        spectrum_(curaii::make_unique_device_ptr<cuFloatComplex>(
            static_cast<size_t>(4) * side_ * side_, stream)),
        gx_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(4) * side_ * side_,
                                                           stream)),
        gy_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(4) * side_ * side_,
                                                           stream)),
        phase_(curaii::make_unique_device_ptr<float>(static_cast<size_t>(side_) * side_, stream)),
        max_norm_(curaii::make_unique_device_ptr<float>(1, stream)),
        indexes_(curaii::make_unique_device_ptr<int>(static_cast<size_t>(side_) * side_, stream)),
        weights_(curaii::make_unique_device_ptr<double>(
            static_cast<size_t>((s_.polynomial_degree + 1) * (s_.polynomial_degree + 2) / 2) *
                side_ * side_,
            stream)),
        coefficients_(curaii::make_unique_device_ptr<double>(
            static_cast<size_t>((s_.polynomial_degree + 1) * (s_.polynomial_degree + 2)), stream)) {
    plan2(plan_, 2 * side_, stream_);
  }

  OpResult execute(SyncCtx &ctx) override {
    if (cancelled(ctx))
      return OpResult::Cancelled;
    const size_t pixels = static_cast<size_t>(side_) * side_;
    const auto   model  = background_projection(
        {reinterpret_cast<const uint8_t *>(ctx.inputs[1].data()), pixels}, side_, s_);
    CUDA_CHECK(cudaMemcpyAsync(indexes_.get(), model.indexes.data(),
                               model.indexes.size() * sizeof(int), cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(weights_.get(), model.weights.data(),
                               model.weights.size() * sizeof(double), cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const int terms  = (s_.polynomial_degree + 1) * (s_.polynomial_degree + 2) / 2;
    auto     *input  = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data());
    auto     *output = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
    for (size_t frame = 0; frame < input_.shape[0]; ++frame) {
      if (cancelled(ctx)) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return OpResult::Cancelled;
      }
      const auto *field = input + frame * pixels;
      maximum_norm<<<1, block_size, 0, stream_>>>(field, max_norm_.get(), static_cast<int>(pixels));
      if (s_.unwrap_phase) {
        mirror_field<<<blocks(4 * pixels), block_size, 0, stream_>>>(field, mirror_.get(), side_);
        CUDA_CHECK(cudaMemcpyAsync(spectrum_.get(), mirror_.get(),
                                   4 * pixels * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice,
                                   stream_));
        CUFFT_CHECK(cufftExecC2C(plan_.get(), spectrum_.get(), spectrum_.get(), CUFFT_FORWARD));
        spectral_gradient<<<blocks(4 * pixels), block_size, 0, stream_>>>(
            spectrum_.get(), gx_.get(), gy_.get(), 2 * side_);
        CUFFT_CHECK(cufftExecC2C(plan_.get(), gx_.get(), gx_.get(), CUFFT_INVERSE));
        CUFFT_CHECK(cufftExecC2C(plan_.get(), gy_.get(), gy_.get(), CUFFT_INVERSE));
        phase_gradient<<<blocks(4 * pixels), block_size, 0, stream_>>>(
            mirror_.get(), gx_.get(), gy_.get(), max_norm_.get(), 2 * side_, s_.damping_global,
            s_.damping_local, s_.gradient_margin);
        CUFFT_CHECK(cufftExecC2C(plan_.get(), gx_.get(), gx_.get(), CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(plan_.get(), gy_.get(), gy_.get(), CUFFT_FORWARD));
        integrate_gradient<<<blocks(4 * pixels), block_size, 0, stream_>>>(
            gx_.get(), gy_.get(), spectrum_.get(), 2 * side_);
        CUFFT_CHECK(cufftExecC2C(plan_.get(), spectrum_.get(), spectrum_.get(), CUFFT_INVERSE));
      }
      extract_phase<<<blocks(pixels), block_size, 0, stream_>>>(
          field, spectrum_.get(), phase_.get(), side_, s_.unwrap_phase);
      fit_background<<<static_cast<unsigned int>(terms), block_size, 0, stream_>>>(
          field, phase_.get(), indexes_.get(), weights_.get(), coefficients_.get(),
          static_cast<int>(model.indexes.size()));
      correct_rytov<<<blocks(pixels), block_size, 0, stream_>>>(
          field, phase_.get(), coefficients_.get(), max_norm_.get(), output + frame * pixels, side_,
          s_.polynomial_degree, s_.amplitude_floor);
      CUDA_CHECK(cudaGetLastError());
    }
    return OpResult::Ok;
  }

private:
  TDesc                     input_;
  RytovCorrectionSettings   s_;
  cudaStream_t              stream_;
  int                       side_;
  DevicePtr<cuFloatComplex> mirror_, spectrum_, gx_, gy_;
  DevicePtr<float>          phase_, max_norm_;
  DevicePtr<int>            indexes_;
  DevicePtr<double>         weights_, coefficients_;
  curaii::CufftHandle       plan_;
};
} // namespace

holoflow::core::InferResult RytovCorrectionFactory::infer(std::span<const TDesc> inputs,
                                                          const nlohmann::json  &settings) const {
  const auto s = settings.get<RytovCorrectionSettings>();
  require(inputs.size() == 2, "Rytov correction requires fields and a background mask");
  const auto &field = inputs[0], &mask = inputs[1];
  require(field.rank() == 3 && contiguous(field) && field.mem_loc == MemLoc::Device &&
              field.dtype == DType::CF32,
          "Rytov fields must be contiguous device CF32 [frames,y,x]");
  require(field.shape[1] == field.shape[2] && field.shape[1] >= 2 && field.shape[1] <= 8192,
          "Rytov requires square supported fields");
  require(mask.mem_loc == MemLoc::Host && mask.dtype == DType::U8 && contiguous(mask),
          "background mask must be contiguous host U8");
  require((mask.rank() == 2 || (mask.rank() == 3 && mask.shape[0] == 1)) &&
              mask.shape[mask.rank() - 2] == field.shape[1] && mask.shape.back() == field.shape[2],
          "mask must contain one image matching the recovered field");
  require(s.polynomial_degree >= 0 && s.polynomial_degree <= 8 && s.sample_step > 0,
          "invalid polynomial degree or sample step");
  require(s.background_margin >= 0 &&
              static_cast<size_t>(s.background_margin) < field.shape[1] / 2 &&
              s.mask_threshold >= 0 && s.mask_threshold < 255,
          "invalid background margin or mask threshold");
  require(s.gradient_margin >= 0 && static_cast<size_t>(s.gradient_margin) < field.shape[1] / 2,
          "invalid gradient border margin");
  require(std::isfinite(s.amplitude_floor) && s.amplitude_floor > 0 && s.amplitude_floor <= 1 &&
              std::isfinite(s.damping_global) && s.damping_global >= 0 &&
              std::isfinite(s.damping_local) && s.damping_local >= 0,
          "invalid Rytov damping or amplitude floor");
  return {.input_descs   = {field, mask},
          .output_descs  = {TDesc(field.shape, field.dtype, field.mem_loc)},
          .owned_inputs  = {false, false},
          .owned_outputs = {false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> RytovCorrectionFactory::create(std::span<const TDesc> inputs,
                                                          const nlohmann::json  &settings,
                                                          const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<RytovCorrection>(inputs[0], settings.get<RytovCorrectionSettings>(),
                                           ctx.stream);
}

} // namespace holotask::syncs
