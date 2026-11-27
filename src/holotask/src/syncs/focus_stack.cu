// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holotask/syncs/focus_stack.hh"
#include <cmath>
#include <cuComplex.h>
#include <limits>
#include <stdexcept>

namespace holotask::syncs {
using namespace holoflow::core;

void to_json(nlohmann::json &j, const FocusStackSettings &s) {
  j = {{"wavelength_m", s.wavelength_m},
       {"pixel_pitch_x_m", s.pixel_pitch_x_m},
       {"pixel_pitch_y_m", s.pixel_pitch_y_m},
       {"z_min_m", s.z_min_m},
       {"z_max_m", s.z_max_m},
       {"planes", s.planes},
       {"padding", s.padding},
       {"isolate_order", s.isolate_order},
       {"carrier_x", s.carrier_x},
       {"carrier_y", s.carrier_y},
       {"order_radius", s.order_radius}};
}
void from_json(const nlohmann::json &j, FocusStackSettings &s) {
  const FocusStackSettings defaults;
  s.wavelength_m    = j.value("wavelength_m", defaults.wavelength_m);
  s.pixel_pitch_x_m = j.value("pixel_pitch_x_m", defaults.pixel_pitch_x_m);
  s.pixel_pitch_y_m = j.value("pixel_pitch_y_m", defaults.pixel_pitch_y_m);
  s.z_min_m         = j.value("z_min_m", defaults.z_min_m);
  s.z_max_m         = j.value("z_max_m", defaults.z_max_m);
  s.planes          = j.value("planes", defaults.planes);
  s.padding         = j.value("padding", defaults.padding);
  s.isolate_order   = j.value("isolate_order", defaults.isolate_order);
  s.carrier_x       = j.value("carrier_x", defaults.carrier_x);
  s.carrier_y       = j.value("carrier_y", defaults.carrier_y);
  s.order_radius    = j.value("order_radius", defaults.order_radius);
}

namespace {
constexpr unsigned block_size = 256;
constexpr double   pi         = 3.14159265358979323846;
unsigned blocks(size_t n) { return static_cast<unsigned>((n + block_size - 1) / block_size); }
void     require(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(std::string("FocusStack: ") + message);
}

template <typename T>
__global__ void prepare(const T *input, cuFloatComplex *output, int w, int h, int pw, int ph) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= pw * ph)
    return;
  const int   x = i % pw - (pw - w) / 2, y = i / pw - (ph - h) / 2;
  const float scale = sizeof(T) == 1 ? 1.0f / 255.0f : 1.0f / 65535.0f;
  output[i]         = make_cuFloatComplex(
      x >= 0 && x < w && y >= 0 && y < h ? static_cast<float>(input[y * w + x]) * scale : 0.0f,
      0.0f);
}

// Recenter an optional off-axis order without changing the sensor sampling pitch.
__global__ void propagate(const cuFloatComplex *spectrum, cuFloatComplex *output, int w, int h,
                          double dx, double dy, double lambda, double z, int cx, int cy,
                          double radius_x, double radius_y, bool isolate) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= w * h)
    return;
  const int ix = i % w, iy = i / w;
  const int kx = ix <= w / 2 ? ix : ix - w, ky = iy <= h / 2 ? iy : iy - h;
  double    weight = 1.0;
  if (isolate) {
    const double radius =
        sqrt((kx / radius_x) * (kx / radius_x) + (ky / radius_y) * (ky / radius_y));
    weight = radius < 1.0 ? (radius < 0.8 ? 1.0 : 0.5 * (1 + cos(pi * (radius - 0.8) / 0.2))) : 0.0;
  }
  const double fx = kx / (w * dx), fy = ky / (h * dy);
  const double argument = 1.0 - lambda * lambda * (fx * fx + fy * fy);
  if (argument <= 0 || weight == 0) {
    output[i] = make_cuFloatComplex(0, 0);
    return;
  }
  // Omit the global piston; normalized intensity is independent of it. The rationalized
  // expression avoids subtracting nearly equal numbers at small spatial frequencies.
  const double phase = -2.0 * pi * z * lambda * (fx * fx + fy * fy) / (sqrt(argument) + 1.0);
  double       sine, cosine;
  sincos(phase, &sine, &cosine);
  const int  sx = (ix + cx + w) % w, sy = (iy + cy + h) % h;
  const auto v = spectrum[sy * w + sx];
  output[i]    = make_cuFloatComplex(static_cast<float>(weight * (v.x * cosine - v.y * sine)),
                                     static_cast<float>(weight * (v.x * sine + v.y * cosine)));
}

__global__ void intensity(const cuFloatComplex *field, float *output, int w, int h, int pw,
                          int ph) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= w * h)
    return;
  const auto  v     = field[(i / w + (ph - h) / 2) * pw + i % w + (pw - w) / 2];
  const float scale = 1.0f / static_cast<float>(pw * ph);
  output[i]         = (v.x * scale) * (v.x * scale) + (v.y * scale) * (v.y * scale);
}

class FocusStack : public ISyncTask {
public:
  FocusStack(TDesc desc, FocusStackSettings settings, cudaStream_t stream)
      : desc_(std::move(desc)), s_(settings), stream_(stream), w_(static_cast<int>(desc_.shape[2])),
        h_(static_cast<int>(desc_.shape[1])), pw_(w_ * (s_.padding ? 2 : 1)),
        ph_(h_ * (s_.padding ? 2 : 1)),
        raw_(curaii::make_unique_device_ptr<std::byte>(desc_.num_bytes(), stream)),
        spectrum_(
            curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(pw_) * ph_, stream)),
        field_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(pw_) * ph_,
                                                              stream)) {
    size_t work = 0;
    CUFFT_CHECK(cufftMakePlan2d(plan_.get(), ph_, pw_, CUFFT_C2C, &work));
    CUFFT_CHECK(cufftSetStream(plan_.get(), stream_));
  }
  OpResult execute(SyncCtx &ctx) override {
    if (ctx.cancelled && ctx.cancelled->load())
      return OpResult::Cancelled;
    const auto *input = ctx.inputs[0].data();
    if (desc_.mem_loc == MemLoc::Host) {
      CUDA_CHECK(
          cudaMemcpyAsync(raw_.get(), input, desc_.num_bytes(), cudaMemcpyHostToDevice, stream_));
      input = raw_.get();
    }
    const auto count = static_cast<size_t>(pw_) * ph_;
    if (desc_.dtype == DType::U8)
      prepare<<<blocks(count), block_size, 0, stream_>>>(reinterpret_cast<const uint8_t *>(input),
                                                         spectrum_.get(), w_, h_, pw_, ph_);
    else
      prepare<<<blocks(count), block_size, 0, stream_>>>(reinterpret_cast<const uint16_t *>(input),
                                                         spectrum_.get(), w_, h_, pw_, ph_);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecC2C(plan_.get(), spectrum_.get(), spectrum_.get(), CUFFT_FORWARD));
    const int factor = s_.padding ? 2 : 1;
    auto     *output = reinterpret_cast<float *>(ctx.outputs[0].data());
    for (int plane = 0; plane < s_.planes; ++plane) {
      if (ctx.cancelled && ctx.cancelled->load()) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return OpResult::Cancelled;
      }
      const double z = s_.z_min_m + (s_.z_max_m - s_.z_min_m) * plane / (s_.planes - 1);
      propagate<<<blocks(count), block_size, 0, stream_>>>(
          spectrum_.get(), field_.get(), pw_, ph_, s_.pixel_pitch_x_m, s_.pixel_pitch_y_m,
          s_.wavelength_m, z, s_.isolate_order ? s_.carrier_x * factor : 0,
          s_.isolate_order ? s_.carrier_y * factor : 0,
          static_cast<double>(s_.order_radius * factor),
          static_cast<double>(s_.order_radius * factor), s_.isolate_order);
      CUDA_CHECK(cudaGetLastError());
      CUFFT_CHECK(cufftExecC2C(plan_.get(), field_.get(), field_.get(), CUFFT_INVERSE));
      intensity<<<blocks(static_cast<size_t>(w_) * h_), block_size, 0, stream_>>>(
          field_.get(), output + static_cast<size_t>(plane) * w_ * h_, w_, h_, pw_, ph_);
      CUDA_CHECK(cudaGetLastError());
    }
    return OpResult::Ok;
  }

private:
  TDesc                                     desc_;
  FocusStackSettings                        s_;
  cudaStream_t                              stream_;
  int                                       w_, h_, pw_, ph_;
  curaii::unique_device_ptr<std::byte>      raw_;
  curaii::unique_device_ptr<cuFloatComplex> spectrum_, field_;
  curaii::CufftHandle                       plan_;
};
} // namespace

InferResult FocusStackFactory::infer(std::span<const TDesc> inputs,
                                     const nlohmann::json  &settings) const {
  const auto s = settings.get<FocusStackSettings>();
  require(inputs.size() == 1, "expected one hologram");
  const auto &in = inputs[0];
  require(in.rank() == 3 && in.shape[0] == 1 && (in.dtype == DType::U8 || in.dtype == DType::U16),
          "input must be one U8/U16 hologram [1,Y,X]");
  size_t stride = size_of(in.dtype);
  for (size_t axis = in.rank(); axis-- > 0;) {
    require(in.strides[axis] == stride && in.shape[axis] > 0 && in.shape[axis] <= 16384,
            "input must be contiguous with dimensions up to 16384");
    stride *= in.shape[axis];
  }
  require(std::isfinite(s.wavelength_m) && s.wavelength_m > 0 && std::isfinite(s.pixel_pitch_x_m) &&
              s.pixel_pitch_x_m > 0 && std::isfinite(s.pixel_pitch_y_m) && s.pixel_pitch_y_m > 0,
          "wavelength and pixel pitches must be positive and finite");
  require(std::isfinite(s.z_min_m) && std::isfinite(s.z_max_m) && s.z_min_m < s.z_max_m &&
              s.planes >= 2 && s.planes <= 2048,
          "Z minimum must be below maximum; choose 2 to 2048 planes");
  const double cx = s.carrier_x, cy = s.carrier_y, radius = s.order_radius;
  require(!s.isolate_order || (radius > 0 && cx * cx + cy * cy > radius * radius &&
                               std::abs(cx) + radius < in.shape[2] / 2.0 &&
                               std::abs(cy) + radius < in.shape[1] / 2.0),
          "Fourier order must exclude DC and fit inside the spectrum");
  const size_t bytes = static_cast<size_t>(s.planes) * in.shape[1] * in.shape[2] * sizeof(float);
  require(bytes <= size_t{4} * 1024 * 1024 * 1024,
          "focus stack exceeds the 4 GiB limit; reduce planes");
  return {.input_descs   = {in},
          .output_descs  = {TDesc({static_cast<size_t>(s.planes), in.shape[1], in.shape[2]},
                                  DType::F32, MemLoc::Device)},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> FocusStackFactory::create(std::span<const TDesc> inputs,
                                                     const nlohmann::json  &settings,
                                                     const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<FocusStack>(inputs[0], settings.get<FocusStackSettings>(), ctx.stream);
}
} // namespace holotask::syncs
