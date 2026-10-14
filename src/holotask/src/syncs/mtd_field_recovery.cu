// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/mtd_field_recovery.hh"
#include "syncs/mtd_common.cuh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const MTDFieldRecoverySettings &s) {
  j = {{"half_size", s.half_size},
       {"center_x", s.center_x},
       {"center_y", s.center_y},
       {"tukey_alpha", s.tukey_alpha}};
}
void from_json(const nlohmann::json &j, MTDFieldRecoverySettings &s) {
  s.half_size   = j.value("half_size", 125);
  s.center_x    = j.value("center_x", 247);
  s.center_y    = j.value("center_y", 247);
  s.tukey_alpha = j.value("tukey_alpha", 0.1f);
}

namespace {
using namespace mtd_detail;

template <typename T>
__global__ void prepare_frame(const T *input, const float *window, cuFloatComplex *output,
                              int size) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= size * size)
    return;
  const int x = i % size, y = i / size;
  const int source = ((y + size / 2) % size) * size + (x + size / 2) % size;
  output[i]        = make_cuFloatComplex(static_cast<float>(input[source]) * window[source], 0.0f);
}

__global__ void crop_order(const cuFloatComplex *spectrum, const float *window,
                           cuFloatComplex *order, int size, int side, int left, int top) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  const int  x = (left + i % side + size / 2) % size;
  const int  y = (top + i / side + size / 2) % size;
  const auto v = spectrum[y * size + x];
  order[i]     = make_cuFloatComplex(v.x * window[i], v.y * window[i]);
}

__global__ void find_peak(const cuFloatComplex *order, float *position, int side) {
  __shared__ float   norms[block_size];
  __shared__ int     indexes[block_size];
  const unsigned int tid   = threadIdx.x;
  float              best  = -1.0f;
  int                index = 0;
  for (int i = static_cast<int>(tid); i < side * side; i += static_cast<int>(block_size)) {
    const auto  value = order[i];
    const float norm  = value.x * value.x + value.y * value.y;
    if (norm > best || (norm == best && i < index)) {
      best  = norm;
      index = i;
    }
  }
  norms[tid]   = best;
  indexes[tid] = index;
  __syncthreads();
  for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride &&
        (norms[tid + stride] > norms[tid] ||
         (norms[tid + stride] == norms[tid] && indexes[tid + stride] < indexes[tid]))) {
      norms[tid]   = norms[tid + stride];
      indexes[tid] = indexes[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    position[0] = static_cast<float>(indexes[0] % side);
    position[1] = static_cast<float>(indexes[0] / side);
  }
}

__global__ void recenter(const cuFloatComplex *order, const float *position, cuFloatComplex *output,
                         int side) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= side * side)
    return;
  const int x = (i % side + static_cast<int>(position[0])) % side;
  const int y = (i / side + static_cast<int>(position[1])) % side;
  output[i]   = order[y * side + x];
}

__global__ void shift_field(const cuFloatComplex *input, cuFloatComplex *output, int side) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < side * side)
    output[i] = input[((i / side + side / 2) % side) * side + (i % side + side / 2) % side];
}

class MTDFieldRecovery : public ISyncTask {
public:
  MTDFieldRecovery(TDesc input, MTDFieldRecoverySettings settings, cudaStream_t stream)
      : input_(std::move(input)), settings_(settings), stream_(stream),
        size_(static_cast<int>(input_.shape[1])), side_(2 * settings.half_size),
        raw_(curaii::make_unique_device_ptr<std::byte>(
            input_.shape[1] * input_.shape[2] * size_of(input_.dtype), stream)),
        spectrum_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(size_) * size_,
                                                                 stream)),
        order_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(side_) * side_,
                                                              stream)),
        field_(curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(side_) * side_,
                                                              stream)),
        holo_window_(
            curaii::make_unique_device_ptr<float>(static_cast<size_t>(size_) * size_, stream)),
        order_window_(
            curaii::make_unique_device_ptr<float>(static_cast<size_t>(side_) * side_, stream)) {
    plan2(forward_, size_, stream_);
    plan2(inverse_, side_, stream_);
    const auto holo_window  = window(size_, settings_.tukey_alpha);
    const auto order_window = window(side_, settings_.tukey_alpha);
    CUDA_CHECK(cudaMemcpyAsync(holo_window_.get(), holo_window.data(),
                               holo_window.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaMemcpyAsync(order_window_.get(), order_window.data(),
                               order_window.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_)); // Host windows go out of scope here.
  }

  OpResult execute(SyncCtx &ctx) override {
    auto        *fields      = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
    auto        *positions   = reinterpret_cast<float *>(ctx.outputs[1].data());
    const size_t plane_bytes = static_cast<size_t>(size_) * size_ * size_of(input_.dtype);
    for (size_t frame = 0; frame < input_.shape[0]; ++frame) {
      if (cancelled(ctx)) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return OpResult::Cancelled;
      }
      const std::byte *input = ctx.inputs[0].data() + frame * plane_bytes;
      if (input_.mem_loc == MemLoc::Host) {
        CUDA_CHECK(
            cudaMemcpyAsync(raw_.get(), input, plane_bytes, cudaMemcpyHostToDevice, stream_));
        input = raw_.get();
      }
      switch (input_.dtype) {
      case DType::U8:
        prepare_frame<<<blocks(static_cast<size_t>(size_) * size_), block_size, 0, stream_>>>(
            reinterpret_cast<const uint8_t *>(input), holo_window_.get(), spectrum_.get(), size_);
        break;
      case DType::U16:
        prepare_frame<<<blocks(static_cast<size_t>(size_) * size_), block_size, 0, stream_>>>(
            reinterpret_cast<const uint16_t *>(input), holo_window_.get(), spectrum_.get(), size_);
        break;
      case DType::F32:
        prepare_frame<<<blocks(static_cast<size_t>(size_) * size_), block_size, 0, stream_>>>(
            reinterpret_cast<const float *>(input), holo_window_.get(), spectrum_.get(), size_);
        break;
      default:
        throw std::logic_error("MTDFieldRecovery: invalid input dtype");
      }
      CUFFT_CHECK(cufftExecC2C(forward_.get(), spectrum_.get(), spectrum_.get(), CUFFT_FORWARD));
      crop_order<<<blocks(static_cast<size_t>(side_) * side_), block_size, 0, stream_>>>(
          spectrum_.get(), order_window_.get(), order_.get(), size_, side_,
          settings_.center_x - settings_.half_size, settings_.center_y - settings_.half_size);
      find_peak<<<1, block_size, 0, stream_>>>(order_.get(), positions + 2 * frame, side_);
      recenter<<<blocks(static_cast<size_t>(side_) * side_), block_size, 0, stream_>>>(
          order_.get(), positions + 2 * frame, field_.get(), side_);
      CUFFT_CHECK(cufftExecC2C(inverse_.get(), field_.get(), field_.get(), CUFFT_INVERSE));
      shift_field<<<blocks(static_cast<size_t>(side_) * side_), block_size, 0, stream_>>>(
          field_.get(), fields + frame * static_cast<size_t>(side_) * side_, side_);
      CUDA_CHECK(cudaGetLastError());
    }
    return OpResult::Ok;
  }

private:
  TDesc                     input_;
  MTDFieldRecoverySettings  settings_;
  cudaStream_t              stream_;
  int                       size_, side_;
  DevicePtr<std::byte>      raw_;
  DevicePtr<cuFloatComplex> spectrum_, order_, field_;
  DevicePtr<float>          holo_window_, order_window_;
  curaii::CufftHandle       forward_, inverse_;
};
} // namespace

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult MTDFieldRecoveryFactory::infer(std::span<const TDesc> inputs,
                                                           const nlohmann::json  &settings) const {
  const auto s = settings.get<MTDFieldRecoverySettings>();
  require(inputs.size() == 1, "field recovery requires one hologram batch");
  const auto &in = inputs[0];
  require(in.rank() == 3 && contiguous(in), "holograms must be contiguous [frames,y,x]");
  require(in.dtype == DType::U8 || in.dtype == DType::U16 || in.dtype == DType::F32,
          "holograms must be U8, U16, or F32");
  require(in.shape[1] == in.shape[2] && in.shape[1] >= 2 && in.shape[1] <= 32768 &&
              in.shape[1] % 2 == 0,
          "holograms must be square with an even supported size");
  const int size = static_cast<int>(in.shape[1]);
  require(s.half_size > 0 && s.half_size <= size / 2,
          "Fourier half_size exceeds hologram dimensions");
  require(s.center_x >= s.half_size && s.center_x <= size - s.half_size &&
              s.center_y >= s.half_size && s.center_y <= size - s.half_size,
          "Fourier crop lies outside the hologram");
  require(std::isfinite(s.tukey_alpha) && s.tukey_alpha >= 0 && s.tukey_alpha <= 1,
          "Tukey alpha must be between zero and one");
  const size_t side = static_cast<size_t>(2 * s.half_size);
  return {.input_descs   = {in},
          .output_descs  = {TDesc({in.shape[0], side, side}, DType::CF32, MemLoc::Device),
                            TDesc({in.shape[0], 2}, DType::F32, MemLoc::Device)},
          .owned_inputs  = {false},
          .owned_outputs = {false, false},
          .kind          = TaskKind::Sync};
}

std::unique_ptr<ISyncTask> MTDFieldRecoveryFactory::create(std::span<const TDesc> inputs,
                                                           const nlohmann::json  &settings,
                                                           const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<MTDFieldRecovery>(inputs[0], settings.get<MTDFieldRecoverySettings>(),
                                            ctx.stream);
}

} // namespace holotask::syncs
