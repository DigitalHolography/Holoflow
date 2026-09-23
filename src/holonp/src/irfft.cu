// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holonp/irfft.hh"
#include "utils/tensor_common.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <cuComplex.h>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"

namespace holonp {

void to_json(nlohmann::json &j, const IRFFTSettings &s) {
  j = nlohmann::json{{"axis", s.axis}, {"n", s.n}, {"norm", s.norm}};
}

void from_json(const nlohmann::json &j, IRFFTSettings &s) {
  s.axis = j.value("axis", -1);
  s.n    = j.value("n", -1);
  if (j.contains("norm"))
    j.at("norm").get_to(s.norm);
  else
    s.norm = FftNorm::Backward;
}

namespace {

constexpr int kMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("IRFFT: " + msg);
}

inline float norm_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward)
    return static_cast<float>(1.0 / static_cast<double>(n_fft));
  if (norm == FftNorm::Forward)
    return 1.0f;
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft)));
}

__global__ void scale_f32_kernel(float *__restrict__ data, std::int64_t n, float scale) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
    data[idx] *= scale;
}

class IRFFT : public holoflow::core::ISyncTask {
public:
  IRFFT(IRFFTSettings settings, holoflow::core::TDesc idesc, curaii::CufftHandle &&plan,
        size_t total_out, size_t n_fft, size_t exec_count, size_t exec_in_stride,
        size_t exec_out_stride, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), plan_(std::move(plan)),
        total_out_(total_out), n_fft_(n_fft), exec_count_(exec_count),
        exec_in_stride_(exec_in_stride), exec_out_stride_(exec_out_stride), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const IRFFTSettings         &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_ = stream;
      CUFFT_CHECK(cufftSetStream(plan_.get(), stream_));
    }
  }

private:
  IRFFTSettings         settings_;
  holoflow::core::TDesc idesc_;
  curaii::CufftHandle   plan_;
  size_t                total_out_;
  size_t                n_fft_;
  size_t                exec_count_;
  size_t                exec_in_stride_;
  size_t                exec_out_stride_;
  cudaStream_t          stream_;
};

} // namespace

holoflow::core::OpResult IRFFT::execute(holoflow::core::SyncCtx &ctx) {
  auto *in  = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data());
  auto *out = reinterpret_cast<float *>(ctx.outputs[0].data());

  for (size_t i = 0; i < exec_count_; ++i) {
    CUFFT_CHECK(cufftExecC2R(plan_.get(), in + i * exec_in_stride_, out + i * exec_out_stride_));
  }

  const float scale = norm_scale(settings_.norm, n_fft_);
  if (scale != 1.0f) {
    const auto    total = static_cast<std::int64_t>(total_out_);
    constexpr int block = 256;
    const int     grid  = static_cast<int>((total + block - 1) / block);
    scale_f32_kernel<<<grid, block, 0, stream_>>>(out, total, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult IRFFTFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<IRFFTSettings>();
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(utils::is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::CF32, "input dtype must be CF32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0 && ndim <= kMaxNDim, "invalid input ndim");
  const int axis = utils::normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");
  check(utils::product_shape(idesc.shape) > 0, "input tensor has zero elements");

  const size_t n_freq = idesc.shape[static_cast<size_t>(axis)];
  check(n_freq > 0, "invalid FFT input length");
  const size_t n_fft = settings.n > 0 ? static_cast<size_t>(settings.n) : 2 * (n_freq - 1);
  check(n_fft > 0, "inverse FFT length must be positive");
  if (settings.n > 0)
    check(settings.n <= static_cast<int>(std::numeric_limits<int>::max()), "FFT length too large");

  auto shape                       = idesc.shape;
  shape[static_cast<size_t>(axis)] = n_fft;
  const holoflow::core::TDesc odesc(shape, holoflow::core::DType::F32,
                                    holoflow::core::MemLoc::Device);
  return holoflow::core::InferResult{.input_descs   = {idesc},
                                     .output_descs  = {odesc},
                                     .in_place      = {},
                                     .owned_inputs  = {false},
                                     .owned_outputs = {false},
                                     .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
IRFFTFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto   infer_result = infer(input_descs, jsettings);
  const auto   settings     = jsettings.get<IRFFTSettings>();
  const auto  &idesc        = input_descs[0];
  const int    ndim         = static_cast<int>(idesc.shape.size());
  const int    axis         = utils::normalize_axis(settings.axis, ndim);
  const size_t n_freq       = idesc.shape[static_cast<size_t>(axis)];
  const size_t n_fft        = infer_result.output_descs[0].shape[static_cast<size_t>(axis)];

  size_t outer = 1;
  for (int i = 0; i < axis; ++i)
    outer *= idesc.shape[static_cast<size_t>(i)];
  size_t inner = 1;
  for (int i = axis + 1; i < ndim; ++i)
    inner *= idesc.shape[static_cast<size_t>(i)];

  const auto max_int = static_cast<size_t>(std::numeric_limits<int>::max());
  check(inner > 0 && inner <= max_int, "FFT stride exceeds cuFFT limits");
  check(n_freq <= max_int && n_fft <= max_int, "FFT length exceeds cuFFT limits");

  const bool loop_inner =
      inner <= outer && outer <= max_int && n_freq <= max_int / inner && n_fft <= max_int / inner;
  int    batch           = 0;
  int    istride         = static_cast<int>(inner);
  int    idist           = 0;
  int    ostride         = static_cast<int>(inner);
  int    odist           = 0;
  size_t exec_count      = 0;
  size_t exec_in_stride  = 0;
  size_t exec_out_stride = 0;
  if (loop_inner) {
    batch           = static_cast<int>(outer);
    idist           = static_cast<int>(n_freq * inner);
    odist           = static_cast<int>(n_fft * inner);
    exec_count      = inner;
    exec_in_stride  = 1;
    exec_out_stride = 1;
  } else {
    check(inner <= max_int, "FFT batch exceeds cuFFT limits");
    batch           = static_cast<int>(inner);
    idist           = 1;
    odist           = 1;
    exec_count      = outer;
    exec_in_stride  = n_freq * inner;
    exec_out_stride = n_fft * inner;
  }

  int                 n[1]       = {static_cast<int>(n_fft)};
  int                 inembed[1] = {static_cast<int>(n_freq)};
  int                 onembed[1] = {static_cast<int>(n_fft)};
  size_t              work_size  = 0;
  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  CUFFT_CHECK(cufftMakePlanMany(plan.get(), 1, n, inembed, istride, idist, onembed, ostride, odist,
                                CUFFT_C2R, batch, &work_size));

  const size_t total_out = (utils::product_shape(idesc.shape) / n_freq) * n_fft;
  return std::unique_ptr<holoflow::core::ISyncTask>(
      new IRFFT(settings, idesc, std::move(plan), total_out, n_fft, exec_count, exec_in_stride,
                exec_out_stride, ctx.stream));
}

std::unique_ptr<holoflow::core::ISyncTask>
IRFFTFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                     std::span<const holoflow::core::TDesc>     input_descs,
                     const nlohmann::json                      &jsettings,
                     const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto *old_irfft = dynamic_cast<IRFFT *>(old_task.get());
  if (old_irfft != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<IRFFTSettings>();
    if (settings == old_irfft->settings() && utils::same_desc(input_descs[0], old_irfft->idesc())) {
      old_irfft->update_stream(ctx.stream);
      return old_task;
    }
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
