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

#include "holonp/fft.hh"

#include <cuComplex.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"

namespace holonp {

void to_json(nlohmann::json &j, const FFTSettings &s) {
  j = nlohmann::json{
      {"axis", s.axis},
      {"norm", s.norm},
  };
}

void from_json(const nlohmann::json &j, FFTSettings &s) {
  if (j.contains("axis")) {
    j.at("axis").get_to(s.axis);
  } else {
    s.axis = -1;
  }

  if (j.contains("norm")) {
    j.at("norm").get_to(s.norm);
  } else {
    s.norm = FftNorm::Backward;
  }
}

namespace {

constexpr int kMaxNDim = 16;

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("FFT: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0) {
    axis += ndim;
  }
  return axis;
}

inline float norm_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward) {
    return 1.0f;
  }
  const double n = static_cast<double>(n_fft);
  if (norm == FftNorm::Forward) {
    return static_cast<float>(1.0 / n);
  }
  return static_cast<float>(1.0 / std::sqrt(n));
}

__global__ void real_to_complex_kernel(const float *__restrict__ in,
                                       cuFloatComplex *__restrict__ out, std::int64_t n) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    out[idx] = make_cuFloatComplex(in[idx], 0.0f);
  }
}

__global__ void scale_cf32_kernel(cuFloatComplex *__restrict__ data, std::int64_t n, float scale) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

class FFT : public holoflow::core::ISyncTask {
public:
  FFT(FFTSettings settings, holoflow::core::TDesc idesc, curaii::CufftHandle &&plan,
      size_t total_elems, size_t n_fft, size_t exec_count, size_t exec_stride,
      holoflow::core::DType input_dtype, cudaStream_t stream, DevPtr<cuFloatComplex> d_tmp)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), plan_(std::move(plan)),
        total_elems_(total_elems), n_fft_(n_fft), exec_count_(exec_count),
        exec_stride_(exec_stride), input_dtype_(input_dtype), stream_(stream),
        d_tmp_(std::move(d_tmp)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const FFTSettings           &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_ = stream;
      CUFFT_CHECK(cufftSetStream(plan_.get(), stream_));
    }
  }

private:
  FFTSettings            settings_;
  holoflow::core::TDesc  idesc_;
  curaii::CufftHandle    plan_;
  size_t                 total_elems_;
  size_t                 n_fft_;
  size_t                 exec_count_;
  size_t                 exec_stride_;
  holoflow::core::DType  input_dtype_;
  cudaStream_t           stream_;
  DevPtr<cuFloatComplex> d_tmp_;
};

} // namespace

holoflow::core::OpResult FFT::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = ctx.inputs[0].data();
  auto *odata = ctx.outputs[0].data();

  const auto total_i64 = static_cast<std::int64_t>(total_elems_);
  const int  block     = 256;
  const int  grid      = static_cast<int>((total_i64 + block - 1) / block);

  const cuFloatComplex *in = nullptr;
  if (input_dtype_ == holoflow::core::DType::F32) {
    auto *in_f = reinterpret_cast<const float *>(idata);
    real_to_complex_kernel<<<grid, block, 0, stream_>>>(in_f, d_tmp_.get(), total_i64);
    in = d_tmp_.get();
  } else {
    in = reinterpret_cast<const cuFloatComplex *>(idata);
  }

  auto *out = reinterpret_cast<cuFloatComplex *>(odata);
  for (size_t i = 0; i < exec_count_; ++i) {
    const auto offset  = i * exec_stride_;
    auto      *in_ptr  = const_cast<cuFloatComplex *>(in + offset);
    auto      *out_ptr = out + offset;
    CUFFT_CHECK(cufftXtExec(plan_.get(), in_ptr, out_ptr, CUFFT_FORWARD));
  }

  const float scale = norm_scale(settings_.norm, n_fft_);
  if (scale != 1.0f) {
    scale_cf32_kernel<<<grid, block, 0, stream_>>>(out, total_i64, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult FFTFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<FFTSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "input dtype must be F32 or CF32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  // holoflow::core::TDesc odesc = idesc;
  // odesc.dtype                 = holoflow::core::DType::CF32;

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<FFTSettings>();

  const auto &idesc = input_descs[0];
  const auto  total = product_shape(idesc.shape);
  const int   ndim  = static_cast<int>(idesc.shape.size());

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto n_fft = idesc.shape[static_cast<size_t>(axis)];
  size_t     outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= idesc.shape[static_cast<size_t>(i)];
  }
  size_t inner = 1;
  for (int i = axis + 1; i < ndim; ++i) {
    inner *= idesc.shape[static_cast<size_t>(i)];
  }

  check(n_fft > 0, "invalid FFT length");
  check(n_fft <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT length exceeds cuFFT limits");

  const auto kIntMax = static_cast<size_t>(std::numeric_limits<int>::max());
  check(inner <= kIntMax, "FFT stride exceeds cuFFT limits");

  const bool can_loop_inner    = (outer <= kIntMax) && (n_fft <= kIntMax / inner);
  const bool prefer_loop_inner = inner <= outer;

  int    batch_i     = 0;
  int    istride     = static_cast<int>(inner);
  int    idist       = 0;
  size_t exec_count  = 0;
  size_t exec_stride = 0;

  if (can_loop_inner && prefer_loop_inner) {
    batch_i     = static_cast<int>(outer);
    idist       = static_cast<int>(n_fft * inner);
    exec_count  = inner;
    exec_stride = 1;
  } else {
    batch_i     = static_cast<int>(inner);
    idist       = 1;
    exec_count  = outer;
    exec_stride = n_fft * inner;
  }

  int           rank          = 1;
  long long int n[1]          = {static_cast<long long int>(n_fft)};
  long long int inembed[1]    = {static_cast<long long int>(n_fft)};
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[1]    = {static_cast<long long int>(n_fft)};
  int           ostride       = istride;
  int           odist         = idist;
  cudaDataType  outputtype    = CUDA_C_32F;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), rank, n, inembed, istride, idist, inputtype, onembed,
                                  ostride, odist, outputtype, batch_i, &work_size, executiontype));

  DevPtr<cuFloatComplex> d_tmp;
  if (idesc.dtype == holoflow::core::DType::F32) {
    d_tmp = curaii::make_unique_device_ptr<cuFloatComplex>(total, ctx.stream);
  }

  return std::unique_ptr<holoflow::core::ISyncTask>(new FFT(settings, idesc, std::move(plan), total,
                                                            n_fft, exec_count, exec_stride,
                                                            idesc.dtype, ctx.stream,
                                                            std::move(d_tmp)));
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_fft = dynamic_cast<FFT *>(old_task.get());
  if (old_fft != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<FFTSettings>();
    if (settings == old_fft->settings() && same_desc(input_descs[0], old_fft->idesc())) {
      old_fft->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
