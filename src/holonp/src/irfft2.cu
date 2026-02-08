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

#include "holonp/irfft2.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuComplex.h>

namespace holonp {

void to_json(nlohmann::json &j, const IRFFT2Settings &s) {
  j = nlohmann::json{{"axes", s.axes}, {"norm", s.norm}};
}

void from_json(const nlohmann::json &j, IRFFT2Settings &s) {
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
  else
    s.axes.clear();
  if (j.contains("norm"))
    j.at("norm").get_to(s.norm);
  else
    s.norm = FftNorm::Backward;
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("IRFFT2: " + msg);
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0)
    axis += ndim;
  return axis;
}

inline float inverse_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward)
    return static_cast<float>(1.0 / static_cast<double>(n_fft));
  if (norm == FftNorm::Forward)
    return 1.0f;
  return static_cast<float>(1.0 / std::sqrt(n_fft));
}

std::vector<size_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty())
    return desc.strides;
  std::vector<size_t> strides(desc.shape.size());
  size_t              acc = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
}

void generate_offsets_recursive(const std::vector<size_t> &shape,
                                const std::vector<size_t> &strides, int dim_idx,
                                size_t current_offset, std::vector<size_t> &out_offsets) {
  if (dim_idx < 0) {
    out_offsets.push_back(current_offset);
    return;
  }
  for (size_t i = 0; i < shape[dim_idx]; ++i) {
    generate_offsets_recursive(shape, strides, dim_idx - 1, current_offset + i * strides[dim_idx],
                               out_offsets);
  }
}

std::array<int, 2> resolve_axes(const std::vector<int> &axes, int ndim) {
  check(ndim >= 2, "ndim must be >= 2");

  std::array<int, 2> resolved{};
  if (axes.empty()) {
    resolved[0] = ndim - 2;
    resolved[1] = ndim - 1;
  } else {
    check(axes.size() == 2, "expected exactly 2 axes");
    resolved[0] = normalize_axis(axes[0], ndim);
    resolved[1] = normalize_axis(axes[1], ndim);
  }

  check(resolved[0] >= 0 && resolved[0] < ndim, "axis 0 out of range");
  check(resolved[1] >= 0 && resolved[1] < ndim, "axis 1 out of range");
  check(resolved[0] < resolved[1], "axes must be ordered");
  check(resolved[0] == ndim - 2 && resolved[1] == ndim - 1,
        "only trailing axes (-2, -1) are supported");
  return resolved;
}

__global__ void scale_kernel(float *__restrict__ data, size_t n, float scale) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

} // namespace

IRFFT2::IRFFT2(const IRFFT2Settings &settings, curaii::CufftHandle &&plan, size_t n_fft_elems,
               size_t total_out_elems, std::vector<size_t> input_offsets,
               size_t output_stride_bytes, cudaStream_t stream)
    : settings_(settings), plan_(std::move(plan)), n_fft_elems_(n_fft_elems),
      total_out_elems_(total_out_elems), input_offsets_(std::move(input_offsets)),
      output_stride_bytes_(output_stride_bytes), stream_(stream) {}

holoflow::core::OpResult IRFFT2::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata_base = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *odata_base = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());

  for (size_t i = 0; i < input_offsets_.size(); ++i) {
    auto *in_ptr  = reinterpret_cast<cuFloatComplex *>(idata_base + input_offsets_[i]);
    auto *out_ptr = reinterpret_cast<float *>(odata_base + i * output_stride_bytes_);
    CUFFT_CHECK(cufftXtExec(plan_.get(), in_ptr, out_ptr, CUFFT_INVERSE));
  }

  const float scale = inverse_scale(settings_.norm, n_fft_elems_);
  if (scale != 1.0f) {
    auto     *odata = reinterpret_cast<float *>(ctx.outputs[0].data());
    const int block = 256;
    const int grid  = static_cast<int>((total_out_elems_ + block - 1) / block);
    scale_kernel<<<grid, block, 0, stream_>>>(odata, total_out_elems_, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult IRFFT2Factory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<IRFFT2Settings>();
  check(input_descs.size() == 1, "expected 1 input");

  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "device memory only");
  check(idesc.dtype == holoflow::core::DType::CF32, "input must be CF32");

  const int  ndim = static_cast<int>(idesc.shape.size());
  const auto axes = resolve_axes(settings.axes, ndim);

  const size_t h      = idesc.shape[static_cast<size_t>(axes[0])];
  const size_t w_freq = idesc.shape[static_cast<size_t>(axes[1])];
  check(h > 0 && w_freq > 0, "invalid FFT shape");

  const size_t w_real                 = (w_freq - 1) * 2;
  auto         shape                  = idesc.shape;
  shape[static_cast<size_t>(axes[1])] = w_real;

  holoflow::core::TDesc odesc(shape, holoflow::core::DType::F32, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{.input_descs   = {idesc},
                                     .output_descs  = {odesc},
                                     .in_place      = {},
                                     .owned_inputs  = {false},
                                     .owned_outputs = {false},
                                     .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
IRFFT2Factory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  const auto settings  = jsettings.get<IRFFT2Settings>();

  const auto &idesc = input_descs[0];
  const auto &odesc = infer_res.output_descs[0];

  const int    ndim = static_cast<int>(idesc.shape.size());
  const auto   axes = resolve_axes(settings.axes, ndim);
  const size_t h    = odesc.shape[static_cast<size_t>(axes[0])];
  const size_t w    = odesc.shape[static_cast<size_t>(axes[1])];
  const size_t w_in = idesc.shape[static_cast<size_t>(axes[1])];

  const size_t n_fft = h * w;
  const size_t esize = holoflow::core::size_of(idesc.dtype);

  auto strides_bytes = get_strides_bytes(idesc);

  check(strides_bytes[axes[1]] % esize == 0, "unsupported stride on FFT axis");
  check(strides_bytes[axes[0]] % strides_bytes[axes[1]] == 0,
        "H axis stride must be a multiple of W stride");

  const int istride =
      static_cast<int>(strides_bytes[axes[1]] / esize); // complex elements between width samples
  const int inembed_w =
      static_cast<int>(strides_bytes[axes[0]] / strides_bytes[axes[1]]); // stored width (freq)

  long long idist         = static_cast<long long>(h * w_in);
  size_t    inner_batches = 1;
  int       first_outer   = axes[0];

  if (first_outer > 0) {
    int k = first_outer - 1;
    idist = static_cast<long long>(strides_bytes[k] / esize);
    inner_batches *= idesc.shape[static_cast<size_t>(k)];
    first_outer = k;
    for (int i = k - 1; i >= 0; --i) {
      size_t expected = strides_bytes[i + 1] * idesc.shape[i + 1];
      if (strides_bytes[i] == expected) {
        inner_batches *= idesc.shape[static_cast<size_t>(i)];
        first_outer = i;
      } else {
        break;
      }
    }
  }

  std::vector<size_t> offsets;
  if (first_outer > 0) {
    generate_offsets_recursive(idesc.shape, strides_bytes, first_outer - 1, 0, offsets);
  } else {
    offsets.push_back(0);
  }

  const auto ll_max = std::numeric_limits<long long>::max();
  check(n_fft <= static_cast<size_t>(ll_max), "FFT size exceeds cuFFT limits");
  check(inner_batches > 0, "invalid batch size");
  check(inner_batches <= static_cast<size_t>(ll_max), "batch size exceeds cuFFT limits");
  check(idist > 0 && idist <= ll_max, "invalid input distance");

  const size_t output_stride_bytes = inner_batches * n_fft * sizeof(float);
  const size_t total_out           = odesc.num_elements();

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  size_t work_size = 0;

  long long n[2]       = {static_cast<long long>(h), static_cast<long long>(w)};
  long long inembed[2] = {0, static_cast<long long>(inembed_w)};
  long long onembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, n, inembed, istride, idist, CUDA_C_32F, onembed, 1,
                                  static_cast<long long>(n_fft), CUDA_R_32F,
                                  static_cast<long long>(inner_batches), &work_size, CUDA_C_32F));

  return std::unique_ptr<IRFFT2>(new IRFFT2(settings, std::move(plan), n_fft, total_out,
                                            std::move(offsets), output_stride_bytes, ctx.stream));
}

} // namespace holonp
