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

#include "holonp/fft2.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace holonp {

// -----------------------------------------------------------------------------
// JSON Serialization
// -----------------------------------------------------------------------------

void to_json(nlohmann::json &j, const FFT2Settings &s) {
  j = nlohmann::json{{"axes", s.axes}, {"norm", s.norm}};
}

void from_json(const nlohmann::json &j, FFT2Settings &s) {
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
  else
    s.axes.clear();
  if (j.contains("norm"))
    j.at("norm").get_to(s.norm);
  else
    s.norm = FftNorm::Backward;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("FFT2: " + msg);
}

inline float get_norm_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward)
    return 1.0f;
  const double n = static_cast<double>(n_fft);
  if (norm == FftNorm::Forward)
    return static_cast<float>(1.0 / n);
  return static_cast<float>(1.0 / std::sqrt(n));
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
                                const std::vector<size_t> &in_strides,
                                const std::vector<size_t> &out_strides,
                                const std::vector<int> &outer_dims, size_t dim_idx,
                                size_t current_in, size_t current_out,
                                std::vector<LaunchOffset> &out_offsets) {
  if (dim_idx == outer_dims.size()) {
    out_offsets.push_back({current_in, current_out});
    return;
  }
  int d = outer_dims[dim_idx];
  for (size_t i = 0; i < shape[d]; ++i) {
    generate_offsets_recursive(shape, in_strides, out_strides, outer_dims, dim_idx + 1,
                               current_in + i * in_strides[d], current_out + i * out_strides[d],
                               out_offsets);
  }
}

__global__ void scale_kernel(cuFloatComplex *__restrict__ data, size_t n, float scale) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

} // namespace

// -----------------------------------------------------------------------------
// FFT2 Implementation
// -----------------------------------------------------------------------------

FFT2::FFT2(const FFT2Settings &settings, curaii::CufftHandle &&plan, size_t n_fft,
           std::vector<LaunchOffset> offsets, cudaStream_t stream)
    : settings_(settings), plan_(std::move(plan)), n_fft_(n_fft), offsets_(std::move(offsets)),
      stream_(stream) {}

holoflow::core::OpResult FFT2::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata_base = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *odata_base = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());

  // Loop over outer fragmented memory blocks
  for (const auto &offset : offsets_) {
    auto *in_ptr  = reinterpret_cast<cuFloatComplex *>(idata_base + offset.in_bytes);
    auto *out_ptr = reinterpret_cast<cuFloatComplex *>(odata_base + offset.out_bytes);
    CUFFT_CHECK(cufftXtExec(plan_.get(), in_ptr, out_ptr, CUFFT_FORWARD));
  }

  // Normalization applies to the entire contiguous output buffer at once
  const float scale = get_norm_scale(settings_.norm, n_fft_);
  if (scale != 1.0f) {
    auto        *odata_full = reinterpret_cast<cuFloatComplex *>(odata_base);
    const size_t total      = ctx.outputs[0].desc.num_elements();
    const int    block      = 256;
    const int    grid       = (int)((total + block - 1) / block);
    scale_kernel<<<grid, block, 0, stream_>>>(odata_full, total, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

// -----------------------------------------------------------------------------
// Factory Implementation
// -----------------------------------------------------------------------------

holoflow::core::InferResult FFT2Factory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<FFT2Settings>();
  check(input_descs.size() == 1, "expected 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "device memory only");
  check(idesc.dtype == holoflow::core::DType::CF32, "input must be CF32");
  check(idesc.shape.size() >= 2, "ndim must be >= 2");

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{.input_descs   = {idesc},
                                     .output_descs  = {odesc},
                                     .in_place      = {},
                                     .owned_inputs  = {false},
                                     .owned_outputs = {false},
                                     .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
FFT2Factory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {

  const auto   settings  = jsettings.get<FFT2Settings>();
  const auto  &idesc     = input_descs[0];
  const int    ndim      = static_cast<int>(idesc.shape.size());
  const size_t elem_size = sizeof(cuFloatComplex);

  const size_t h           = idesc.shape[ndim - 2];
  const size_t w           = idesc.shape[ndim - 1];
  const size_t n_fft_elems = h * w;

  auto in_strides_bytes = get_strides_bytes(idesc);

  // Output is physically dense
  std::vector<size_t> out_strides_bytes(ndim);
  size_t acc = elem_size;
  for (int i = ndim - 1; i >= 0; --i) {
    out_strides_bytes[i] = acc;
    acc *= idesc.shape[i];
  }

  if (in_strides_bytes[ndim - 2] % in_strides_bytes[ndim - 1] != 0) {
    throw std::invalid_argument(
        "FFT2: Last two dimensions must have compatible strides (H stride % W stride == 0)");
  }

  const long long istride   = static_cast<long long>(in_strides_bytes[ndim - 1] / elem_size);
  const long long inembed_h = static_cast<long long>(in_strides_bytes[ndim - 2] / in_strides_bytes[ndim - 1]);

  // 1. Group contiguous batch dimensions
  struct BatchGroup {
    size_t    size;
    long long idist_elem;
    long long odist_elem;
    int       start_dim;
    int       end_dim;
  };

  std::vector<BatchGroup> groups;
  if (ndim > 2) {
    int end = ndim - 3;
    int start = end;
    for (int i = ndim - 4; i >= 0; --i) {
      if (in_strides_bytes[i] == in_strides_bytes[i + 1] * idesc.shape[i + 1]) {
        start = i;
      } else {
        size_t size = 1;
        for (int d = start; d <= end; ++d) size *= idesc.shape[d];
        groups.push_back({size, static_cast<long long>(in_strides_bytes[end] / elem_size),
                          static_cast<long long>(out_strides_bytes[end] / elem_size), start, end});
        end = i;
        start = i;
      }
    }
    size_t size = 1;
    for (int d = start; d <= end; ++d) size *= idesc.shape[d];
    groups.push_back({size, static_cast<long long>(in_strides_bytes[end] / elem_size),
                      static_cast<long long>(out_strides_bytes[end] / elem_size), start, end});
  }

  // 2. Select the group that gives the largest inner batch
  BatchGroup best_group = {1, static_cast<long long>(n_fft_elems),
                           static_cast<long long>(n_fft_elems), -1, -1};
  for (const auto &g : groups) {
    if (g.size > best_group.size)
      best_group = g;
  }

  // 3. Keep the non-grouped dims for the outer loop
  std::vector<int> outer_dims;
  for (int i = 0; i < ndim - 2; ++i) {
    if (i < best_group.start_dim || i > best_group.end_dim) {
      outer_dims.push_back(i);
    }
  }

  // 4. Precalculate offsets for the outer loop launches
  std::vector<LaunchOffset> offsets;
  generate_offsets_recursive(idesc.shape, in_strides_bytes, out_strides_bytes, outer_dims, 0, 0, 0,
                             offsets);
  if (offsets.empty()) {
    offsets.push_back({0, 0});
  }

  // 5. Create Plan
  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  size_t work_size = 0;

  long long n[2]       = {static_cast<long long>(h), static_cast<long long>(w)};
  long long inembed[2] = {0, inembed_h};
  long long onembed[2] = {static_cast<long long>(h), static_cast<long long>(w)};

  CUFFT_CHECK(cufftXtMakePlanMany(
      plan.get(), 2, n, inembed, istride, best_group.idist_elem, CUDA_C_32F, onembed, 1,
      best_group.odist_elem, CUDA_C_32F, static_cast<long long>(best_group.size), &work_size, CUDA_C_32F));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFT2(settings, std::move(plan), n_fft_elems, std::move(offsets), ctx.stream));
}

} // namespace holonp