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

#include "holonp/mul.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const MulSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, MulSettings &) {}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Mul: " + msg);
  }
}

constexpr int          kMaxNDim = 16;
constexpr std::int64_t kMaxI64  = std::numeric_limits<std::int64_t>::max();

inline std::int64_t to_i64(size_t v, const std::string &msg) {
  check(v <= static_cast<size_t>(kMaxI64), msg);
  return static_cast<std::int64_t>(v);
}

inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const std::string &msg) {
  if (a == 0 || b == 0) {
    return 0;
  }
  check(a <= (kMaxI64 / b), msg);
  return a * b;
}

inline std::vector<std::int64_t> make_contig_strides(std::span<const size_t> shape) {
  const int                 ndim = static_cast<int>(shape.size());
  std::vector<std::int64_t> strides(ndim, 1);
  std::int64_t              acc = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = acc;
    acc        = checked_mul(acc, to_i64(shape[static_cast<size_t>(i)], "shape too large"),
                             "shape too large");
  }
  return strides;
}

inline std::vector<size_t> align_shape(std::span<const size_t> shape, int out_ndim) {
  std::vector<size_t> out(static_cast<size_t>(out_ndim), 1);
  const int           offset = out_ndim - static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    out[static_cast<size_t>(offset + static_cast<int>(i))] = shape[i];
  }
  return out;
}

inline std::vector<std::int64_t> align_strides(std::span<const size_t>       shape,
                                               std::span<const std::int64_t> strides,
                                               std::span<const size_t>       aligned_shape) {
  const int                 out_ndim = static_cast<int>(aligned_shape.size());
  const int                 offset   = out_ndim - static_cast<int>(shape.size());
  std::vector<std::int64_t> out(static_cast<size_t>(out_ndim), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    const int out_i = offset + static_cast<int>(i);
    if (aligned_shape[static_cast<size_t>(out_i)] == 1) {
      out[static_cast<size_t>(out_i)] = 0;
    } else {
      out[static_cast<size_t>(out_i)] = strides[i];
    }
  }
  return out;
}

struct MulPlan {
  std::vector<size_t>       out_shape;
  std::vector<std::int64_t> out_strides;
  std::vector<std::int64_t> a_strides;
  std::vector<std::int64_t> b_strides;
  std::int64_t              total_out = 0;
  int                       out_ndim  = 0;
};

MulPlan build_plan(const holoflow::core::TDesc &adesc, const holoflow::core::TDesc &bdesc) {
  const int a_ndim   = static_cast<int>(adesc.shape.size());
  const int b_ndim   = static_cast<int>(bdesc.shape.size());
  const int out_ndim = std::max(a_ndim, b_ndim);
  check(out_ndim <= kMaxNDim, "input ndim too large");

  const auto a_aligned = align_shape(adesc.shape, out_ndim);
  const auto b_aligned = align_shape(bdesc.shape, out_ndim);

  MulPlan plan;
  plan.out_ndim = out_ndim;
  plan.out_shape.resize(static_cast<size_t>(out_ndim));

  for (int i = 0; i < out_ndim; ++i) {
    const auto a_dim = a_aligned[static_cast<size_t>(i)];
    const auto b_dim = b_aligned[static_cast<size_t>(i)];
    check(a_dim == b_dim || a_dim == 1 || b_dim == 1, "inputs are not broadcast compatible");
    plan.out_shape[static_cast<size_t>(i)] = std::max(a_dim, b_dim);
  }

  plan.out_strides     = make_contig_strides(plan.out_shape);
  const auto a_strides = make_contig_strides(adesc.shape);
  const auto b_strides = make_contig_strides(bdesc.shape);

  plan.a_strides = align_strides(adesc.shape, a_strides, a_aligned);
  plan.b_strides = align_strides(bdesc.shape, b_strides, b_aligned);

  std::int64_t total = 1;
  for (const auto dim : plan.out_shape) {
    total = checked_mul(total, to_i64(dim, "output shape too large"), "output shape too large");
  }
  plan.total_out = total;

  return plan;
}

template <typename T>
__global__ void
mul_broadcast_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                     int ndim, const std::int64_t *__restrict__ out_strides,
                     const std::int64_t *__restrict__ a_strides,
                     const std::int64_t *__restrict__ b_strides, std::int64_t total_out) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx >= total_out) {
    return;
  }

  std::int64_t rem   = idx;
  std::int64_t a_off = 0;
  std::int64_t b_off = 0;

  for (int i = 0; i < ndim; ++i) {
    const std::int64_t stride = out_strides[i];
    const std::int64_t coord  = (stride > 0) ? (rem / stride) : 0;
    rem -= coord * stride;
    a_off += coord * a_strides[i];
    b_off += coord * b_strides[i];
  }

  out[idx] = static_cast<T>(a[a_off] * b[b_off]);
}

__global__ void mul_broadcast_kernel_cf32(const cuFloatComplex *__restrict__ a,
                                          const cuFloatComplex *__restrict__ b,
                                          cuFloatComplex *__restrict__ out, int ndim,
                                          const std::int64_t *__restrict__ out_strides,
                                          const std::int64_t *__restrict__ a_strides,
                                          const std::int64_t *__restrict__ b_strides,
                                          std::int64_t total_out) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx >= total_out) {
    return;
  }

  std::int64_t rem   = idx;
  std::int64_t a_off = 0;
  std::int64_t b_off = 0;

  for (int i = 0; i < ndim; ++i) {
    const std::int64_t stride = out_strides[i];
    const std::int64_t coord  = (stride > 0) ? (rem / stride) : 0;
    rem -= coord * stride;
    a_off += coord * a_strides[i];
    b_off += coord * b_strides[i];
  }

  out[idx] = cuCmulf(a[a_off], b[b_off]);
}

} // namespace

Mul::Mul(const MulSettings &settings, cudaStream_t stream, size_t out_ndim, std::int64_t total_out,
         HostPtr<std::int64_t> h_out_strides, DevPtr<std::int64_t> d_out_strides,
         HostPtr<std::int64_t> h_a_strides, DevPtr<std::int64_t> d_a_strides,
         HostPtr<std::int64_t> h_b_strides, DevPtr<std::int64_t> d_b_strides)
    : settings_(settings), stream_(stream), out_ndim_(out_ndim), total_out_(total_out),
      h_out_strides_(std::move(h_out_strides)), d_out_strides_(std::move(d_out_strides)),
      h_a_strides_(std::move(h_a_strides)), d_a_strides_(std::move(d_a_strides)),
      h_b_strides_(std::move(h_b_strides)), d_b_strides_(std::move(d_b_strides)) {}

holoflow::core::OpResult Mul::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.inputs.size() != 2 || ctx.outputs.size() != 1) {
    logger()->error("[Mul::execute] expected 2 inputs and 1 output");
    std::abort();
  }

  // auto [a_data, a_desc] = ctx.inputs[0];
  // auto [b_data, b_desc] = ctx.inputs[1];
  // auto [o_data, o_desc] = ctx.outputs[0];
  // (void)b_desc;
  // (void)o_desc;
  auto *a_data = ctx.inputs[0].data();
  auto *b_data = ctx.inputs[1].data();
  auto *o_data = ctx.outputs[0].data();
  const auto &a_desc = ctx.inputs[0].desc;

  const auto total_out = total_out_;
  if (total_out == 0) {
    return holoflow::core::OpResult::Ok;
  }

  constexpr int block = 256;
  const int     grid  = static_cast<int>((total_out + block - 1) / block);

  switch (a_desc.dtype) {
  case holoflow::core::DType::U8: {
    auto *a   = reinterpret_cast<const std::uint8_t *>(a_data);
    auto *b   = reinterpret_cast<const std::uint8_t *>(b_data);
    auto *out = reinterpret_cast<std::uint8_t *>(o_data);
    mul_broadcast_kernel<<<grid, block, 0, stream_>>>(a, b, out, static_cast<int>(out_ndim_),
                                                      d_out_strides_.get(), d_a_strides_.get(),
                                                      d_b_strides_.get(), total_out);
    break;
  }
  case holoflow::core::DType::U16: {
    auto *a   = reinterpret_cast<const std::uint16_t *>(a_data);
    auto *b   = reinterpret_cast<const std::uint16_t *>(b_data);
    auto *out = reinterpret_cast<std::uint16_t *>(o_data);
    mul_broadcast_kernel<<<grid, block, 0, stream_>>>(a, b, out, static_cast<int>(out_ndim_),
                                                      d_out_strides_.get(), d_a_strides_.get(),
                                                      d_b_strides_.get(), total_out);
    break;
  }
  case holoflow::core::DType::F32: {
    auto *a   = reinterpret_cast<const float *>(a_data);
    auto *b   = reinterpret_cast<const float *>(b_data);
    auto *out = reinterpret_cast<float *>(o_data);
    mul_broadcast_kernel<<<grid, block, 0, stream_>>>(a, b, out, static_cast<int>(out_ndim_),
                                                      d_out_strides_.get(), d_a_strides_.get(),
                                                      d_b_strides_.get(), total_out);
    break;
  }
  case holoflow::core::DType::CF32: {
    auto *a   = reinterpret_cast<const cuFloatComplex *>(a_data);
    auto *b   = reinterpret_cast<const cuFloatComplex *>(b_data);
    auto *out = reinterpret_cast<cuFloatComplex *>(o_data);
    mul_broadcast_kernel_cf32<<<grid, block, 0, stream_>>>(a, b, out, static_cast<int>(out_ndim_),
                                                           d_out_strides_.get(), d_a_strides_.get(),
                                                           d_b_strides_.get(), total_out);
    break;
  }
  default:
    logger()->error("[Mul::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MulFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  (void)jsettings;

  check(input_descs.size() == 2, "expected exactly 2 inputs");
  const auto &adesc = input_descs[0];
  const auto &bdesc = input_descs[1];

  check(adesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(adesc.mem_loc == bdesc.mem_loc, "inputs must share the same mem_loc");
  check(adesc.dtype == bdesc.dtype, "inputs must share the same dtype");
  check(adesc.dtype == holoflow::core::DType::U8 || adesc.dtype == holoflow::core::DType::U16 ||
            adesc.dtype == holoflow::core::DType::F32 || adesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");

  const auto plan = build_plan(adesc, bdesc);
  check(plan.total_out > 0, "output tensor has zero elements");

  holoflow::core::TDesc odesc = adesc;
  odesc.shape                 = plan.out_shape;

  return holoflow::core::InferResult{
      .input_descs   = {adesc, bdesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
MulFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  (void)infer_res;

  const auto  settings = jsettings.get<MulSettings>();
  const auto &adesc    = input_descs[0];
  const auto &bdesc    = input_descs[1];
  const auto  plan     = build_plan(adesc, bdesc);

  const size_t ndim = static_cast<size_t>(plan.out_ndim);

  HostPtr<std::int64_t> h_out_strides;
  DevPtr<std::int64_t>  d_out_strides;
  HostPtr<std::int64_t> h_a_strides;
  DevPtr<std::int64_t>  d_a_strides;
  HostPtr<std::int64_t> h_b_strides;
  DevPtr<std::int64_t>  d_b_strides;

  if (ndim > 0) {
    h_out_strides = curaii::make_unique_host_ptr<std::int64_t>(ndim);
    h_a_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
    h_b_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      h_out_strides.get()[i] = plan.out_strides[i];
      h_a_strides.get()[i]   = plan.a_strides[i];
      h_b_strides.get()[i]   = plan.b_strides[i];
    }

    d_out_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
    d_a_strides   = curaii::make_unique_device_ptr<std::int64_t>(ndim);
    d_b_strides   = curaii::make_unique_device_ptr<std::int64_t>(ndim);

    CUDA_CHECK(cudaMemcpyAsync(d_out_strides.get(), h_out_strides.get(),
                               sizeof(std::int64_t) * ndim, cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a_strides.get(), h_a_strides.get(), sizeof(std::int64_t) * ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b_strides.get(), h_b_strides.get(), sizeof(std::int64_t) * ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new Mul(settings, ctx.stream, ndim, plan.total_out, std::move(h_out_strides),
              std::move(d_out_strides), std::move(h_a_strides), std::move(d_a_strides),
              std::move(h_b_strides), std::move(d_b_strides)));
}

} // namespace holonp
