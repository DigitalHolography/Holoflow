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
#include <cuComplex.h>
#include <type_traits>
#include <utility>

namespace holonp {

void to_json(nlohmann::json &j, const MulSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, MulSettings &) {}

namespace {

// Runtime type promotion logic (NumPy style)
holoflow::core::DType promote_dtype(holoflow::core::DType a, holoflow::core::DType b) {
  if (a == b)
    return a;
  if (a == holoflow::core::DType::CF32 || b == holoflow::core::DType::CF32)
    return holoflow::core::DType::CF32;
  if (a == holoflow::core::DType::F32 || b == holoflow::core::DType::F32)
    return holoflow::core::DType::F32;
  if (a == holoflow::core::DType::U16 || b == holoflow::core::DType::U16)
    return holoflow::core::DType::U16;
  return holoflow::core::DType::U8;
}

// Compile-time type promotion for C++ types
template <typename T1, typename T2> struct Promote {
  using type = decltype(std::declval<T1>() + std::declval<T2>());
};

// Specializations for cuFloatComplex (as standard C++ doesn't natively add it to reals)
template <typename T> struct Promote<cuFloatComplex, T> {
  using type = cuFloatComplex;
};
template <typename T> struct Promote<T, cuFloatComplex> {
  using type = cuFloatComplex;
};
template <> struct Promote<cuFloatComplex, cuFloatComplex> {
  using type = cuFloatComplex;
};

// Helper to convert scalar reals to complex for mixed-type complex operations
template <typename T> __device__ inline cuFloatComplex to_complex(T v) {
  if constexpr (std::is_same_v<T, cuFloatComplex>) {
    return v;
  } else {
    return make_cuFloatComplex(static_cast<float>(v), 0.0f);
  }
}

template <typename TA, typename TB, typename TO>
__global__ void mul_kernel(const TA *__restrict__ a, const TB *__restrict__ b, TO *__restrict__ out,
                           size_t total_out, size_t ndim, const size_t *__restrict__ out_shape,
                           const size_t *__restrict__ a_strides,
                           const size_t *__restrict__ b_strides) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_out)
    return;

  size_t a_off = 0, b_off = 0, rem = idx;

  for (int i = int(ndim) - 1; i >= 0; --i) {
    size_t coord = rem % out_shape[i];
    rem /= out_shape[i];
    a_off += coord * a_strides[i];
    b_off += coord * b_strides[i];
  }

  if constexpr (std::is_same_v<TO, cuFloatComplex>) {
    out[idx] = cuCmulf(to_complex(a[a_off]), to_complex(b[b_off]));
  } else {
    out[idx] = static_cast<TO>(a[a_off]) * static_cast<TO>(b[b_off]);
  }
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Mul: " + msg);
  }
}

std::vector<size_t> get_elem_strides(const holoflow::core::TDesc &d) {
  size_t esize = holoflow::core::size_of(d.dtype);
  if (!d.strides.empty()) {
    std::vector<size_t> s;
    for (auto val : d.strides)
      s.push_back(val / esize);
    return s;
  }
  std::vector<size_t> s(d.shape.size());
  size_t              acc = 1;
  for (int i = int(d.shape.size()) - 1; i >= 0; --i) {
    s[i] = acc;
    acc *= d.shape[i];
  }
  return s;
}

} // namespace

Mul::Mul(cudaStream_t stream, holoflow::core::DType dtype_a, holoflow::core::DType dtype_b,
         size_t total_out, size_t ndim, DevPtr<size_t> d_out_shape, DevPtr<size_t> d_a_strides,
         DevPtr<size_t> d_b_strides)
    : stream_(stream), dtype_a_(dtype_a), dtype_b_(dtype_b), total_out_(total_out), ndim_(ndim),
      d_out_shape_(std::move(d_out_shape)), d_a_strides_(std::move(d_a_strides)),
      d_b_strides_(std::move(d_b_strides)) {}

holoflow::core::OpResult Mul::execute(holoflow::core::SyncCtx &ctx) {
  if (total_out_ == 0)
    return holoflow::core::OpResult::Ok;

  int block = 256;
  int grid  = static_cast<int>((total_out_ + block - 1) / block);

// Double dispatch macros to handle all cross-type combinations
#define DISPATCH_MUL(TA, TB)                                                                       \
  do {                                                                                             \
    using TO = typename Promote<TA, TB>::type;                                                     \
    mul_kernel<TA, TB, TO><<<grid, block, 0, stream_>>>(                                           \
        (const TA *)ctx.inputs[0].data(), (const TB *)ctx.inputs[1].data(),                        \
        (TO *)ctx.outputs[0].data(), total_out_, ndim_, d_out_shape_.get(), d_a_strides_.get(),    \
        d_b_strides_.get());                                                                       \
  } while (0)

#define DISPATCH_B(TA)                                                                             \
  switch (dtype_b_) {                                                                              \
  case holoflow::core::DType::F32:                                                                 \
    DISPATCH_MUL(TA, float);                                                                       \
    break;                                                                                         \
  case holoflow::core::DType::CF32:                                                                \
    DISPATCH_MUL(TA, cuFloatComplex);                                                              \
    break;                                                                                         \
  case holoflow::core::DType::U16:                                                                 \
    DISPATCH_MUL(TA, uint16_t);                                                                    \
    break;                                                                                         \
  case holoflow::core::DType::U8:                                                                  \
    DISPATCH_MUL(TA, uint8_t);                                                                     \
    break;                                                                                         \
  default:                                                                                         \
    std::abort();                                                                                  \
  }

  switch (dtype_a_) {
  case holoflow::core::DType::F32:
    DISPATCH_B(float);
    break;
  case holoflow::core::DType::CF32:
    DISPATCH_B(cuFloatComplex);
    break;
  case holoflow::core::DType::U16:
    DISPATCH_B(uint16_t);
    break;
  case holoflow::core::DType::U8:
    DISPATCH_B(uint8_t);
    break;
  default:
    std::abort();
  }

#undef DISPATCH_B
#undef DISPATCH_MUL

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MulFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                              const nlohmann::json &) const {
  check(inputs.size() == 2, "expected exactly 2 input tensors");
  check(inputs[0].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 0 must be in device memory");
  check(inputs[1].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 1 must be in device memory");

  const auto &a = inputs[0], &b = inputs[1];
  size_t      ndim = std::max(a.shape.size(), b.shape.size());

  std::vector<size_t> out_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    size_t ad    = (i < ndim - a.shape.size()) ? 1 : a.shape[i - (ndim - a.shape.size())];
    size_t bd    = (i < ndim - b.shape.size()) ? 1 : b.shape[i - (ndim - b.shape.size())];
    out_shape[i] = std::max(ad, bd);
  }

  holoflow::core::DType out_dtype = promote_dtype(a.dtype, b.dtype);
  holoflow::core::TDesc o(out_shape, out_dtype, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {a, b},
      .output_descs  = {o},
      .in_place      = {},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
MulFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &j,
                   const holoflow::core::SyncCreateCtx &ctx) const {
  const auto &a = inputs[0], &b = inputs[1];
  auto        res   = infer(inputs, j);
  const auto &odesc = res.output_descs[0];
  size_t      ndim  = odesc.shape.size();
  size_t      total = 1;

  std::vector<size_t> a_strides_h(ndim), b_strides_h(ndim);
  auto                as_raw = get_elem_strides(a);
  auto                bs_raw = get_elem_strides(b);

  for (size_t i = 0; i < ndim; ++i) {
    total *= odesc.shape[i];
    auto map = [&](const auto &shape, const auto &strides) {
      int axis = int(i) - (int(ndim) - int(shape.size()));
      return (axis < 0 || shape[axis] == 1) ? 0 : strides[axis];
    };
    a_strides_h[i] = map(a.shape, as_raw);
    b_strides_h[i] = map(b.shape, bs_raw);
  }

  auto d_shape = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_a_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_b_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto bytes   = ndim * sizeof(size_t);
  auto h2d     = cudaMemcpyHostToDevice;

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), odesc.shape.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_a_str.get(), a_strides_h.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_b_str.get(), b_strides_h.data(), bytes, h2d, ctx.stream));

  return std::make_unique<Mul>(ctx.stream, a.dtype, b.dtype, total, ndim, std::move(d_shape),
                               std::move(d_a_str), std::move(d_b_str));
}

} // namespace holonp