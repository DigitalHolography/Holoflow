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

#include "holonp/mean_abs.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

namespace {

constexpr int kMaxNDim = 16;

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("MeanAbs: " + msg);
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

inline size_t num_elements(std::span<const size_t> shape) {
  constexpr size_t max = std::numeric_limits<size_t>::max();
  size_t           n   = 1;
  for (auto d : shape) {
    if (d == 0) {
      return 0;
    }
    if (n > max / d) {
      throw std::overflow_error("MeanAbs: num_elements overflow");
    }
    n *= d;
  }
  return n;
}

inline std::vector<std::int64_t> make_contig_strides(std::span<const size_t> shape) {
  const int                 ndim = static_cast<int>(shape.size());
  std::vector<std::int64_t> strides(ndim, 1);
  std::int64_t              acc = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= static_cast<std::int64_t>(shape[static_cast<size_t>(i)]);
  }
  return strides;
}

inline std::vector<int> normalize_axes(std::span<const int> axes, int ndim) {
  std::vector<int> out;
  if (axes.empty()) {
    if (ndim == 0) {
      return out;
    }
    out.resize(ndim);
    std::iota(out.begin(), out.end(), 0);
    return out;
  }

  out.reserve(axes.size());
  for (int a : axes) {
    if (a < 0) {
      a += ndim;
    }
    check(a >= 0 && a < ndim, "axis out of range");
    out.push_back(a);
  }

  std::sort(out.begin(), out.end());
  const auto dup = std::adjacent_find(out.begin(), out.end());
  check(dup == out.end(), "axes must be unique");
  return out;
}

struct MeanAbsPlan {
  std::vector<int>          reduce_axes;
  std::vector<size_t>       out_shape;
  std::vector<int>          out_to_in;
  std::vector<std::int64_t> in_strides;
  std::vector<std::int64_t> out_strides;
  std::vector<std::int64_t> red_strides;
  std::int64_t              total_out = 0;
  std::int64_t              total_red = 0;
  int                       out_ndim  = 0;
  int                       red_ndim  = 0;
};

MeanAbsPlan build_plan(const MeanAbsSettings &settings, std::span<const size_t> shape) {
  const int ndim = static_cast<int>(shape.size());
  check(ndim <= kMaxNDim, "input ndim too large");

  const auto        reduce_axes = normalize_axes(settings.axis, ndim);
  std::vector<bool> reduce_mask(static_cast<size_t>(ndim), false);
  for (int a : reduce_axes) {
    reduce_mask[static_cast<size_t>(a)] = true;
  }

  MeanAbsPlan plan;
  plan.reduce_axes = reduce_axes;
  plan.red_ndim    = static_cast<int>(reduce_axes.size());

  if (settings.keepdims) {
    plan.out_shape.assign(shape.begin(), shape.end());
    plan.out_to_in.resize(static_cast<size_t>(ndim));
    for (int i = 0; i < ndim; ++i) {
      plan.out_to_in[static_cast<size_t>(i)] = i;
      if (reduce_mask[static_cast<size_t>(i)]) {
        plan.out_shape[static_cast<size_t>(i)] = 1;
      }
    }
  } else {
    plan.out_shape.clear();
    plan.out_to_in.clear();
    for (int i = 0; i < ndim; ++i) {
      if (!reduce_mask[static_cast<size_t>(i)]) {
        plan.out_shape.push_back(shape[static_cast<size_t>(i)]);
        plan.out_to_in.push_back(i);
      }
    }
  }

  const auto total_in = num_elements(shape);
  check(total_in > 0, "input tensor has zero elements");

  size_t total_red = 1;
  if (!reduce_axes.empty()) {
    for (int a : reduce_axes) {
      const auto dim = shape[static_cast<size_t>(a)];
      if (dim == 0) {
        total_red = 0;
        break;
      }
      if (total_red > std::numeric_limits<size_t>::max() / dim) {
        throw std::overflow_error("MeanAbs: reduction size overflow");
      }
      total_red *= dim;
    }
  }

  if (reduce_axes.empty() && ndim == 0) {
    total_red = 1;
  }

  check(total_red > 0, "reduction has zero elements");

  const auto total_out = num_elements(plan.out_shape);
  check(total_out > 0, "output tensor has zero elements");

  check(total_out <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
        "output size exceeds limits");
  check(total_red <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
        "reduction size exceeds limits");

  plan.total_out = static_cast<std::int64_t>(total_out);
  plan.total_red = static_cast<std::int64_t>(total_red);
  plan.out_ndim  = static_cast<int>(plan.out_shape.size());

  plan.in_strides  = make_contig_strides(shape);
  plan.out_strides = make_contig_strides(plan.out_shape);

  plan.red_strides.resize(plan.reduce_axes.size(), 1);
  std::int64_t acc = 1;
  for (int i = static_cast<int>(plan.reduce_axes.size()) - 1; i >= 0; --i) {
    plan.red_strides[static_cast<size_t>(i)] = acc;
    acc *= static_cast<std::int64_t>(shape[static_cast<size_t>(plan.reduce_axes[i])]);
  }

  return plan;
}

template <typename T> __device__ float abs_as_f32(T v) {
  return v < 0 ? static_cast<float>(-v) : static_cast<float>(v);
}
template <> __device__ float abs_as_f32(std::uint8_t v) { return static_cast<float>(v); }
template <> __device__ float abs_as_f32(std::uint16_t v) { return static_cast<float>(v); }
template <> __device__ float abs_as_f32(float v) { return fabsf(v); }
template <> __device__ float abs_as_f32(cuFloatComplex v) { return sqrtf(v.x * v.x + v.y * v.y); }

template <typename InT>
__global__ void
mean_abs_kernel(const InT *__restrict__ in, float *__restrict__ out, int out_ndim,
                const std::int64_t *__restrict__ out_strides, const int *__restrict__ out_to_in,
                const std::int64_t *__restrict__ in_strides, const int *__restrict__ red_axes,
                const std::int64_t *__restrict__ red_strides, int red_ndim, std::int64_t total_out,
                std::int64_t total_red) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_out) {
    return;
  }

  std::int64_t rem     = tid;
  std::int64_t in_base = 0;
  for (int k = 0; k < out_ndim; ++k) {
    const std::int64_t stride = out_strides[k];
    const std::int64_t idx    = (stride > 0) ? (rem / stride) : 0;
    rem -= idx * stride;
    const int in_axis = out_to_in[k];
    in_base += idx * in_strides[in_axis];
  }

  float sum = 0.0F;
  for (std::int64_t r = 0; r < total_red; ++r) {
    std::int64_t rem_r  = r;
    std::int64_t in_off = in_base;
    for (int i = 0; i < red_ndim; ++i) {
      const std::int64_t stride = red_strides[i];
      const std::int64_t idx    = (stride > 0) ? (rem_r / stride) : 0;
      rem_r -= idx * stride;
      in_off += idx * in_strides[red_axes[i]];
    }
    sum += abs_as_f32(in[in_off]);
  }

  out[tid] = sum / static_cast<float>(total_red);
}

class MeanAbs : public holoflow::core::ISyncTask {
public:
  MeanAbs(MeanAbsSettings settings, holoflow::core::TDesc idesc, cudaStream_t stream,
          size_t out_ndim, size_t red_ndim, std::int64_t total_out, std::int64_t total_red,
          HostPtr<std::int64_t> h_in_strides, DevPtr<std::int64_t> d_in_strides,
          HostPtr<std::int64_t> h_out_strides, DevPtr<std::int64_t> d_out_strides,
          HostPtr<int> h_out_to_in, DevPtr<int> d_out_to_in, HostPtr<int> h_red_axes,
          DevPtr<int> d_red_axes, HostPtr<std::int64_t> h_red_strides,
          DevPtr<std::int64_t> d_red_strides)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream),
        out_ndim_(out_ndim), red_ndim_(red_ndim), total_out_(total_out), total_red_(total_red),
        h_in_strides_(std::move(h_in_strides)), d_in_strides_(std::move(d_in_strides)),
        h_out_strides_(std::move(h_out_strides)), d_out_strides_(std::move(d_out_strides)),
        h_out_to_in_(std::move(h_out_to_in)), d_out_to_in_(std::move(d_out_to_in)),
        h_red_axes_(std::move(h_red_axes)), d_red_axes_(std::move(d_red_axes)),
        h_red_strides_(std::move(h_red_strides)), d_red_strides_(std::move(d_red_strides)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  const MeanAbsSettings       &settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  MeanAbsSettings       settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                out_ndim_;
  size_t                red_ndim_;
  std::int64_t          total_out_;
  std::int64_t          total_red_;
  HostPtr<std::int64_t> h_in_strides_;
  DevPtr<std::int64_t>  d_in_strides_;
  HostPtr<std::int64_t> h_out_strides_;
  DevPtr<std::int64_t>  d_out_strides_;
  HostPtr<int>          h_out_to_in_;
  DevPtr<int>           d_out_to_in_;
  HostPtr<int>          h_red_axes_;
  DevPtr<int>           d_red_axes_;
  HostPtr<std::int64_t> h_red_strides_;
  DevPtr<std::int64_t>  d_red_strides_;
};

} // namespace

holoflow::core::OpResult MeanAbs::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  constexpr int block = 256;
  const int     grid  = static_cast<int>((total_out_ + block - 1) / block);

  auto *out = reinterpret_cast<float *>(odata);
  switch (idesc.dtype) {
  case holoflow::core::DType::U8: {
    const auto *in = reinterpret_cast<const std::uint8_t *>(idata);
    mean_abs_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::U16: {
    const auto *in = reinterpret_cast<const std::uint16_t *>(idata);
    mean_abs_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::F32: {
    const auto *in = reinterpret_cast<const float *>(idata);
    mean_abs_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::CF32: {
    const auto *in = reinterpret_cast<const cuFloatComplex *>(idata);
    mean_abs_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  default:
    logger()->error("[MeanAbs::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
MeanAbsFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<MeanAbsSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");

  const auto plan = build_plan(settings, idesc.shape);

  holoflow::core::TDesc odesc(plan.out_shape, holoflow::core::DType::F32, idesc.mem_loc);

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
MeanAbsFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<MeanAbsSettings>();

  const auto &idesc = input_descs[0];
  const auto  plan  = build_plan(settings, idesc.shape);

  const size_t ndim     = idesc.shape.size();
  const size_t out_ndim = static_cast<size_t>(plan.out_ndim);
  const size_t red_ndim = static_cast<size_t>(plan.red_ndim);

  curaii::unique_host_ptr<std::int64_t>   h_in_strides;
  curaii::unique_device_ptr<std::int64_t> d_in_strides;
  if (ndim > 0) {
    h_in_strides = curaii::make_unique_host_ptr<std::int64_t>(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      h_in_strides.get()[i] = plan.in_strides[i];
    }
    d_in_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_in_strides.get(), h_in_strides.get(), sizeof(std::int64_t) * ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  curaii::unique_host_ptr<std::int64_t>   h_out_strides;
  curaii::unique_device_ptr<std::int64_t> d_out_strides;
  if (out_ndim > 0) {
    h_out_strides = curaii::make_unique_host_ptr<std::int64_t>(out_ndim);
    for (size_t i = 0; i < out_ndim; ++i) {
      h_out_strides.get()[i] = plan.out_strides[i];
    }
    d_out_strides = curaii::make_unique_device_ptr<std::int64_t>(out_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_out_strides.get(), h_out_strides.get(),
                               sizeof(std::int64_t) * out_ndim, cudaMemcpyHostToDevice,
                               ctx.stream));
  }

  curaii::unique_host_ptr<int>   h_out_to_in;
  curaii::unique_device_ptr<int> d_out_to_in;
  if (out_ndim > 0) {
    h_out_to_in = curaii::make_unique_host_ptr<int>(out_ndim);
    for (size_t i = 0; i < out_ndim; ++i) {
      h_out_to_in.get()[i] = plan.out_to_in[i];
    }
    d_out_to_in = curaii::make_unique_device_ptr<int>(out_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_out_to_in.get(), h_out_to_in.get(), sizeof(int) * out_ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  curaii::unique_host_ptr<int>   h_red_axes;
  curaii::unique_device_ptr<int> d_red_axes;
  if (red_ndim > 0) {
    h_red_axes = curaii::make_unique_host_ptr<int>(red_ndim);
    for (size_t i = 0; i < red_ndim; ++i) {
      h_red_axes.get()[i] = plan.reduce_axes[i];
    }
    d_red_axes = curaii::make_unique_device_ptr<int>(red_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_red_axes.get(), h_red_axes.get(), sizeof(int) * red_ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  curaii::unique_host_ptr<std::int64_t>   h_red_strides;
  curaii::unique_device_ptr<std::int64_t> d_red_strides;
  if (red_ndim > 0) {
    h_red_strides = curaii::make_unique_host_ptr<std::int64_t>(red_ndim);
    for (size_t i = 0; i < red_ndim; ++i) {
      h_red_strides.get()[i] = plan.red_strides[i];
    }
    d_red_strides = curaii::make_unique_device_ptr<std::int64_t>(red_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_red_strides.get(), h_red_strides.get(),
                               sizeof(std::int64_t) * red_ndim, cudaMemcpyHostToDevice,
                               ctx.stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new MeanAbs(settings, idesc, ctx.stream, out_ndim, red_ndim, plan.total_out, plan.total_red,
                  std::move(h_in_strides), std::move(d_in_strides), std::move(h_out_strides),
                  std::move(d_out_strides), std::move(h_out_to_in), std::move(d_out_to_in),
                  std::move(h_red_axes), std::move(d_red_axes), std::move(h_red_strides),
                  std::move(d_red_strides)));
}

std::unique_ptr<holoflow::core::ISyncTask>
MeanAbsFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {

  (void)infer(input_descs, jsettings);

  auto *old_mean = dynamic_cast<MeanAbs *>(old_task.get());
  if (old_mean != nullptr && input_descs.size() == 1) {

    const auto  new_settings = jsettings.get<MeanAbsSettings>();
    const auto &new_idesc    = input_descs[0];
    const auto &old_idesc    = old_mean->idesc();

    bool can_reuse = (new_settings == old_mean->settings()) && same_desc(new_idesc, old_idesc);

    if (can_reuse) {
      old_mean->update_stream(ctx.stream);
      return old_task;
    }
  }

  // Fallback: Structural change detected or invalid old task.
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
