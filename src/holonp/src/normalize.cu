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

#include "holonp/normalize.hh"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cub/cub.cuh>
#include <numeric>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const NormalizeSettings &s) {
  j = nlohmann::json{{"lo", s.lo}, {"hi", s.hi}};
  if (s.axes.empty()) {
    j["axes"] = nullptr;
  } else if (s.axes.size() == 1) {
    j["axes"] = s.axes[0];
  } else {
    j["axes"] = s.axes;
  }
}

void from_json(const nlohmann::json &j, NormalizeSettings &s) {
  s.axes.clear();
  s.lo = j.value("lo", 0.0f);
  s.hi = j.value("hi", 1.0f);

  const auto parse_axes = [&](const std::string &key) {
    if (!j.contains(key) || j.at(key).is_null()) {
      return;
    }
    const auto &ja = j.at(key);
    if (ja.is_number_integer()) {
      s.axes = {ja.get<int>()};
    } else if (ja.is_array()) {
      ja.get_to(s.axes);
    } else {
      throw std::invalid_argument("NormalizeSettings: axes/axis must be int, array, or null");
    }
  };

  parse_axes("axes");
  if (s.axes.empty()) {
    parse_axes("axis");
  }
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Normalize: " + msg);
  }
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

int get_dtype_size(holoflow::core::DType dt) {
  using namespace holoflow::core;
  switch (dt) {
  case DType::U8:
    return 1;
  case DType::U16:
    return 2;
  case DType::F32:
    return 4;
  default:
    return 1;
  }
}

size_t num_elements(std::span<const size_t> shape) {
  size_t n = 1;
  for (auto d : shape) {
    if (d == 0) {
      return 0;
    }
    n *= d;
  }
  return n;
}

std::vector<size_t> compute_compact_strides(std::span<const size_t> shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<size_t> strides(shape.size());
  size_t              acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = acc;
    acc *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

std::vector<size_t> bytes_to_elements(std::span<const size_t> byte_strides, int itemsize) {
  std::vector<size_t> elem_strides;
  elem_strides.reserve(byte_strides.size());
  for (auto s : byte_strides) {
    elem_strides.push_back(static_cast<size_t>(s) / static_cast<size_t>(itemsize));
  }
  return elem_strides;
}

std::vector<int> normalize_axes(std::span<const int> axes, int ndim) {
  std::vector<int> out;
  if (axes.empty()) {
    out.resize(static_cast<size_t>(ndim));
    std::iota(out.begin(), out.end(), 0);
  } else {
    out.reserve(axes.size());
    for (int a : axes) {
      int norm = (a < 0) ? (a + ndim) : a;
      check(norm >= 0 && norm < ndim, "axis out of range");
      out.push_back(norm);
    }
    std::sort(out.begin(), out.end());
    if (std::unique(out.begin(), out.end()) != out.end()) {
      throw std::invalid_argument("Normalize: duplicate axes");
    }
  }
  return out;
}

template <typename T> __device__ __forceinline__ float as_f32(T v) { return static_cast<float>(v); }

template <typename T, int BLOCK_SIZE>
__global__ void reduce_group_minmax_kernel(const T *__restrict__ in, float *__restrict__ mins,
                                           float *__restrict__ maxs, size_t total_groups,
                                           size_t total_red, int group_ndim, int red_ndim,
                                           const size_t *__restrict__ in_strides,
                                           const size_t *__restrict__ group_strides,
                                           const int *__restrict__ group_axes,
                                           const size_t *__restrict__ red_strides,
                                           const int *__restrict__ red_axes) {
  const size_t group_idx = static_cast<size_t>(blockIdx.x);
  if (group_idx >= total_groups) {
    return;
  }

  size_t tmp_group = group_idx;
  size_t in_base   = 0;
  for (int i = 0; i < group_ndim; ++i) {
    const size_t stride = group_strides[i];
    const size_t coord  = (stride > 0) ? (tmp_group / stride) : 0;
    tmp_group -= coord * stride;
    in_base += coord * in_strides[group_axes[i]];
  }

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;
  for (size_t r = static_cast<size_t>(threadIdx.x); r < total_red; r += BLOCK_SIZE) {
    size_t tmp_r  = r;
    size_t in_off = in_base;
    for (int i = 0; i < red_ndim; ++i) {
      const size_t stride = red_strides[i];
      const size_t coord  = (stride > 0) ? (tmp_r / stride) : 0;
      tmp_r -= coord * stride;
      in_off += coord * in_strides[red_axes[i]];
    }
    const float v = as_f32(in[in_off]);
    local_min     = fminf(local_min, v);
    local_max     = fmaxf(local_max, v);
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage min_storage;
  __shared__ typename BlockReduce::TempStorage max_storage;
  const float block_min = BlockReduce(min_storage).Reduce(local_min, cuda::minimum<float>());
  __syncthreads();
  const float block_max = BlockReduce(max_storage).Reduce(local_max, cuda::maximum<float>());

  if (threadIdx.x == 0) {
    mins[group_idx] = block_min;
    maxs[group_idx] = block_max;
  }
}

template <typename T>
__global__ void
apply_normalize_kernel(const T *__restrict__ in, float *__restrict__ out, size_t total_elems,
                       int ndim, int group_ndim, const size_t *__restrict__ shape,
                       const size_t *__restrict__ in_strides,
                       const size_t *__restrict__ out_strides, const int *__restrict__ group_axes,
                       const size_t *__restrict__ group_strides, const float *__restrict__ mins,
                       const float *__restrict__ maxs, float lo, float hi) {
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
  if (tid >= total_elems) {
    return;
  }

  size_t rem    = tid;
  size_t in_off = 0;
  for (int i = 0; i < ndim; ++i) {
    const size_t stride = out_strides[i];
    const size_t coord  = (stride > 0) ? (rem / stride) : 0;
    rem -= coord * stride;
    in_off += coord * in_strides[i];
  }

  size_t group_idx = 0;
  for (int k = 0; k < group_ndim; ++k) {
    const int    axis      = group_axes[k];
    const size_t axis_step = out_strides[axis];
    const size_t axis_dim  = shape[axis];
    const size_t coord     = (axis_dim > 0) ? ((tid / axis_step) % axis_dim) : 0;
    group_idx += coord * group_strides[k];
  }

  const float x     = as_f32(in[in_off]);
  const float mn    = mins[group_idx];
  const float mx    = maxs[group_idx];
  const float denom = (mx - mn) + 1e-7f;
  const float norm  = (x - mn) / denom;
  out[tid]          = norm * (hi - lo) + lo;
}

struct NormalizePlan {
  std::vector<size_t> shape;
  std::vector<size_t> in_strides;
  std::vector<size_t> out_strides;
  std::vector<int>    group_axes;
  std::vector<size_t> group_strides;
  std::vector<int>    red_axes;
  std::vector<size_t> red_strides;
  size_t              total_elems  = 0;
  size_t              total_groups = 0;
  size_t              total_red    = 0;
};

NormalizePlan build_plan(const holoflow::core::TDesc &idesc, const NormalizeSettings &settings) {
  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");

  NormalizePlan plan;
  plan.shape.assign(idesc.shape.begin(), idesc.shape.end());

  const auto red_axes = normalize_axes(settings.axes, ndim);
  plan.red_axes       = red_axes;

  std::vector<bool> reduced(static_cast<size_t>(ndim), false);
  for (int a : red_axes) {
    reduced[static_cast<size_t>(a)] = true;
  }

  for (int i = 0; i < ndim; ++i) {
    if (!reduced[static_cast<size_t>(i)]) {
      plan.group_axes.push_back(i);
    }
  }

  const int itemsize = get_dtype_size(idesc.dtype);
  if (idesc.strides.empty()) {
    plan.in_strides = compute_compact_strides(idesc.shape);
  } else {
    plan.in_strides = bytes_to_elements(idesc.strides, itemsize);
  }

  plan.out_strides = compute_compact_strides(idesc.shape);

  std::vector<size_t> group_shape;
  group_shape.reserve(plan.group_axes.size());
  for (int a : plan.group_axes) {
    group_shape.push_back(idesc.shape[static_cast<size_t>(a)]);
  }
  plan.group_strides = compute_compact_strides(group_shape);

  std::vector<size_t> red_shape;
  red_shape.reserve(plan.red_axes.size());
  for (int a : plan.red_axes) {
    red_shape.push_back(idesc.shape[static_cast<size_t>(a)]);
  }
  plan.red_strides = compute_compact_strides(red_shape);

  plan.total_elems  = num_elements(idesc.shape);
  plan.total_groups = group_shape.empty() ? 1 : num_elements(group_shape);
  plan.total_red    = red_shape.empty() ? 1 : num_elements(red_shape);

  check(plan.total_elems > 0, "input tensor has zero elements");
  check(plan.total_groups > 0, "group size is zero");
  check(plan.total_red > 0, "reduction size is zero");
  return plan;
}

class Normalize : public holoflow::core::ISyncTask {
public:
  Normalize(NormalizeSettings settings, holoflow::core::TDesc idesc, cudaStream_t stream,
            size_t ndim, size_t group_ndim, size_t red_ndim, size_t total_elems,
            size_t total_groups, size_t total_red, DevPtr<size_t> shape,
            DevPtr<size_t> in_strides, DevPtr<size_t> out_strides, DevPtr<int> group_axes,
            DevPtr<size_t> group_strides, DevPtr<int> red_axes, DevPtr<size_t> red_strides,
            DevPtr<float> group_mins, DevPtr<float> group_maxs)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream), ndim_(ndim),
        group_ndim_(group_ndim), red_ndim_(red_ndim), total_elems_(total_elems),
        total_groups_(total_groups), total_red_(total_red), d_shape_(std::move(shape)),
        d_in_strides_(std::move(in_strides)), d_out_strides_(std::move(out_strides)),
        d_group_axes_(std::move(group_axes)), d_group_strides_(std::move(group_strides)),
        d_red_axes_(std::move(red_axes)), d_red_strides_(std::move(red_strides)),
        d_group_mins_(std::move(group_mins)), d_group_maxs_(std::move(group_maxs)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const NormalizeSettings     &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  NormalizeSettings     settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                ndim_;
  size_t                group_ndim_;
  size_t                red_ndim_;
  size_t                total_elems_;
  size_t                total_groups_;
  size_t                total_red_;
  DevPtr<size_t>        d_shape_;
  DevPtr<size_t>        d_in_strides_;
  DevPtr<size_t>        d_out_strides_;
  DevPtr<int>           d_group_axes_;
  DevPtr<size_t>        d_group_strides_;
  DevPtr<int>           d_red_axes_;
  DevPtr<size_t>        d_red_strides_;
  DevPtr<float>         d_group_mins_;
  DevPtr<float>         d_group_maxs_;
};

} // namespace

holoflow::core::OpResult Normalize::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  constexpr int reduce_block = 256;
  const int     reduce_grid  = static_cast<int>(total_groups_);
  constexpr int apply_block  = 256;
  const int     apply_grid   = static_cast<int>((total_elems_ + apply_block - 1) / apply_block);

#define LAUNCH_KERNELS(InType)                                                                     \
  reduce_group_minmax_kernel<InType, reduce_block><<<reduce_grid, reduce_block, 0, stream_>>>(     \
      reinterpret_cast<const InType *>(idata), d_group_mins_.get(), d_group_maxs_.get(),           \
      total_groups_, total_red_, static_cast<int>(group_ndim_), static_cast<int>(red_ndim_),       \
      d_in_strides_.get(), d_group_strides_.get(), d_group_axes_.get(), d_red_strides_.get(),      \
      d_red_axes_.get());                                                                          \
  apply_normalize_kernel<InType><<<apply_grid, apply_block, 0, stream_>>>(                         \
      reinterpret_cast<const InType *>(idata), reinterpret_cast<float *>(odata), total_elems_,     \
      static_cast<int>(ndim_), static_cast<int>(group_ndim_), d_shape_.get(), d_in_strides_.get(), \
      d_out_strides_.get(), d_group_axes_.get(), d_group_strides_.get(), d_group_mins_.get(),      \
      d_group_maxs_.get(), settings_.lo, settings_.hi)

  switch (idesc.dtype) {
  case holoflow::core::DType::U8:
    LAUNCH_KERNELS(std::uint8_t);
    break;
  case holoflow::core::DType::U16:
    LAUNCH_KERNELS(std::uint16_t);
    break;
  case holoflow::core::DType::F32:
    LAUNCH_KERNELS(float);
    break;
  default:
    logger()->error("[Normalize::execute] unsupported dtype");
    std::abort();
  }

#undef LAUNCH_KERNELS

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
NormalizeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc    = input_descs[0];
  const auto  settings = jsettings.get<NormalizeSettings>();

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be on Device");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32,
        "supported dtypes are U8/U16/F32");

  (void)build_plan(idesc, settings);

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::F32, idesc.mem_loc);
  return {.input_descs   = {idesc},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
NormalizeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);

  const auto  settings = jsettings.get<NormalizeSettings>();
  const auto &idesc    = input_descs[0];
  const auto  plan     = build_plan(idesc, settings);

  auto upload_size_t = [&](const std::vector<size_t> &host_vec) -> DevPtr<size_t> {
    if (host_vec.empty()) {
      return nullptr;
    }
    auto d_ptr = curaii::make_unique_device_ptr<size_t>(host_vec.size());
    CUDA_CHECK(cudaMemcpyAsync(d_ptr.get(), host_vec.data(), host_vec.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, ctx.stream));
    return d_ptr;
  };

  auto upload_int = [&](const std::vector<int> &host_vec) -> DevPtr<int> {
    if (host_vec.empty()) {
      return nullptr;
    }
    auto d_ptr = curaii::make_unique_device_ptr<int>(host_vec.size());
    CUDA_CHECK(cudaMemcpyAsync(d_ptr.get(), host_vec.data(), host_vec.size() * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));
    return d_ptr;
  };

  auto d_shape         = upload_size_t(plan.shape);
  auto d_in_strides    = upload_size_t(plan.in_strides);
  auto d_out_strides   = upload_size_t(plan.out_strides);
  auto d_group_axes    = upload_int(plan.group_axes);
  auto d_group_strides = upload_size_t(plan.group_strides);
  auto d_red_axes      = upload_int(plan.red_axes);
  auto d_red_strides   = upload_size_t(plan.red_strides);
  auto d_group_mins    = curaii::make_unique_device_ptr<float>(plan.total_groups);
  auto d_group_maxs    = curaii::make_unique_device_ptr<float>(plan.total_groups);

  // Note: Stream sync removed here to prevent blocking the CPU thread!

  return std::make_unique<Normalize>(
      settings, idesc, ctx.stream, idesc.shape.size(), plan.group_axes.size(), plan.red_axes.size(),
      plan.total_elems, plan.total_groups, plan.total_red, std::move(d_shape),
      std::move(d_in_strides), std::move(d_out_strides), std::move(d_group_axes),
      std::move(d_group_strides), std::move(d_red_axes), std::move(d_red_strides),
      std::move(d_group_mins), std::move(d_group_maxs));
}

std::unique_ptr<holoflow::core::ISyncTask>
NormalizeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                         std::span<const holoflow::core::TDesc>     input_descs,
                         const nlohmann::json                      &jsettings,
                         const holoflow::core::SyncCreateCtx       &ctx) const {

  auto *old_norm = dynamic_cast<Normalize *>(old_task.get());
  if (old_norm != nullptr && input_descs.size() == 1) {

    const auto  new_settings = jsettings.get<NormalizeSettings>();
    const auto &new_idesc    = input_descs[0];
    const auto &old_idesc    = old_norm->idesc();

    bool can_reuse = (new_settings == old_norm->settings()) && same_desc(new_idesc, old_idesc);

    if (can_reuse) {
      old_norm->update_stream(ctx.stream);
      return old_task;
    }
  }

  // Fallback: Structural change detected or invalid old task.
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
