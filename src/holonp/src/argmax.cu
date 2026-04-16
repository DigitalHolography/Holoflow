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

#include "holonp/argmax.hh"

#include <algorithm>
#include <cuComplex.h>
#include <limits>
#include <numeric>

namespace holonp {

// -----------------------------------------------------------------------------
// JSON Serialization
// -----------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ArgmaxSettings &s) {
  j["keepdims"] = s.keepdims;
  if (s.axis.empty())
    j["axis"] = nullptr;
  else if (s.axis.size() == 1)
    j["axis"] = s.axis[0];
  else
    j["axis"] = s.axis;
}

void from_json(const nlohmann::json &j, ArgmaxSettings &s) {
  s.axis.clear();
  s.keepdims = j.value("keepdims", false);

  const auto parse_axes = [&](const std::string &key) {
    if (!j.contains(key) || j[key].is_null())
      return;
    if (j[key].is_number_integer())
      s.axis = {j[key].get<int>()};
    else if (j[key].is_array())
      j[key].get_to(s.axis);
    else
      throw std::invalid_argument("ArgmaxSettings: axis must be int, array, or null");
  };

  parse_axes("axis");
  if (s.axis.empty())
    parse_axes("axes");
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

// -------------------------------------------------------------------------------------------------
// Helpers & Utilities
// -------------------------------------------------------------------------------------------------

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Argmax: " + msg);
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
  case DType::CF32:
    return 8;
  default:
    return 1;
  }
}

std::vector<size_t> compute_compact_strides(std::span<const size_t> shape) {
  if (shape.empty())
    return {};
  std::vector<size_t> strides(shape.size());
  size_t              acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= shape[i];
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
    out.resize(ndim);
    std::iota(out.begin(), out.end(), 0);
  } else {
    out.reserve(axes.size());
    for (int a : axes) {
      int norm = (a < 0) ? a + ndim : a;
      check(norm >= 0 && norm < ndim, "axis out of range");
      out.push_back(norm);
    }
    std::sort(out.begin(), out.end());
    if (std::unique(out.begin(), out.end()) != out.end()) {
      throw std::invalid_argument("Argmax: duplicate axes");
    }
  }
  return out;
}

// -------------------------------------------------------------------------------------------------
// Device Kernels
// -------------------------------------------------------------------------------------------------

template <typename T> struct ArgmaxTraits {
  __device__ static bool is_greater(T a, T b) { return a > b; }
};

template <> struct ArgmaxTraits<cuFloatComplex> {
  __device__ static bool is_greater(cuFloatComplex a, cuFloatComplex b) {
    if (a.x != b.x)
      return a.x > b.x;
    return a.y > b.y;
  }
};

template <typename T, typename IdxT>
__global__ void
argmax_kernel(const T *__restrict__ in, IdxT *__restrict__ out, size_t total_out, size_t total_red,
              int out_ndim, int red_ndim, const size_t *__restrict__ out_strides,
              const int *__restrict__ out_to_in_map, const size_t *__restrict__ in_strides,
              const size_t *__restrict__ red_strides, const int *__restrict__ red_axes_map) {

  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_out)
    return;

  // 1. Calculate input offset from output coordinates
  size_t in_base_offset = 0;
  size_t temp_tid       = tid;

  for (int i = 0; i < out_ndim; ++i) {
    size_t stride = out_strides[i];
    size_t coord  = temp_tid / stride;
    temp_tid -= coord * stride;

    in_base_offset += coord * in_strides[out_to_in_map[i]];
  }

  // 2. Initialize with the first element of the reduction (r=0)
  T      best_val = in[in_base_offset];
  size_t best_idx = 0;

  // 3. Iterate remaining reduction elements
  for (size_t r = 1; r < total_red; ++r) {
    size_t in_offset = in_base_offset;
    size_t temp_r    = r;

    for (int i = 0; i < red_ndim; ++i) {
      size_t stride = red_strides[i];
      size_t coord  = temp_r / stride;
      temp_r -= coord * stride;

      in_offset += coord * in_strides[red_axes_map[i]];
    }

    const T val = in[in_offset];
    if (ArgmaxTraits<T>::is_greater(val, best_val)) {
      best_val = val;
      best_idx = r;
    }
  }

  out[tid] = static_cast<IdxT>(best_idx);
}

// -------------------------------------------------------------------------------------------------
// Task implementation
// -------------------------------------------------------------------------------------------------

class Argmax : public holoflow::core::ISyncTask {
public:
  Argmax(ArgmaxSettings settings, holoflow::core::TDesc idesc, cudaStream_t stream,
         size_t total_out, size_t total_red, int out_ndim, int red_ndim, DevPtr<size_t> in_strides,
         DevPtr<size_t> out_strides, DevPtr<int> out_to_in_map, DevPtr<size_t> red_strides,
         DevPtr<int> red_axes_map)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream),
        total_out_(total_out), total_red_(total_red), out_ndim_(out_ndim), red_ndim_(red_ndim),
        d_in_strides_(std::move(in_strides)), d_out_strides_(std::move(out_strides)),
        d_out_to_in_map_(std::move(out_to_in_map)), d_red_strides_(std::move(red_strides)),
        d_red_axes_map_(std::move(red_axes_map)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const ArgmaxSettings        &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ArgmaxSettings        settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                total_out_;
  size_t                total_red_;
  int                   out_ndim_;
  int                   red_ndim_;
  DevPtr<size_t>        d_in_strides_;
  DevPtr<size_t>        d_out_strides_;
  DevPtr<int>           d_out_to_in_map_;
  DevPtr<size_t>        d_red_strides_;
  DevPtr<int>           d_red_axes_map_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Argmax Task Implementation
// -------------------------------------------------------------------------------------------------

holoflow::core::OpResult Argmax::execute(holoflow::core::SyncCtx &ctx) {
  const auto &idesc = ctx.inputs[0].desc;
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();

  if (total_out_ == 0)
    return holoflow::core::OpResult::Ok;

  constexpr int block_size = 256;
  const int     grid_size  = static_cast<int>((total_out_ + block_size - 1) / block_size);

#define LAUNCH_ARGMAX_KERNEL(Type)                                                                 \
  argmax_kernel<Type, std::uint16_t><<<grid_size, block_size, 0, stream_>>>(                       \
      reinterpret_cast<const Type *>(idata), reinterpret_cast<std::uint16_t *>(odata), total_out_, \
      total_red_, out_ndim_, red_ndim_, d_out_strides_.get(), d_out_to_in_map_.get(),              \
      d_in_strides_.get(), d_red_strides_.get(), d_red_axes_map_.get())

  switch (idesc.dtype) {
  case holoflow::core::DType::U8:
    LAUNCH_ARGMAX_KERNEL(std::uint8_t);
    break;
  case holoflow::core::DType::U16:
    LAUNCH_ARGMAX_KERNEL(std::uint16_t);
    break;
  case holoflow::core::DType::F32:
    LAUNCH_ARGMAX_KERNEL(float);
    break;
  case holoflow::core::DType::CF32:
    LAUNCH_ARGMAX_KERNEL(cuFloatComplex);
    break;
  default:
    logger()->error("[Argmax::execute] unsupported dtype");
    std::abort();
  }

#undef LAUNCH_ARGMAX_KERNEL

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// Factory Implementation
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult ArgmaxFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc    = input_descs[0];
  const auto  settings = jsettings.get<ArgmaxSettings>();

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be on Device");
  check(idesc.rank() > 1 || settings.keepdims, "1D reduction must keepdims to avoid scalar output");

  const int  ndim = static_cast<int>(idesc.shape.size());
  const auto axes = normalize_axes(settings.axis, ndim);

  std::vector<bool> is_reduced(ndim, false);
  for (int a : axes)
    is_reduced[a] = true;

  std::vector<size_t> out_shape;
  if (settings.keepdims) {
    out_shape = std::vector<size_t>(idesc.shape.begin(), idesc.shape.end());
    for (int a : axes)
      out_shape[a] = 1;
  } else {
    out_shape.reserve(ndim - axes.size());
    for (int i = 0; i < ndim; ++i) {
      if (!is_reduced[i])
        out_shape.push_back(idesc.shape[i]);
    }
  }

  // Safety check for U16 output
  size_t total_red = 1;
  for (int a : axes) {
    total_red *= idesc.shape[static_cast<size_t>(a)];
  }
  check(total_red > 0, "reduction has zero elements");
  check(total_red <= static_cast<size_t>(std::numeric_limits<std::uint16_t>::max()),
        "reduction size exceeds U16 index range");

  // Use constructor to default to compact strides
  // Fixed Output DType: U16
  holoflow::core::TDesc odesc(out_shape, holoflow::core::DType::U16, idesc.mem_loc);

  return {.input_descs   = {idesc},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
ArgmaxFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);

  const auto  settings = jsettings.get<ArgmaxSettings>();
  const auto &idesc    = input_descs[0];
  const int   ndim     = static_cast<int>(idesc.shape.size());
  const int   itemsize = get_dtype_size(idesc.dtype);

  const auto        reduce_axes = normalize_axes(settings.axis, ndim);
  std::vector<bool> is_reduced(ndim, false);
  for (int a : reduce_axes)
    is_reduced[a] = true;

  // 1. Output Shape & Map
  std::vector<size_t> out_shape;
  std::vector<int>    out_to_in_map;

  if (settings.keepdims) {
    out_shape = std::vector<size_t>(idesc.shape.begin(), idesc.shape.end());
    out_to_in_map.resize(ndim);
    std::iota(out_to_in_map.begin(), out_to_in_map.end(), 0);
    for (int a : reduce_axes)
      out_shape[a] = 1;
  } else {
    for (int i = 0; i < ndim; ++i) {
      if (!is_reduced[i]) {
        out_shape.push_back(idesc.shape[i]);
        out_to_in_map.push_back(i);
      }
    }
  }

  // 2. Strides Setup

  // Input: Convert provided byte strides to element strides (size_t)
  std::vector<size_t> in_strides_elem;
  if (idesc.strides.empty()) {
    in_strides_elem = compute_compact_strides(idesc.shape);
  } else {
    in_strides_elem = bytes_to_elements(idesc.strides, itemsize);
  }

  // Output: We know output is contiguous (enforced in infer)
  auto out_strides_elem = compute_compact_strides(out_shape);

  // Reduction: Fake stride for iterating linear 'r'
  std::vector<size_t> red_dims_shape;
  for (int a : reduce_axes)
    red_dims_shape.push_back(idesc.shape[a]);
  auto red_strides_elem = compute_compact_strides(red_dims_shape);

  size_t total_out = 1;
  for (auto d : out_shape)
    total_out *= d;
  size_t total_red = 1;
  for (auto d : red_dims_shape)
    total_red *= d;

  // 3. Upload
  auto upload = [&]<typename T>(const std::vector<T> &host_vec) -> DevPtr<T> {
    if (host_vec.empty())
      return nullptr;
    auto d_ptr = curaii::make_unique_device_ptr<T>(host_vec.size());
    CUDA_CHECK(cudaMemcpyAsync(d_ptr.get(), host_vec.data(), host_vec.size() * sizeof(T),
                               cudaMemcpyHostToDevice, ctx.stream));
    return d_ptr;
  };

  auto d_in_strides  = upload(in_strides_elem);
  auto d_out_strides = upload(out_strides_elem);
  auto d_out_to_in   = upload(out_to_in_map);
  auto d_red_strides = upload(red_strides_elem);
  auto d_red_axes    = upload(reduce_axes);

  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  return std::make_unique<Argmax>(
      settings, idesc, ctx.stream, total_out, total_red, static_cast<int>(out_shape.size()),
      static_cast<int>(reduce_axes.size()), std::move(d_in_strides), std::move(d_out_strides),
      std::move(d_out_to_in), std::move(d_red_strides), std::move(d_red_axes));
}

std::unique_ptr<holoflow::core::ISyncTask>
ArgmaxFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_argmax = dynamic_cast<Argmax *>(old_task.get());
  if (old_argmax != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<ArgmaxSettings>();
    if (settings == old_argmax->settings() && same_desc(input_descs[0], old_argmax->idesc())) {
      old_argmax->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
