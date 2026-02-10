#include "holonp/min.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const MinSettings &s) {
  j = nlohmann::json{{"keepdims", s.keepdims}};
  if (s.axis.empty()) {
    j["axis"] = nullptr;
  } else if (s.axis.size() == 1) {
    j["axis"] = s.axis[0];
  } else {
    j["axis"] = s.axis;
  }
}

void from_json(const nlohmann::json &j, MinSettings &s) {
  s.axis.clear();
  if (j.contains("axis")) {
    const auto &ja = j.at("axis");
    if (ja.is_null()) {
      s.axis.clear();
    } else if (ja.is_number_integer()) {
      s.axis = {ja.get<int>()};
    } else if (ja.is_array()) {
      ja.get_to(s.axis);
    } else {
      throw std::invalid_argument("MinSettings: axis must be int, array, or null");
    }
  } else if (j.contains("axes")) {
    const auto &ja = j.at("axes");
    if (ja.is_null()) {
      s.axis.clear();
    } else if (ja.is_number_integer()) {
      s.axis = {ja.get<int>()};
    } else if (ja.is_array()) {
      ja.get_to(s.axis);
    } else {
      throw std::invalid_argument("MinSettings: axes must be int, array, or null");
    }
  }

  if (j.contains("keepdims")) {
    j.at("keepdims").get_to(s.keepdims);
  } else {
    s.keepdims = false;
  }
}

namespace {

constexpr int kMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Min: " + msg);
  }
}

inline size_t num_elements(std::span<const size_t> shape) {
  constexpr size_t max = std::numeric_limits<size_t>::max();
  size_t           n   = 1;
  for (auto d : shape) {
    if (d == 0) {
      return 0;
    }
    if (n > max / d) {
      throw std::overflow_error("Min: num_elements overflow");
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
  auto dup = std::adjacent_find(out.begin(), out.end());
  check(dup == out.end(), "axes must be unique");
  return out;
}

struct MinPlan {
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

MinPlan build_plan(const MinSettings &settings, std::span<const size_t> shape) {
  const int ndim = static_cast<int>(shape.size());
  check(ndim <= kMaxNDim, "input ndim too large");

  const auto        reduce_axes = normalize_axes(settings.axis, ndim);
  std::vector<bool> reduce_mask(static_cast<size_t>(ndim), false);
  for (int a : reduce_axes) {
    reduce_mask[static_cast<size_t>(a)] = true;
  }

  MinPlan plan;
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
        throw std::overflow_error("Min: reduction size overflow");
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

template <typename T> struct MinTraits {
  __device__ static bool better(T v, T best) { return v < best; }
};

// Match NumPy's lexicographic ordering on complex values (real, then imag).
template <> struct MinTraits<cuFloatComplex> {
  __device__ static bool better(cuFloatComplex v, cuFloatComplex best) {
    if (v.x < best.x) {
      return true;
    }
    if (v.x > best.x) {
      return false;
    }
    return v.y < best.y;
  }
};

template <typename InT>
__global__ void
min_kernel(const InT *__restrict__ in, InT *out, int out_ndim,
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

  InT  best{};
  bool init = false;
  for (std::int64_t r = 0; r < total_red; ++r) {
    std::int64_t rem_r  = r;
    std::int64_t in_off = in_base;
    for (int i = 0; i < red_ndim; ++i) {
      const std::int64_t stride = red_strides[i];
      const std::int64_t idx    = (stride > 0) ? (rem_r / stride) : 0;
      rem_r -= idx * stride;
      in_off += idx * in_strides[red_axes[i]];
    }
    const InT v = in[in_off];
    if (!init) {
      best = v;
      init = true;
    } else if (MinTraits<InT>::better(v, best)) {
      best = v;
    }
  }

  out[tid] = best;
}

} // namespace

Min::Min(const MinSettings &settings, cudaStream_t stream, size_t out_ndim, size_t red_ndim,
         std::int64_t total_out, std::int64_t total_red, HostPtr<std::int64_t> h_in_strides,
         DevPtr<std::int64_t> d_in_strides, HostPtr<std::int64_t> h_out_strides,
         DevPtr<std::int64_t> d_out_strides, HostPtr<int> h_out_to_in, DevPtr<int> d_out_to_in,
         HostPtr<int> h_red_axes, DevPtr<int> d_red_axes, HostPtr<std::int64_t> h_red_strides,
         DevPtr<std::int64_t> d_red_strides)
    : settings_(settings), stream_(stream), out_ndim_(out_ndim), red_ndim_(red_ndim),
      total_out_(total_out), total_red_(total_red), h_in_strides_(std::move(h_in_strides)),
      d_in_strides_(std::move(d_in_strides)), h_out_strides_(std::move(h_out_strides)),
      d_out_strides_(std::move(d_out_strides)), h_out_to_in_(std::move(h_out_to_in)),
      d_out_to_in_(std::move(d_out_to_in)), h_red_axes_(std::move(h_red_axes)),
      d_red_axes_(std::move(d_red_axes)), h_red_strides_(std::move(h_red_strides)),
      d_red_strides_(std::move(d_red_strides)) {}

holoflow::core::OpResult Min::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  const auto    total_out = total_out_;
  constexpr int block     = 256;
  const int     grid      = static_cast<int>((total_out + block - 1) / block);

  switch (idesc.dtype) {
  case holoflow::core::DType::U8: {
    auto *in  = reinterpret_cast<const std::uint8_t *>(idata);
    auto *out = reinterpret_cast<std::uint8_t *>(odata);
    min_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::U16: {
    auto *in  = reinterpret_cast<const std::uint16_t *>(idata);
    auto *out = reinterpret_cast<std::uint16_t *>(odata);
    min_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::F32: {
    auto *in  = reinterpret_cast<const float *>(idata);
    auto *out = reinterpret_cast<float *>(odata);
    min_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  case holoflow::core::DType::CF32: {
    auto *in  = reinterpret_cast<const cuFloatComplex *>(idata);
    auto *out = reinterpret_cast<cuFloatComplex *>(odata);
    min_kernel<<<grid, block, 0, stream_>>>(
        in, out, static_cast<int>(out_ndim_), d_out_strides_.get(), d_out_to_in_.get(),
        d_in_strides_.get(), d_red_axes_.get(), d_red_strides_.get(), static_cast<int>(red_ndim_),
        total_out_, total_red_);
    break;
  }
  default:
    logger()->error("[Min::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MinFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<MinSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");

  const auto plan = build_plan(settings, idesc.shape);

  holoflow::core::TDesc odesc = idesc;
  odesc.shape                 = plan.out_shape;

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
MinFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  const auto settings  = jsettings.get<MinSettings>();
  (void)infer_res;

  const auto &idesc = input_descs[0];
  const auto  plan  = build_plan(settings, idesc.shape);

  const size_t ndim     = idesc.shape.size();
  const size_t out_ndim = static_cast<size_t>(plan.out_ndim);
  const size_t red_ndim = static_cast<size_t>(plan.red_ndim);

  HostPtr<std::int64_t> h_in_strides;
  DevPtr<std::int64_t>  d_in_strides;
  if (ndim > 0) {
    h_in_strides = curaii::make_unique_host_ptr<std::int64_t>(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      h_in_strides.get()[i] = plan.in_strides[i];
    }
    d_in_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_in_strides.get(), h_in_strides.get(), sizeof(std::int64_t) * ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  HostPtr<std::int64_t> h_out_strides;
  DevPtr<std::int64_t>  d_out_strides;
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

  HostPtr<int> h_out_to_in;
  DevPtr<int>  d_out_to_in;
  if (out_ndim > 0) {
    h_out_to_in = curaii::make_unique_host_ptr<int>(out_ndim);
    for (size_t i = 0; i < out_ndim; ++i) {
      h_out_to_in.get()[i] = plan.out_to_in[i];
    }
    d_out_to_in = curaii::make_unique_device_ptr<int>(out_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_out_to_in.get(), h_out_to_in.get(), sizeof(int) * out_ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  HostPtr<int> h_red_axes;
  DevPtr<int>  d_red_axes;
  if (red_ndim > 0) {
    h_red_axes = curaii::make_unique_host_ptr<int>(red_ndim);
    for (size_t i = 0; i < red_ndim; ++i) {
      h_red_axes.get()[i] = plan.reduce_axes[i];
    }
    d_red_axes = curaii::make_unique_device_ptr<int>(red_ndim);
    CUDA_CHECK(cudaMemcpyAsync(d_red_axes.get(), h_red_axes.get(), sizeof(int) * red_ndim,
                               cudaMemcpyHostToDevice, ctx.stream));
  }

  HostPtr<std::int64_t> h_red_strides;
  DevPtr<std::int64_t>  d_red_strides;
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
      new Min(settings, ctx.stream, out_ndim, red_ndim, plan.total_out, plan.total_red,
              std::move(h_in_strides), std::move(d_in_strides), std::move(h_out_strides),
              std::move(d_out_strides), std::move(h_out_to_in), std::move(d_out_to_in),
              std::move(h_red_axes), std::move(d_red_axes), std::move(h_red_strides),
              std::move(d_red_strides)));
}

} // namespace holonp
