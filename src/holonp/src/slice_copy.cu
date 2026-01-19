#include "holonp/slice_copy.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const SliceItem &s) {
  j = nlohmann::json{
      {"start", s.start.has_value() ? nlohmann::json(*s.start) : nlohmann::json(nullptr)},
      {"stop", s.stop.has_value() ? nlohmann::json(*s.stop) : nlohmann::json(nullptr)},
      {"step", s.step},
  };
}

void from_json(const nlohmann::json &j, SliceItem &s) {
  if (j.contains("start") && !j.at("start").is_null()) {
    s.start = j.at("start").get<std::int64_t>();
  } else {
    s.start = std::nullopt;
  }

  if (j.contains("stop") && !j.at("stop").is_null()) {
    s.stop = j.at("stop").get<std::int64_t>();
  } else {
    s.stop = std::nullopt;
  }

  j.at("step").get_to(s.step);
}

void to_json(nlohmann::json &j, const SliceCopySettings &s) {
  j = nlohmann::json{{"slices", s.slices}};
}

void from_json(const nlohmann::json &j, SliceCopySettings &s) { j.at("slices").get_to(s.slices); }

namespace {

constexpr int kMaxNDim = 16;

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("SliceCopy: " + msg);
  }
}

struct NormalizedSlice {
  std::int64_t start;
  std::int64_t stop; // exclusive
  std::int64_t step; // > 0
};

// NumPy-like normalization for basic positive-step slicing.
// Supports negative start/stop by wrapping with dim.
// Defaults: start=0, stop=dim, step=1.
// Clamps to [0, dim].
inline NormalizedSlice normalize_slice(const SliceItem &s, std::int64_t dim) {
  check(dim >= 0, "invalid dimension");

  const auto step = s.step;
  check(step > 0, "only positive step is supported for now");

  std::int64_t start = s.start.value_or(0);
  std::int64_t stop  = s.stop.value_or(dim);

  if (start < 0)
    start += dim;
  if (stop < 0)
    stop += dim;

  start = std::clamp<std::int64_t>(start, 0, dim);
  stop  = std::clamp<std::int64_t>(stop, 0, dim);

  // If start >= stop => empty slice (produces 0-length dim)
  return NormalizedSlice{start, stop, step};
}

inline std::int64_t out_len(const NormalizedSlice &ns) {
  if (ns.start >= ns.stop) {
    return 0;
  }
  const auto span = ns.stop - ns.start;
  // ceil_div(span, step)
  return (span + ns.step - 1) / ns.step;
}

inline std::vector<NormalizedSlice>
normalize_and_validate_slices(std::span<const SliceItem> slices, std::span<const size_t> in_shape) {
  const int ndim = static_cast<int>(in_shape.size());
  check(static_cast<int>(slices.size()) == ndim, "slices length must match input ndim");
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  std::vector<NormalizedSlice> out;
  out.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    out.push_back(normalize_slice(slices[i], static_cast<std::int64_t>(in_shape[i])));
  }
  return out;
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

// Data-moving basic slice copy for contiguous input/output.
// Output is written in row-major order (linear tid).
__global__ void
slice_copy_nd_contig_kernel_u8(const std::uint8_t *__restrict__ in, std::uint8_t *__restrict__ out,
                               int elem_size, int ndim,
                               const std::int64_t *__restrict__ in_strides, // [ndim] (elements)
                               const std::int64_t *__restrict__ start,      // [ndim]
                               const std::int64_t *__restrict__ step,       // [ndim]
                               const std::int64_t *__restrict__ out_shape,  // [ndim]
                               std::int64_t total_out_elems) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_out_elems) {
    return;
  }

  // Decode tid (out linear) -> out multi-index, then map to input linear offset.
  std::int64_t rem       = tid;
  std::int64_t in_off_el = 0;

  for (int k = 0; k < ndim; ++k) {
    std::int64_t out_suffix = 1;
    for (int j = k + 1; j < ndim; ++j) {
      out_suffix *= out_shape[j];
    }

    const std::int64_t out_idx_k = (out_suffix > 0) ? (rem / out_suffix) : 0;
    rem -= out_idx_k * out_suffix;

    const std::int64_t in_idx_k = start[k] + out_idx_k * step[k];
    in_off_el += in_idx_k * in_strides[k];
  }

  const auto *src = in + static_cast<size_t>(in_off_el) * static_cast<size_t>(elem_size);
  auto       *dst = out + static_cast<size_t>(tid) * static_cast<size_t>(elem_size);

  for (int b = 0; b < elem_size; ++b) {
    dst[b] = src[b];
  }
}

} // namespace

SliceCopy::SliceCopy(const SliceCopySettings &settings, cudaStream_t stream, size_t ndim,
                     size_t total_out, HostPtr<std::int64_t> h_in_shape,
                     DevPtr<std::int64_t> d_in_shape, HostPtr<std::int64_t> h_in_strides,
                     DevPtr<std::int64_t> d_in_strides, HostPtr<std::int64_t> h_start,
                     DevPtr<std::int64_t> d_start, HostPtr<std::int64_t> h_step,
                     DevPtr<std::int64_t> d_step, HostPtr<std::int64_t> h_out_shape,
                     DevPtr<std::int64_t> d_out_shape)
    : settings_(settings), stream_(stream), ndim_(ndim), total_out_(total_out),
      h_in_shape_(std::move(h_in_shape)), d_in_shape_(std::move(d_in_shape)),
      h_in_strides_(std::move(h_in_strides)), d_in_strides_(std::move(d_in_strides)),
      h_start_(std::move(h_start)), d_start_(std::move(d_start)), h_step_(std::move(h_step)),
      d_step_(std::move(d_step)), h_out_shape_(std::move(h_out_shape)),
      d_out_shape_(std::move(d_out_shape)) {}

holoflow::core::OpResult SliceCopy::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = ctx.inputs[0].data();
  auto *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  constexpr int block_size = 256;
  const auto    total_i64  = static_cast<std::int64_t>(total_out_);
  const int     grid_size  = static_cast<int>((total_i64 + block_size - 1) / block_size);

  const int esz = static_cast<int>(size_of(idesc.dtype));

  slice_copy_nd_contig_kernel_u8<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<const std::uint8_t *>(idata), reinterpret_cast<std::uint8_t *>(odata), esz,
      static_cast<int>(ndim_), d_in_strides_.get(), d_start_.get(), d_step_.get(),
      d_out_shape_.get(), total_i64);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
SliceCopyFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  // Ensure dtype supported (throws if unknown).
  (void)size_of(idesc.dtype);

  // For now, require non-empty input for execution simplicity.
  check(product_shape(idesc.shape) > 0, "input tensor has zero elements");

  const auto settings = jsettings.get<SliceCopySettings>();
  const auto nslices  = normalize_and_validate_slices(settings.slices, idesc.shape);

  // holoflow::core::TDesc odesc = idesc;
  // odesc.shape.resize(static_cast<size_t>(ndim));
  // for (int i = 0; i < ndim; ++i) {
  //   odesc.shape[static_cast<size_t>(i)] =
  //       static_cast<size_t>(out_len(nslices[static_cast<size_t>(i)]));
  // }
  // 

  std::vector<size_t> out_shape(static_cast<size_t>(ndim));
  for (int i = 0; i < ndim; ++i) {
    out_shape[static_cast<size_t>(i)] =
        static_cast<size_t>(out_len(nslices[static_cast<size_t>(i)]));
  }

  const auto total_out = product_shape(out_shape);
  check(total_out > 0, "slice produces an empty tensor (not supported for now)");

  holoflow::core::TDesc odesc(out_shape, idesc.dtype, holoflow::core::MemLoc::Device);

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
SliceCopyFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  // Reuse infer to validate and compute output desc consistency.
  const auto infer_res = this->infer(input_descs, jsettings);

  const auto  settings = jsettings.get<SliceCopySettings>();
  const auto &idesc    = input_descs[0];

  const size_t ndim = idesc.shape.size();
  const auto   ns   = normalize_and_validate_slices(settings.slices, idesc.shape);

  // Precompute in_shape (int64), in_strides (elements), start/step, out_shape (int64)
  auto h_in_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_in_shape.get()[i] = static_cast<std::int64_t>(idesc.shape[i]);
  }

  const auto in_strides_vec = make_contig_strides(idesc.shape);
  auto       h_in_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_in_strides.get()[i] = in_strides_vec[i];
  }

  auto h_start = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  auto h_step  = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_start.get()[i] = ns[i].start;
    h_step.get()[i]  = ns[i].step;
  }

  auto        h_out_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  const auto &odesc       = infer_res.output_descs[0];
  for (size_t i = 0; i < ndim; ++i) {
    h_out_shape.get()[i] = static_cast<std::int64_t>(odesc.shape[i]);
  }

  const size_t total_out = product_shape(odesc.shape);

  auto d_in_shape   = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_in_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_start      = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_step       = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_out_shape  = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_in_shape.get(), h_in_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in_strides.get(), h_in_strides.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_start.get(), h_start.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_step.get(), h_step.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_out_shape.get(), h_out_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(new SliceCopy(
      settings, ctx.stream, ndim, total_out, std::move(h_in_shape), std::move(d_in_shape),
      std::move(h_in_strides), std::move(d_in_strides), std::move(h_start), std::move(d_start),
      std::move(h_step), std::move(d_step), std::move(h_out_shape), std::move(d_out_shape)));
}

} // namespace holonp