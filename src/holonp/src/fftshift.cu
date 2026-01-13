#include "holonp/fftshift.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const FFTShiftSettings &s) { j = nlohmann::json{{"axes", s.axes}}; }

void from_json(const nlohmann::json &j, FFTShiftSettings &s) {
  if (j.contains("axes") && !j.at("axes").is_null()) {
    j.at("axes").get_to(s.axes);
  } else {
    s.axes.clear();
  }
}

namespace {

constexpr int kMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("FFTShift: " + msg);
  }
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

inline std::vector<int> normalize_axes(const std::vector<int> &axes, int ndim) {
  if (axes.empty()) {
    std::vector<int> out(ndim);
    std::iota(out.begin(), out.end(), 0);
    return out;
  }

  std::vector<int> out;
  out.reserve(axes.size());
  for (int a : axes) {
    if (a < 0) {
      a += ndim;
    }
    check(a >= 0 && a < ndim, "axis out of range");
    out.push_back(a);
  }

  std::ranges::sort(out);
  auto dup = std::adjacent_find(out.begin(), out.end());
  check(dup == out.end(), "axes must be unique");
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

__global__ void fftshift_nd_contig_kernel_u8(const std::uint8_t *__restrict__ in,
                                             std::uint8_t *__restrict__ out, int elem_size,
                                             int ndim, const std::int64_t *__restrict__ shape,
                                             const std::int64_t *__restrict__ strides,
                                             const std::int64_t *__restrict__ shifts,
                                             std::int64_t total_elems) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_elems) {
    return;
  }

  std::int64_t rem       = tid;
  std::int64_t in_off_el = 0;

  for (int k = 0; k < ndim; ++k) {
    const std::int64_t stride    = strides[k];
    const std::int64_t out_idx_k = (stride > 0) ? (rem / stride) : 0;
    rem -= out_idx_k * stride;

    const std::int64_t dim      = shape[k];
    const std::int64_t shift    = shifts[k];
    std::int64_t       in_idx_k = out_idx_k + shift;
    if (in_idx_k >= dim) {
      in_idx_k -= dim;
    }
    in_off_el += in_idx_k * stride;
  }

  const auto *src = in + static_cast<size_t>(in_off_el) * static_cast<size_t>(elem_size);
  auto       *dst = out + static_cast<size_t>(tid) * static_cast<size_t>(elem_size);

  for (int b = 0; b < elem_size; ++b) {
    dst[b] = src[b];
  }
}

} // namespace

FFTShift::FFTShift(const FFTShiftSettings &settings, cudaStream_t stream, size_t ndim,
                   size_t total_elems, HostPtr<std::int64_t> h_shape, DevPtr<std::int64_t> d_shape,
                   HostPtr<std::int64_t> h_strides, DevPtr<std::int64_t> d_strides,
                   HostPtr<std::int64_t> h_shifts, DevPtr<std::int64_t> d_shifts)
    : settings_(settings), stream_(stream), ndim_(ndim), total_elems_(total_elems),
      h_shape_(std::move(h_shape)), d_shape_(std::move(d_shape)), h_strides_(std::move(h_strides)),
      d_strides_(std::move(d_strides)), h_shifts_(std::move(h_shifts)),
      d_shifts_(std::move(d_shifts)) {}

holoflow::core::OpResult FFTShift::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  (void)idesc;
  (void)odesc;

  constexpr int block_size = 256;
  const auto    total_i64  = static_cast<std::int64_t>(total_elems_);
  const int     grid_size  = static_cast<int>((total_i64 + block_size - 1) / block_size);

  const int esz = static_cast<int>(size_of(idesc.dtype));

  fftshift_nd_contig_kernel_u8<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<const std::uint8_t *>(idata), reinterpret_cast<std::uint8_t *>(odata), esz,
      static_cast<int>(ndim_), d_shape_.get(), d_strides_.get(), d_shifts_.get(), total_i64);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
FFTShiftFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<FFTShiftSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  (void)size_of(idesc.dtype);
  (void)normalize_axes(settings.axes, ndim);

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTShiftFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  const auto settings  = jsettings.get<FFTShiftSettings>();
  (void)infer_res;

  const auto &idesc = input_descs[0];
  const int   ndim  = static_cast<int>(idesc.shape.size());
  const auto  total = product_shape(idesc.shape);

  const auto axes = normalize_axes(settings.axes, ndim);

  auto h_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_shape.get()[i] = static_cast<std::int64_t>(idesc.shape[static_cast<size_t>(i)]);
  }

  const auto strides_vec = make_contig_strides(idesc.shape);
  auto       h_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_strides.get()[i] = strides_vec[static_cast<size_t>(i)];
  }

  auto h_shifts = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_shifts.get()[i] = 0;
  }
  for (int a : axes) {
    h_shifts.get()[a] = h_shape.get()[a] / 2;
  }

  auto d_shape   = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_shifts  = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), h_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_strides.get(), h_strides.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_shifts.get(), h_shifts.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFTShift(settings, ctx.stream, static_cast<size_t>(ndim), total, std::move(h_shape),
                   std::move(d_shape), std::move(h_strides), std::move(d_strides),
                   std::move(h_shifts), std::move(d_shifts)));
}

} // namespace holonp
