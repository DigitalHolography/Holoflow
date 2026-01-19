#include "holonp/transpose.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const TransposeSettings &s) {
  j = nlohmann::json{{"axes", s.axes}};
}

void from_json(const nlohmann::json &j, TransposeSettings &s) { j.at("axes").get_to(s.axes); }

namespace {

constexpr int kMaxNDim = 16;

inline std::vector<int> normalize_and_validate_axes(std::span<const int> axes, int ndim) {
  if (static_cast<int>(axes.size()) != ndim) {
    throw std::invalid_argument("Transpose: axes length must match input ndim");
  }

  std::vector<int> norm;
  norm.reserve(axes.size());

  for (int a : axes) {
    if (a < 0) {
      a += ndim;
    }
    if (a < 0 || a >= ndim) {
      throw std::invalid_argument("Transpose: axis out of range");
    }
    norm.push_back(a);
  }

  auto sorted = norm;
  std::ranges::sort(sorted);
  for (int i = 0; i < ndim; ++i) {
    if (sorted[i] != i) {
      throw std::invalid_argument("Transpose: axes must be a permutation of [0..ndim-1]");
    }
  }

  return norm;
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  // Prevent silent overflow turning into small totals.
  // If you want explicit overflow checks, add them here.
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

// Data-moving generic transpose for contiguous input/output.
// out is written in row-major order (linear tid).
__global__ void transpose_nd_contig_kernel_u8(const std::uint8_t *__restrict__ in,
                                              std::uint8_t *__restrict__ out, int elem_size,
                                              int ndim,
                                              const std::int64_t *__restrict__ in_shape, // [ndim]
                                              const int *__restrict__ axes,              // [ndim]
                                              std::int64_t total_elems) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_elems) {
    return;
  }

  // Decode tid (out linear) -> out multi-index, then map to input linear offset.
  // out_shape[k] == in_shape[axes[k]]
  std::int64_t rem       = tid;
  std::int64_t in_off_el = 0;

  for (int k = 0; k < ndim; ++k) {
    // suffix = product(out_shape[k+1..]) = product(in_shape[axes[k+1..]])
    std::int64_t out_suffix = 1;
    for (int j = k + 1; j < ndim; ++j) {
      out_suffix *= in_shape[axes[j]];
    }

    const std::int64_t out_idx_k = rem / out_suffix;
    rem -= out_idx_k * out_suffix;

    // Input contiguous stride(in_axis) = product(in_shape[in_axis+1..])
    const int    in_axis   = axes[k];
    std::int64_t in_stride = 1;
    for (int j = in_axis + 1; j < ndim; ++j) {
      in_stride *= in_shape[j];
    }

    in_off_el += out_idx_k * in_stride;
  }

  const auto *src = in + static_cast<size_t>(in_off_el) * static_cast<size_t>(elem_size);
  auto       *dst = out + static_cast<size_t>(tid) * static_cast<size_t>(elem_size);

  for (int b = 0; b < elem_size; ++b) {
    dst[b] = src[b];
  }
}

} // namespace

Transpose::Transpose(const TransposeSettings &settings, cudaStream_t stream, size_t ndim,
                     size_t total, HostPtr<std::int64_t> h_in_shape,
                     DevPtr<std::int64_t> d_in_shape, HostPtr<int> h_axes, DevPtr<int> d_axes)
    : settings_(settings), stream_(stream), ndim_(ndim), total_(total),
      h_in_shape_(std::move(h_in_shape)), d_in_shape_(std::move(d_in_shape)),
      h_axes_(std::move(h_axes)), d_axes_(std::move(d_axes)) {}

holoflow::core::OpResult Transpose::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = ctx.inputs[0].data();
  auto *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  constexpr int block_size = 256;
  const auto    total_i64  = static_cast<std::int64_t>(total_);
  const int     grid_size  = static_cast<int>((total_i64 + block_size - 1) / block_size);

  const int esz = static_cast<int>(size_of(idesc.dtype));

  transpose_nd_contig_kernel_u8<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<const std::uint8_t *>(idata), reinterpret_cast<std::uint8_t *>(odata), esz,
      static_cast<int>(ndim_), d_in_shape_.get(), d_axes_.get(), total_i64);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
TransposeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      throw std::invalid_argument("TransposeFactory inference error: " + msg);
    }
  };

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  // Ensure dtype supported by the kernel/copy.
  (void)size_of(idesc.dtype);

  const auto settings = jsettings.get<TransposeSettings>();
  const auto axes     = normalize_and_validate_axes(settings.axes, ndim);

  holoflow::core::TDesc odesc = idesc;
  odesc.shape.resize(ndim);
  for (int k = 0; k < ndim; ++k) {
    odesc.shape[k] = idesc.shape[static_cast<size_t>(axes[k])];
  }

  // total must be > 0 for execution (consistent with your Arange pattern).
  check(product_shape(idesc.shape) > 0, "input tensor has zero elements");

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
TransposeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  // Infer performs all validation. Also ensures output desc is coherent.
  (void)this->infer(input_descs, jsettings);

  const auto  settings = jsettings.get<TransposeSettings>();
  const auto &idesc    = input_descs[0];

  const size_t ndim  = idesc.shape.size();
  const auto   axes  = normalize_and_validate_axes(settings.axes, static_cast<int>(ndim));
  const size_t total = product_shape(idesc.shape);

  auto h_in_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_in_shape.get()[i] = static_cast<std::int64_t>(idesc.shape[i]);
  }

  auto h_axes = curaii::make_unique_host_ptr<int>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_axes.get()[i] = axes[i];
  }

  auto d_in_shape = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_axes     = curaii::make_unique_device_ptr<int>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_in_shape.get(), h_in_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_axes.get(), h_axes.get(), sizeof(int) * ndim, cudaMemcpyHostToDevice,
                             ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new Transpose(settings, ctx.stream, ndim, total, std::move(h_in_shape), std::move(d_in_shape),
                    std::move(h_axes), std::move(d_axes)));
}

} // namespace holonp