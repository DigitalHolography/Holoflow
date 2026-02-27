#include "holonp/transpose.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace holonp {

// -----------------------------------------------------------------------------
// JSON Serialization
// -----------------------------------------------------------------------------

void to_json(nlohmann::json &j, const TransposeSettings &s) {
  j = nlohmann::json{{"axes", s.axes}};
}

void from_json(const nlohmann::json &j, TransposeSettings &s) {
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
}

// -----------------------------------------------------------------------------
// Helpers & Kernels
// -----------------------------------------------------------------------------

namespace {

constexpr int kKernelMaxNDim = 16; // Hard limit for kernel registers

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Transpose: " + msg);
}

std::vector<int> normalize_axes(const std::vector<int> &axes, int ndim) {
  check(static_cast<int>(axes.size()) == ndim, "axes length must match ndim");

  std::vector<int> norm;
  norm.reserve(ndim);
  std::vector<bool> seen(ndim, false);

  for (int a : axes) {
    if (a < 0)
      a += ndim;
    check(a >= 0 && a < ndim, "axis out of bounds");
    check(!seen[a], "duplicate axis in permutation");
    seen[a] = true;
    norm.push_back(a);
  }
  return norm;
}

// Ensure strides exist or create contiguous default
std::vector<std::int64_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  std::vector<std::int64_t> strides(desc.shape.size());
  if (!desc.strides.empty()) {
    for (size_t i = 0; i < desc.strides.size(); ++i)
      strides.at(i) = static_cast<std::int64_t>(desc.strides.at(i));
  } else {
    size_t acc = holoflow::core::size_of(desc.dtype);
    for (size_t i = desc.shape.size(); i-- > 0;) {
      strides.at(i) = static_cast<std::int64_t>(acc);
      acc *= desc.shape.at(i);
    }
  }
  return strides;
}

// General N-Dimensional Transpose Kernel
// Strategy: Linearize threads over output elements (contiguous).
// Decompose output index -> coordinates -> recompute input offset using permuted strides.
// This handles arbitrary input strides and arbitrary permutations.
//
// Note: For 2D/3D specific cases, specialized shared-memory tiled kernels are much faster.
// This generic kernel is a fallback for N-D.
__global__ void transpose_general_kernel(
    const std::uint8_t *__restrict__ in, std::uint8_t *__restrict__ out, int elem_size, int ndim,
    const std::int64_t
        *__restrict__ in_strides_permuted,        // Stride of the axis mapped to output dim i
    const std::int64_t *__restrict__ out_shape,   // Shape of output
    const std::int64_t *__restrict__ out_strides, // Strides of output (computed or dense)
    std::int64_t total_elems) {
  const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elems)
    return;

  // 1. Calculate Multi-Index from Linear Output ID
  // We traverse the output dimensions to reconstruct coordinates.
  // While doing so, we simultaneously calculate the Input Offset.

  std::int64_t temp_idx  = tid;
  std::int64_t in_offset = 0;

  // Unroll manually for small N if needed, or loop.
  // Using registers for shape/strides is faster than global memory.
  // We assume ndim <= kKernelMaxNDim (checked in factory).

  // We iterate from dim 0 to ndim-1.
  // Standard row-major index reconstruction:
  // coord[i] = temp_idx / stride[i]; temp_idx %= stride[i];

  for (int i = 0; i < ndim; ++i) {
    // Optim: out_strides should be pre-calculated in bytes or elements.
    // Here we assume bytes for flexibility or elements?
    // Let's assume input arrays are Bytes.
    // But for index calc, we need element strides.

    // Actually, simpler logic:
    // coord = (tid / out_stride_elem[i]) % out_shape[i]
    // But division is expensive.

    // Recursive remainder approach:
    // out_stride[i] is the stride in ELEMENTS for dimension i.
    // We rely on the caller providing 'out_strides' which are the accumulated products.

    std::int64_t stride = out_strides[i]; // E.g. H*W, W, 1
    std::int64_t coord  = temp_idx / stride;
    temp_idx -= coord * stride;

    // Map this coordinate to input memory
    in_offset += coord * in_strides_permuted[i];
  }

  // 2. Copy Element
  const auto *src = in + in_offset; // Strides are in bytes
  auto       *dst = out + tid * elem_size;

  // Vectorized copy if possible, else byte loop
  for (int b = 0; b < elem_size; ++b) {
    dst[b] = src[b];
  }
}

} // namespace

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

Transpose::Transpose(const TransposeSettings &settings, cudaStream_t stream, size_t ndim,
                     size_t total_elems, DevPtr<std::int64_t> d_in_strides,
                     DevPtr<std::int64_t> d_out_strides, DevPtr<std::int64_t> d_out_shape)
    : settings_(settings), stream_(stream), ndim_(ndim), total_elems_(total_elems),
      d_in_strides_(std::move(d_in_strides)), d_out_strides_(std::move(d_out_strides)),
      d_out_shape_(std::move(d_out_shape)) {}

holoflow::core::OpResult Transpose::execute(holoflow::core::SyncCtx &ctx) {
  auto     *idata     = reinterpret_cast<const std::uint8_t *>(ctx.inputs[0].data());
  auto     *odata     = reinterpret_cast<std::uint8_t *>(ctx.outputs[0].data());
  const int elem_size = static_cast<int>(holoflow::core::size_of(ctx.inputs[0].desc.dtype));

  const int          block_size = 256;
  const std::int64_t total      = static_cast<std::int64_t>(total_elems_);
  const int          grid_size  = static_cast<int>((total + block_size - 1) / block_size);

  transpose_general_kernel<<<grid_size, block_size, 0, stream_>>>(
      idata, odata, elem_size, static_cast<int>(ndim_), d_in_strides_.get(), d_out_shape_.get(),
      d_out_strides_.get(), total);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
TransposeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "device memory only");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0 && ndim <= kKernelMaxNDim,
        "invalid ndim (max " + std::to_string(kKernelMaxNDim) + ")");

  auto settings = jsettings.get<TransposeSettings>();
  auto axes     = normalize_axes(settings.axes, ndim);

  // Calculate Output Shape
  std::vector<size_t> oshape(ndim);
  for (int i = 0; i < ndim; ++i) {
    oshape[i] = idesc.shape[axes[i]];
  }

  holoflow::core::TDesc odesc(oshape, idesc.dtype, idesc.mem_loc);

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

  // 1. Setup
  const auto  &idesc = input_descs[0];
  const int    ndim  = static_cast<int>(idesc.shape.size());
  const size_t total_elems =
      std::accumulate(idesc.shape.begin(), idesc.shape.end(), size_t{1}, std::multiplies<>{});

  auto settings = jsettings.get<TransposeSettings>();
  auto axes     = normalize_axes(settings.axes, ndim);

  // 2. Prepare Metadata for Kernel
  // We need:
  // a. Output Shape (for coordinate reconstruction)
  // b. Output Strides (element-wise, for division)
  // c. Input Strides Permuted (byte-wise, for final address calc)

  std::vector<std::int64_t> h_out_shape(ndim);
  std::vector<std::int64_t> h_out_strides(ndim);     // Element strides
  std::vector<std::int64_t> h_in_strides_perm(ndim); // Byte strides

  // Get input strides (handle defaults if empty)
  auto input_strides_bytes = get_strides_bytes(idesc);

  // Calculate Output shape & permuted input strides
  for (int i = 0; i < ndim; ++i) {
    h_out_shape[i]       = static_cast<std::int64_t>(idesc.shape[axes[i]]);
    h_in_strides_perm[i] = input_strides_bytes[axes[i]];
  }

  // Calculate Output Strides (Compact/Contiguous logic)
  // Row-major: Stride[i] = product(Shape[i+1...])
  std::int64_t acc = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    h_out_strides[i] = acc;
    acc *= h_out_shape[i];
  }

  // 3. Upload to Device
  auto d_out_shape   = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_out_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_in_strides  = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_out_shape.get(), h_out_shape.data(), ndim * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_out_strides.get(), h_out_strides.data(), ndim * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in_strides.get(), h_in_strides_perm.data(),
                             ndim * sizeof(std::int64_t), cudaMemcpyHostToDevice, ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new Transpose(settings, ctx.stream, ndim, total_elems, std::move(d_in_strides),
                    std::move(d_out_strides), std::move(d_out_shape)));
}

} // namespace holonp