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
// Helpers
// -----------------------------------------------------------------------------

namespace {

constexpr int kKernelMaxNDim = 16;

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
std::vector<size_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  std::vector<size_t> strides(desc.shape.size());
  if (!desc.strides.empty()) {
    for (size_t i = 0; i < desc.strides.size(); ++i)
      strides.at(i) = desc.strides.at(i);
  } else {
    size_t acc = holoflow::core::size_of(desc.dtype);
    for (size_t i = desc.shape.size(); i-- > 0;) {
      strides.at(i) = acc;
      acc *= desc.shape.at(i);
    }
  }
  return strides;
}

} // namespace

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

Transpose::Transpose(cudaStream_t stream, size_t num_bytes)
    : stream_(stream), num_bytes_(num_bytes) {}

holoflow::core::OpResult Transpose::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<const std::uint8_t *>(ctx.inputs[0].data());
  auto *odata = reinterpret_cast<std::uint8_t *>(ctx.outputs[0].data());

  // 1. Short-circuit: The framework respected our in-place request.
  // This is a true zero-cost view transpose.
  if (idata == odata) {
    return holoflow::core::OpResult::Ok;
  }

  // 2. Fallback: The framework allocated a new buffer.
  // Because infer() returned permuted strides, the output tensor's metadata
  // expects the exact same physical memory layout as the input.
  // A flat byte copy is all that is needed.
  CUDA_CHECK(cudaMemcpyAsync(odata, idata, num_bytes_, cudaMemcpyDeviceToDevice, stream_));

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

  // 1. Get input strides (calculate dense ones if empty)
  auto in_strides = get_strides_bytes(idesc);

  // 2. Permute shape AND strides
  std::vector<size_t> oshape(ndim);
  std::vector<size_t> ostrides(ndim);
  for (int i = 0; i < ndim; ++i) {
    oshape[i]   = idesc.shape[axes[i]];
    ostrides[i] = in_strides[axes[i]];
  }

  // 3. Create output descriptor using the permuted strides and identical offset
  holoflow::core::TDesc odesc(oshape, idesc.dtype, idesc.mem_loc, ostrides, idesc.offset);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {{0, 0}}, // Crucial: Request aliasing from graph engine
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
TransposeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)jsettings;
  const auto &idesc = input_descs[0];

  // We only need the total byte size to handle the memcpy fallback if aliasing fails.
  size_t num_bytes = idesc.num_bytes();

  return std::unique_ptr<holoflow::core::ISyncTask>(new Transpose(ctx.stream, num_bytes));
}

} // namespace holonp