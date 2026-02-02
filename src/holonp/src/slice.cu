#include "holonp/slice.hh"

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

void to_json(nlohmann::json &j, const SliceSettings &s) {
  j = nlohmann::json{{"slices", s.slices}};
}

void from_json(const nlohmann::json &j, SliceSettings &s) { j.at("slices").get_to(s.slices); }

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
    throw std::invalid_argument("Slice: " + msg);
  }
}

struct NormalizedSlice {
  std::int64_t start;
  std::int64_t stop; // exclusive
  std::int64_t step; // > 0
};

// NumPy-like normalization with Strict Bound Checking
inline NormalizedSlice normalize_slice(const SliceItem &s, std::int64_t dim) {
  check(dim >= 0, "invalid dimension");

  const auto step = s.step;
  check(step > 0, "only positive step is supported for now");

  std::int64_t start = s.start.value_or(0);
  std::int64_t stop  = s.stop.value_or(dim);

  // 1. Handle negative wrapping (e.g. -1 becomes dim-1)
  if (start < 0)
    start += dim;
  if (stop < 0)
    stop += dim;

  // 2. Strict Bound Checking 
  // Unlike NumPy which clamps silently, we throw if indices are outside [0, dim].
  if (start < 0 || start > dim) {
      throw std::out_of_range("Slice 'start' index " + std::to_string(start) + 
                              " is out of bounds for dimension size " + std::to_string(dim));
  }
  if (stop < 0 || stop > dim) {
      throw std::out_of_range("Slice 'stop' index " + std::to_string(stop) + 
                              " is out of bounds for dimension size " + std::to_string(dim));
  }

  // 3. Clamping (Redundant if strict checks above are enabled, but safe to keep)
  start = std::clamp<std::int64_t>(start, 0, dim);
  stop  = std::clamp<std::int64_t>(stop, 0, dim);

  return NormalizedSlice{start, stop, step};
}

inline std::int64_t out_len(const NormalizedSlice &ns) {
  if (ns.start >= ns.stop) {
    return 0;
  }
  const auto span = ns.stop - ns.start;
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

// [Updated] Helper to get consistent BYTES strides
// Requires dtype size to calculate default contiguous bytes
inline std::vector<size_t> ensure_strides(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty()) {
    return desc.strides;
  }
  std::vector<size_t> strides(desc.shape.size());
  // Start accumulation with element size in bytes
  size_t acc = holoflow::core::size_of(desc.dtype); 
  for (int i = static_cast<int>(desc.shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
}

} // namespace

holoflow::core::OpResult Slice::execute(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
SliceFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                   &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  // Get current strides (Byte Strides)
  const auto in_strides = ensure_strides(idesc);

  const auto settings = jsettings.get<SliceSettings>();
  const auto nslices  = normalize_and_validate_slices(settings.slices, idesc.shape);

  std::vector<size_t> out_shape(static_cast<size_t>(ndim));
  std::vector<size_t> out_strides(static_cast<size_t>(ndim));
  
  // Calculate new Offset relative to current input offset
  size_t added_offset_bytes = 0;
  // NOTE: We do NOT multiply by elem_size here because in_strides are already bytes.
  
  for (int i = 0; i < ndim; ++i) {
    const auto &ns = nslices[i];
    
    // New Shape: (stop - start) / step
    out_shape[i] = static_cast<size_t>(out_len(ns));
    
    // New Stride: old_stride_bytes * step
    out_strides[i] = in_strides[i] * static_cast<size_t>(ns.step);
    
    // Offset shift: start * old_stride_bytes
    added_offset_bytes += static_cast<size_t>(ns.start) * in_strides[i];
  }

  // Construct Output Descriptor
  const size_t final_offset = idesc.offset + added_offset_bytes;

  holoflow::core::TDesc odesc(out_shape, idesc.dtype, idesc.mem_loc, out_strides, final_offset);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false}, 
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SliceFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                   &jsettings,
                         const holoflow::core::SyncCreateCtx    &ctx) const {
  (void) infer(input_descs, jsettings);
  (void) ctx;
  return std::unique_ptr<holoflow::core::ISyncTask>(new Slice());
}

} // namespace holonp