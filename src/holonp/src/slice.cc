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

#include "holonp/slice.hh"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SliceRange &s) {
  j = nlohmann::json{
      {"start", s.start.has_value() ? nlohmann::json(*s.start) : nlohmann::json(nullptr)},
      {"stop", s.stop.has_value() ? nlohmann::json(*s.stop) : nlohmann::json(nullptr)},
      {"step", s.step},
  };
}

void from_json(const nlohmann::json &j, SliceRange &s) {
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

  // Default step to 1 if not present
  s.step = j.value("step", 1);
}

void to_json(nlohmann::json &j, const SliceItem &s) {
  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::int64_t>) {
          j = arg; // Serialize as integer
        } else {
          j = arg; // Serialize as SliceRange object
        }
      },
      s);
}

void from_json(const nlohmann::json &j, SliceItem &s) {
  if (j.is_number_integer()) {
    // If it's a number, it's a direct index (e.g. 5)
    s = j.get<std::int64_t>();
  } else {
    // If it's an object (or null/empty), treat as SliceRange
    s = j.get<SliceRange>();
  }
}

void to_json(nlohmann::json &j, const SliceSettings &s) {
  j = nlohmann::json{{"slices", s.slices}};
}

void from_json(const nlohmann::json &j, SliceSettings &s) { j.at("slices").get_to(s.slices); }

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

constexpr int kMaxNDim = 16;

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

// Helper: Validate and Normalize a single Integer Index
inline std::int64_t normalize_index(std::int64_t idx, std::int64_t dim) {
  check(dim > 0, "cannot index into 0-sized dimension");

  // Handle negative wrapping
  if (idx < 0) {
    idx += dim;
  }

  // Strict Bound Checking
  if (idx < 0 || idx >= dim) {
    throw std::out_of_range("Slice index " + std::to_string(idx) +
                            " is out of bounds for dimension size " + std::to_string(dim));
  }
  return idx;
}

// Helper: Validate and Normalize a Slice Range
inline NormalizedSlice normalize_slice_range(const SliceRange &s, std::int64_t dim) {
  check(dim >= 0, "invalid dimension");

  const auto step = s.step;
  check(step > 0, "only positive step is supported for now");

  std::int64_t start = s.start.value_or(0);
  std::int64_t stop  = s.stop.value_or(dim);

  // 1. Handle negative wrapping
  if (start < 0)
    start += dim;
  if (stop < 0)
    stop += dim;

  // 2. Strict Bound Checking (optional, depending on desired strictness vs numpy leniency)
  // For safety in this environment, we check bounds strictly relative to 0.
  // Note: NumPy usually clamps start/stop, but throws on integer indexing.
  // Here we clamp to maintain view safety.
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

// Helper to get consistent BYTES strides
inline std::vector<size_t> ensure_strides(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty()) {
    return desc.strides;
  }
  std::vector<size_t> strides(desc.shape.size());
  size_t              acc = holoflow::core::size_of(desc.dtype);
  for (int i = static_cast<int>(desc.shape.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
}

// -------------------------------------------------------------------------------------------------
// Slice task implementation
// -------------------------------------------------------------------------------------------------

class Slice : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;
};

} // namespace

holoflow::core::OpResult Slice::execute(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// SliceFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult SliceFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  const auto in_strides = ensure_strides(idesc);
  const auto settings   = jsettings.get<SliceSettings>();

  check(static_cast<int>(settings.slices.size()) == ndim,
        "number of slice items must match input ndim");

  std::vector<size_t> out_shape;
  std::vector<size_t> out_strides;
  out_shape.reserve(ndim);
  out_strides.reserve(ndim);

  // Calculate new Offset relative to current input offset
  size_t added_offset_bytes = 0;

  for (int i = 0; i < ndim; ++i) {
    const auto &item       = settings.slices[i];
    const auto  dim_size   = static_cast<std::int64_t>(idesc.shape[i]);
    const auto  dim_stride = in_strides[i];

    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;

          if constexpr (std::is_same_v<T, std::int64_t>) {
            // === CASE 1: Integer Index (Dimensionality Reduction) ===
            // Calculate offset, but do NOT add to out_shape/out_strides
            const std::int64_t idx = normalize_index(arg, dim_size);
            added_offset_bytes += static_cast<size_t>(idx) * dim_stride;
          } else {
            // === CASE 2: Slice Range (Preserve Dimension) ===
            const auto ns = normalize_slice_range(arg, dim_size);

            // Add offset for the start of the slice
            added_offset_bytes += static_cast<size_t>(ns.start) * dim_stride;

            // Push new dimension shape and stride
            out_shape.push_back(static_cast<size_t>(out_len(ns)));
            out_strides.push_back(dim_stride * static_cast<size_t>(ns.step));
          }
        },
        item);
  }

  // Construct Output Descriptor
  const size_t final_offset = idesc.offset + added_offset_bytes;

  holoflow::core::TDesc odesc(out_shape, idesc.dtype, idesc.mem_loc, out_strides, final_offset);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {{0, 0}}, // Input 0 -> Output 0
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SliceFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  (void)ctx;
  return std::make_unique<Slice>();
}

std::unique_ptr<holoflow::core::ISyncTask>
SliceFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                     std::span<const holoflow::core::TDesc>     input_descs,
                     const nlohmann::json                      &jsettings,
                     const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)ctx;
  (void)infer(input_descs, jsettings);

  auto *old_slice = dynamic_cast<Slice *>(old_task.get());
  if (old_slice == nullptr || input_descs.size() != 1) {
    return create(input_descs, jsettings, ctx);
  }

  return old_task;
}

} // namespace holonp
