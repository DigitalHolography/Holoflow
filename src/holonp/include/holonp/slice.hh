// Copyright 2025 Digital Holography Foundation
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

#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holonp {

// Represents a range selection (e.g., 0:5:1)
// Preserves dimensionality.
struct SliceRange {
  std::optional<std::int64_t> start = std::nullopt;
  std::optional<std::int64_t> stop  = std::nullopt;
  std::int64_t                step  = 1;
};

// Represents either a SliceRange or a direct Index
// - SliceRange: keeps dimension (e.g. X[0:1])
// - int64: drops dimension (e.g. X[0])
using SliceItem = std::variant<SliceRange, std::int64_t>;

struct SliceSettings {
  std::vector<SliceItem> slices;
};

void to_json(nlohmann::json &j, const SliceRange &s);
void from_json(const nlohmann::json &j, SliceRange &s);

void to_json(nlohmann::json &j, const SliceItem &s);
void from_json(const nlohmann::json &j, SliceItem &s);

void to_json(nlohmann::json &j, const SliceSettings &s);
void from_json(const nlohmann::json &j, SliceSettings &s);

// Creates a view into the tensor without copying data.
// Supports both slicing (sub-views) and integer indexing (dimensionality reduction).
class Slice : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Slice() = default;

  friend class SliceFactory;
};

class SliceFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp