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

#include <spdlog/fmt/fmt.h>

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct RecordSettings {
  std::string    file_path;
  int            recording_count;
  nlohmann::json pipeline_config;

  size_t record_queue_count;

  bool operator==(const RecordSettings &other) const {
    return recording_count == other.recording_count &&
           record_queue_count == other.record_queue_count && file_path == other.file_path &&
           pipeline_config == other.pipeline_config;
  }

  bool requires_rebuild(const RecordSettings &old) const;
};

void to_json(nlohmann::json &j, const RecordSettings &s);
void from_json(const nlohmann::json &j, RecordSettings &s);

} // namespace holotask::sources

template <>
struct fmt::formatter<holotask::sources::RecordSettings> : fmt::formatter<std::string_view> {
  auto format(const holotask::sources::RecordSettings &r, fmt::format_context &ctx) const {
    return fmt::format_to(ctx.out(), "{{ file_path: {}, recording_count: {} }}", r.file_path,
                          r.recording_count);
  }
};

template <typename T> struct fmt::formatter<std::optional<T>> : fmt::formatter<std::string_view> {
  auto format(const std::optional<T> &o, fmt::format_context &ctx) const {
    if (!o.has_value())
      return fmt::format_to(ctx.out(), "None");
    return fmt::format_to(ctx.out(), "{}", *o);
  }
};
