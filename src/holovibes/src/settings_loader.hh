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

#include "pipeline/settings.hh"

#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>

namespace holovibes::pipeline {

/**
 * @brief Convert the new Settings structure to the legacy JSON format.
 */
nlohmann::json settings_to_old_json(const Settings &settings);

/**
 * @brief Safely get a value from JSON with a default fallback.
 *
 * Returns @p def if:
 * - the key does not exist
 * - the value is null
 * - conversion to T throws
 */
template <typename T> inline T val(const nlohmann::json &j, const char *key, const T &def) {
  try {
    if (!j.is_object() || !j.contains(key) || j.at(key).is_null()) {
      return def;
    }
    return j.at(key).get<T>();
  } catch (...) {
    return def;
  }
}

inline std::string val(const nlohmann::json &j, const char *key, const char *def) {
  return val<std::string>(j, key, std::string(def));
}

/**
 * @brief Convert legacy JSON format to the new Settings structure.
 *
 * Any field that does not exist in the legacy format falls back to
 * @p default_settings.
 */
Settings old_json_to_settings(const nlohmann::json &j, const Settings &default_settings);

} // namespace holovibes::pipeline