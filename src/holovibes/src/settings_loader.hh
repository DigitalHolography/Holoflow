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

namespace holovibes::pipeline {

/**
 * @brief Convert new Settings structure to the legacy JSON format.
 */
nlohmann::json settings_to_old_json(const Settings &settings);

/**
 * @brief Utility function to safely get a value from JSON with a default fallback.
 */
template <typename T> inline T val(const nlohmann::json &j, const char *key, const T &def) {
  return j.contains(key) ? j.at(key).get<T>() : def;
}

inline std::string val(const nlohmann::json &j, const char *key, const char *def) {
  return j.contains(key) ? j.at(key).get<std::string>() : std::string(def);
}

/**
 * @brief Convert legacy JSON format to the new Settings structure.
 * @param j The input JSON (old format)
 * @param default_settings A reference Settings object used for missing defaults
 */
Settings old_json_to_settings(const nlohmann::json &j, const Settings &default_settings);

} // namespace holovibes::pipeline
