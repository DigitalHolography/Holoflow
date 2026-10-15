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

#include "pipeline/settings.hh"
#include <nlohmann/json.hpp>

namespace holovibes::pipeline {

// -------------------------------------------------------------------------------------------------
// MTD settings serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const MTDReconstructionSettings &s) {
  j                 = {{"dimension", s.dimension}, {"iterations", s.iterations}};
  j["medium_index"] = s.medium_index ? nlohmann::json(*s.medium_index) : nlohmann::json(nullptr);
}
void from_json(const nlohmann::json &j, MTDReconstructionSettings &s) {
  s.dimension  = j.value("dimension", 512);
  s.iterations = j.value("iterations", 10);
  s.medium_index.reset();
  if (j.contains("medium_index") && !j.at("medium_index").is_null())
    s.medium_index = j.at("medium_index").get<double>();
}
void to_json(nlohmann::json &j, const MTDViewSettings &s) {
  j = {{"x", s.x},
       {"y", s.y},
       {"z", s.z},
       {"component", s.component},
       {"minimum", s.minimum},
       {"maximum", s.maximum}};
}
void from_json(const nlohmann::json &j, MTDViewSettings &s) {
  s.x         = j.value("x", 256);
  s.y         = j.value("y", 256);
  s.z         = j.value("z", 280);
  s.component = j.value("component", std::string("real"));
  s.minimum   = j.value("minimum", 0.0);
  s.maximum   = j.value("maximum", 0.06);
}
} // namespace holovibes::pipeline
