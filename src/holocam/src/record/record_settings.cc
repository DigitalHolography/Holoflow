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

#include "holocam/record_settings.hh"

namespace holocam {
bool RecordSettings::requires_rebuild(const RecordSettings &old) const {
  return recording_count > old.recording_count;
}
// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const RecordSettings &s) {
  j = nlohmann::json{{"file_path", s.file_path},
                     {"recording_count", s.recording_count},
                     {"pipeline_config", s.pipeline_config},
                     {"record_queue_count", s.record_queue_count}};
}

void from_json(const nlohmann::json &j, RecordSettings &s) {
  j.at("file_path").get_to(s.file_path);
  j.at("recording_count").get_to(s.recording_count);
  j.at("pipeline_config").get_to(s.pipeline_config);
  j.at("record_queue_count").get_to(s.record_queue_count);
}
} // namespace holotask::sources