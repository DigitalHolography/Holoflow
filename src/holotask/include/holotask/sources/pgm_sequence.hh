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

/// @file
/// @brief Synchronous source that reads an ordered sequence of PGM images.
/// @details
/// - **Inputs:** none
/// - **Outputs:** [batch_size, height, width] tensor of u8 or u16
/// - **Settings:** @ref holotask::sources::PgmSequenceSettings

#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::sources {

/// @brief Settings for a PGM image sequence source.
struct PgmSequenceSettings {
  std::string directory;              ///< Directory containing the PGM files.
  std::string filename_prefix = "i";  ///< Only files starting with this prefix are frames.
  int         batch_size      = 1;    ///< Number of frames in each output tensor.
  bool        repeat          = true; ///< False: emit the complete scan once, then Eof.
  int         expected_count  = 0; ///< Zero: any count; otherwise require exactly this many files.

  bool operator==(const PgmSequenceSettings &) const = default;
};

void to_json(nlohmann::json &j, const PgmSequenceSettings &settings);
void from_json(const nlohmann::json &j, PgmSequenceSettings &settings);

/// @brief Factory for ordered PGM image sequence source tasks.
class PgmSequenceFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::sources
