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

#pragma once

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holonp {

#ifndef HOLONP_FFT_NORM_DECLARED
#define HOLONP_FFT_NORM_DECLARED

enum class FftNorm {
  Backward,
  Forward,
  Ortho,
};

void to_json(nlohmann::json &j, FftNorm norm);
void from_json(const nlohmann::json &j, FftNorm &norm);

#endif

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct CrossCorrelation2Settings {
  struct Ellipse {
    float cx    = 0.5f;
    float cy    = 0.5f;
    float rx    = 0.5f;
    float ry    = 0.5f;
    float angle = 0.0f;

    bool operator==(const Ellipse &) const = default;
  };

  std::vector<int> axes;
  FftNorm          norm = FftNorm::Backward;
  Ellipse          roi;

  bool operator==(const CrossCorrelation2Settings &) const = default;
};

void to_json(nlohmann::json &j, const CrossCorrelation2Settings::Ellipse &e);
void from_json(const nlohmann::json &j, CrossCorrelation2Settings::Ellipse &e);
void to_json(nlohmann::json &j, const CrossCorrelation2Settings &s);
void from_json(const nlohmann::json &j, CrossCorrelation2Settings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class CrossCorrelation2Factory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holonp
