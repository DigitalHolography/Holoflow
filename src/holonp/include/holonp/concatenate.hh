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
//
#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct ConcatenateSettings {
  // NumPy-like axis selection. null => flatten inputs before concatenation.
  std::optional<int> axis = 0;
};

void to_json(nlohmann::json &j, const ConcatenateSettings &s);
void from_json(const nlohmann::json &j, ConcatenateSettings &s);

struct ConcatenateInputPlan {
  std::int64_t total_in    = 0;
  std::int64_t axis_dim    = 0;
  std::int64_t axis_offset = 0;
};

class Concatenate : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Concatenate(const ConcatenateSettings &settings, cudaStream_t stream, std::int64_t inner,
              std::int64_t out_axis_dim, std::vector<ConcatenateInputPlan> inputs);
  friend class ConcatenateFactory;

  ConcatenateSettings               settings_;
  cudaStream_t                      stream_;
  std::int64_t                      inner_;
  std::int64_t                      out_axis_dim_;
  std::vector<ConcatenateInputPlan> inputs_;
};

class ConcatenateFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
