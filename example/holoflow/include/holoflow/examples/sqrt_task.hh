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

#include "holoflow/core/tasks.hh"

namespace holoflow_examples {

struct SqrtSettings {
  bool                                  inplace = false;
  std::optional<holoflow::core::MemLoc> mem_loc;
};

class SqrtTask final : public holoflow::core::ISyncTask {
public:
  SqrtTask()           = default;
  ~SqrtTask() override = default;

  [[nodiscard]] holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;
};

class SqrtTaskFactory final : public holoflow::core::ISyncTaskFactory {
public:
  [[nodiscard]] holoflow::core::InferResult
  infer(std::span<const holoflow::core::TDesc> input_descs,
        const nlohmann::json                  &jsettings) const override;

  [[nodiscard]] std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holoflow_examples
