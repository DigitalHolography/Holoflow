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

#include "holoflow/core/tasks.hh"
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

namespace holonp {

struct SumSettings {
  std::vector<int> axis;
  bool             keepdims                              = false;
  bool             operator==(const SumSettings &) const = default;
};

void to_json(nlohmann::json &, const SumSettings &);

void from_json(const nlohmann::json &, SumSettings &);

class SumFactory : public holoflow::core::ISyncTaskFactory {

public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc>,
         const nlohmann::json &, const holoflow::core::SyncCreateCtx &) const override;
};

} // namespace holonp
