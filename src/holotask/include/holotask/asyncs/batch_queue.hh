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
#include <span>

#include "holoflow/core/tasks.hh"

namespace holotask::asyncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct BatchQueueSettings {
  int target_capacity; // Target capacity of the queue
  int output_size;     // Size of each output batch
  int output_stride;   // Stride between batches

  bool operator==(const BatchQueueSettings &) const = default;
};

void to_json(nlohmann::json &j, const BatchQueueSettings &bqs);
void from_json(const nlohmann::json &j, BatchQueueSettings &bqs);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class BatchQueueFactory : public holoflow::core::IAsyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;
};

} // namespace holotask::asyncs
