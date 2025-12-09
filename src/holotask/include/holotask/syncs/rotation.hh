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

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

struct RotationSettings {
  enum class Axis { X, Y, Z };

  size_t angle = 0;
  Axis   axis  = Axis::Z;
};

void to_json(nlohmann::json &j, const RotationSettings &s);
void from_json(const nlohmann::json &j, RotationSettings &s);

class Rotation : public holoflow::core::ISyncTask {
public:
  Rotation(const RotationSettings &settings, cudaStream_t stream);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  RotationSettings settings_;
  cudaStream_t     stream_;
};

class RotationFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs