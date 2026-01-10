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
#include <span>

#include "holoflow/core/tasks.hh"

namespace holonp {

struct ArangeSettings {
  double                                start  = 0.0;
  double                                stop   = 1.0;
  double                                step   = 1.0;
  std::optional<holoflow::core::DType>  dtype  = std::nullopt;
  std::optional<holoflow::core::MemLoc> device = std::nullopt;
};

void to_json(nlohmann::json &j, const ArangeSettings &s);
void from_json(const nlohmann::json &j, ArangeSettings &s);

class Arange : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Arange(const ArangeSettings &settings, cudaStream_t stream);
  friend class ArangeFactory;

  ArangeSettings settings_;
  cudaStream_t   stream_;
};

class ArangeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp