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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

namespace holonp {

struct ZerosSettings {
  std::vector<size_t>                   shape{};
  std::optional<holoflow::core::DType>  dtype  = std::nullopt;
  std::string                           order  = "C";
  std::optional<holoflow::core::MemLoc> device = std::nullopt;
};

void to_json(nlohmann::json &j, const ZerosSettings &s);
void from_json(const nlohmann::json &j, ZerosSettings &s);

class Zeros : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Zeros(const ZerosSettings &settings, holoflow::core::DType dtype, size_t total_elems,
        cudaStream_t stream);
  friend class ZerosFactory;

  ZerosSettings         settings_;
  holoflow::core::DType dtype_;
  size_t                total_elems_;
  cudaStream_t          stream_;
};

class ZerosFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
