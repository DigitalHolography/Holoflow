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
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

struct ZernikeSettings {
  std::vector<int> indexes;
  float            lambda; ///< Wavelength in meters.
  float            dx;     ///< Pixel pitch in meters.
  float            dy;     ///< Pixel pitch in meters.
  float            z;      ///< Propagation distance in meters.
  size_t           ny = 1; ///< Number of grid regions along Y (default 1).
  size_t           nx = 1; ///< Number of grid regions along X (default 1).
};

void to_json(nlohmann::json &j, const ZernikeSettings &s);
void from_json(const nlohmann::json &j, ZernikeSettings &s);

class Zernike : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  explicit Zernike(const ZernikeSettings &settings, cudaStream_t stream);

  friend class ZernikeFactory;

  ZernikeSettings settings_;
  cudaStream_t    stream_;
};

class ZernikeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs