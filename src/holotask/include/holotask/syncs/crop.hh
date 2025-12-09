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

#include <vector>
#include "holoflow/core/tasks.hh"
#include "curaii/cuda.hh"

namespace holotask::syncs {

struct CropSettings {
  std::vector<std::size_t> origin; // Point de départ du crop [dim0, dim1, dim2, ...]
  std::vector<std::size_t> shape;  // Forme résultante [new_dim0, new_dim1, new_dim2, ...]
};

void to_json(nlohmann::json& j, const CropSettings& settings);
void from_json(const nlohmann::json& j, CropSettings& settings);

class Crop : public holoflow::core::ISyncTask {
public:
  Crop(const CropSettings& settings, cudaStream_t stream);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx& ctx) override;

private:
  CropSettings settings_;
  cudaStream_t stream_;
  
  // Méthodes pour les différentes dimensions
  void crop_1d(const holoflow::core::TDesc& input_desc, 
               const holoflow::core::TDesc& output_desc,
               float* input, float* output);
  
  void crop_2d(const holoflow::core::TDesc& input_desc,
               const holoflow::core::TDesc& output_desc,
               float* input, float* output);
  
  void crop_3d(const holoflow::core::TDesc& input_desc,
               const holoflow::core::TDesc& output_desc,
               float* input, float* output);
};

class CropFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json& jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs,
         const nlohmann::json& jsettings,
         const holoflow::core::SyncCreateCtx& ctx) const override;
};

} // namespace holotask::syncs