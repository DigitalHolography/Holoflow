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

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

/// @brief Settings for the cuda_stream_synchronize task.
/// @details No configurable options — the task always calls cudaStreamSynchronize on the
/// pipeline stream and forwards the input tensor unchanged.
struct CudaStreamSynchronizeSettings {};

/// @name JSON serialization
/// @{
void to_json(nlohmann::json &j, const CudaStreamSynchronizeSettings &s);
void from_json(const nlohmann::json &j, CudaStreamSynchronizeSettings &s);
/// @}

class CudaStreamSynchronize : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  CudaStreamSynchronize(cudaStream_t stream);

  friend class CudaStreamSynchronizeFactory;

  cudaStream_t stream_;
};

class CudaStreamSynchronizeFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
