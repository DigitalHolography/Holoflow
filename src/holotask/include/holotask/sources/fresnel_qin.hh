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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <span>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::sources {

struct FresnelQinSettings {
  float  lambda; ///< Wavelength in meters.
  float  dx;     ///< Pixel pitch in meters.
  float  dy;     ///< Pixel pitch in meters.
  size_t nx;     ///< Number of pixels in x.
  size_t ny;     ///< Number of pixels in y.

  bool operator==(const FresnelQinSettings &other) const {
    return lambda == other.lambda && dx == other.dx && dy == other.dy &&
           nx == other.nx && ny == other.ny;
  }
};

void to_json(nlohmann::json &j, const FresnelQinSettings &fqs);
void from_json(const nlohmann::json &j, FresnelQinSettings &fqs);

class FresnelQin : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc& get_idesc() const { return idesc_; }
  const FresnelQinSettings&    get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  FresnelQin(const FresnelQinSettings &settings, const holoflow::core::TDesc &idesc, 
             DevPtr<float> &&d_r2, cudaStream_t stream);

  friend class FresnelQinFactory;

  FresnelQinSettings    settings_;
  holoflow::core::TDesc idesc_;
  DevPtr<float>         d_r2_;
  cudaStream_t          stream_;
};

class FresnelQinFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc>     input_descs,
         const nlohmann::json                       &jsettings,
         const holoflow::core::SyncCreateCtx        &ctx) const override;
};

} // namespace holotask::sources