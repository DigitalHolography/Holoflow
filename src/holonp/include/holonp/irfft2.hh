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
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"
#include "holonp/fft_common.hh"

namespace holonp {

struct IRFFT2Settings {
  std::vector<int> axes;
  FftNorm          norm = FftNorm::Backward;

  bool operator==(const IRFFT2Settings &other) const {
    return axes == other.axes && norm == other.norm;
  }
};

void to_json(nlohmann::json &j, const IRFFT2Settings &s);
void from_json(const nlohmann::json &j, IRFFT2Settings &s);

class IRFFT2 : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const IRFFT2Settings        &get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream);

private:
  IRFFT2(const IRFFT2Settings &settings, const holoflow::core::TDesc &idesc,
         curaii::CufftHandle &&plan, size_t n_fft_elems, size_t total_out_elems,
         std::vector<size_t> input_offsets, size_t output_stride_bytes, cudaStream_t stream);

  friend class IRFFT2Factory;

  IRFFT2Settings        settings_;
  holoflow::core::TDesc idesc_;
  curaii::CufftHandle   plan_;
  size_t                n_fft_elems_;
  size_t                total_out_elems_;
  std::vector<size_t>   input_offsets_;
  size_t                output_stride_bytes_;
  cudaStream_t          stream_;
};

class IRFFT2Factory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp