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
//
#pragma once

#include <nlohmann/json.hpp>
#include <span>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"
#include "holonp/fft_common.hh"

namespace holonp {

struct RFFTSettings {
  // NumPy-like axis selection (default: last axis).
  int     axis = -1;
  FftNorm norm = FftNorm::Backward;
};

void to_json(nlohmann::json &j, const RFFTSettings &s);
void from_json(const nlohmann::json &j, RFFTSettings &s);

class RFFT : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  RFFT(const RFFTSettings &settings, curaii::CufftHandle &&plan, size_t total_out, size_t n_fft,
       size_t exec_count, size_t exec_in_stride, size_t exec_out_stride, cudaStream_t stream);
  friend class RFFTFactory;

  RFFTSettings        settings_;
  curaii::CufftHandle plan_;
  size_t              total_out_;
  size_t              n_fft_;
  size_t              exec_count_;
  size_t              exec_in_stride_;
  size_t              exec_out_stride_;
  cudaStream_t        stream_;
};

class RFFTFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
