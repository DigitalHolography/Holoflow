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

#include <cuComplex.h>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"
#include "holonp/fft_common.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

struct FFT2Settings {
  // NumPy-like axes selection (default: last two axes).
  std::vector<int> axes;
  FftNorm          norm = FftNorm::Backward;
};

void to_json(nlohmann::json &j, const FFT2Settings &s);
void from_json(const nlohmann::json &j, FFT2Settings &s);

class FFT2 : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  FFT2(const FFT2Settings &settings, curaii::CufftHandle &&plan, size_t total_elems, size_t n_fft,
       holoflow::core::DType input_dtype, cudaStream_t stream, DevPtr<cuFloatComplex> d_tmp);

  friend class FFT2Factory;

  FFT2Settings           settings_;
  curaii::CufftHandle    plan_;
  size_t                 total_elems_;
  size_t                 n_fft_;
  holoflow::core::DType  input_dtype_;
  cudaStream_t           stream_;
  DevPtr<cuFloatComplex> d_tmp_;
};

class FFT2Factory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
