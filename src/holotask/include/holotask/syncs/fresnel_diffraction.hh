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
#include <string>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

struct FresnelDiffractionSettings {
  float            lambda;                  ///< Wavelength in meters.
  float            dx;                      ///< Pixel pitch in meters.
  float            dy;                      ///< Pixel pitch in meters.
  float            z;                       ///< Propagation distance in meters.
  std::vector<int> axes             = {-2, -1}; ///< Axes to perform diffraction over.
  bool             skip_phase_shift = true; ///< Omit the output-plane quadratic phase term.

  bool operator==(const FresnelDiffractionSettings &other) const {
    return lambda == other.lambda && dx == other.dx && dy == other.dy && z == other.z &&
           axes == other.axes && skip_phase_shift == other.skip_phase_shift;
  }
};

void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds);
void from_json(const nlohmann::json &j, FresnelDiffractionSettings &fds);

struct LaunchOffset {
  size_t in_bytes;
  size_t out_bytes;
};

class FresnelDiffraction : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc      &get_idesc() const { return idesc_; }
  const FresnelDiffractionSettings &get_settings() const { return settings_; }
  void                              update_stream(cudaStream_t stream);

private:
  FresnelDiffraction(const FresnelDiffractionSettings &settings, holoflow::core::TDesc idesc,
                     curaii::CufftHandle &&fft_handle, std::vector<LaunchOffset> offsets,
                     size_t inner_batch, int height, int width, long long out_idist,
                     long long out_stride_h, long long out_istride, cudaStream_t stream,
                     DevPtr<cuFloatComplex> &&d_lens,
                     DevPtr<void> &&d_caller_info,
                     std::vector<char> &&lto);

  friend class FresnelDiffractionFactory;

  FresnelDiffractionSettings settings_;
  holoflow::core::TDesc      idesc_;
  curaii::CufftHandle        fft_handle_;

  std::vector<LaunchOffset> offsets_;
  size_t                   inner_batch_;
  cudaStream_t               stream_;
  int                        height_;
  int                        width_;
  long long int              out_idist_;
  long long int              out_stride_h_;
  long long int              out_istride_;
  DevPtr<cuFloatComplex>     d_lens_;
  DevPtr<void>               d_caller_info_;
  std::vector<char>          lto_;
};

class FresnelDiffractionFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holotask::syncs
