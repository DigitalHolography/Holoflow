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

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

struct CrossCorrelation2Settings {
  struct Ellipse {
    float cx    = 0.5f; // Center x (0-1).
    float cy    = 0.5f; // Center y (0-1).
    float rx    = 0.5f; // Radius x (0-1).
    float ry    = 0.5f; // Radius y (0-1).
    float angle = 0.0f; // Rotation angle in degrees.

    bool operator==(const Ellipse &other) const {
      return cx == other.cx && cy == other.cy && rx == other.rx && ry == other.ry &&
             angle == other.angle;
    }
  };

  std::vector<int> axes;
  FftNorm          norm = FftNorm::Backward;
  Ellipse          roi;

  bool operator==(const CrossCorrelation2Settings &other) const {
    return axes == other.axes && norm == other.norm && roi == other.roi;
  }
};

void to_json(nlohmann::json &j, const CrossCorrelation2Settings::Ellipse &e);
void from_json(const nlohmann::json &j, CrossCorrelation2Settings::Ellipse &e);
void to_json(nlohmann::json &j, const CrossCorrelation2Settings &s);
void from_json(const nlohmann::json &j, CrossCorrelation2Settings &s);

struct TensorLayout {
  std::vector<size_t> offsets;
  size_t              inner_batches;
  long long           idist;
  int                 istride;
  int                 inembed_w;
};

class CrossCorrelation2 : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const CrossCorrelation2Settings &get_settings() const { return settings_; }
  const holoflow::core::TDesc     &get_moving_desc() const { return moving_desc_; }
  const holoflow::core::TDesc     &get_reference_desc() const { return reference_desc_; }
  void                             update_stream(cudaStream_t stream);

private:
  CrossCorrelation2(CrossCorrelation2Settings settings, holoflow::core::TDesc moving_desc,
                    holoflow::core::TDesc reference_desc, holoflow::core::TDesc moving_freq_desc,
                    curaii::CufftHandle &&moving_fwd_plan, curaii::CufftHandle &&reference_fwd_plan,
                    curaii::CufftHandle &&inverse_plan, TensorLayout moving_layout,
                    TensorLayout reference_layout, size_t h, size_t w, size_t freq_elems_per_batch,
                    size_t total_moving_freq_elems, size_t total_out_elems,
                    DevPtr<cuFloatComplex> d_moving_spatial,
                    DevPtr<cuFloatComplex> d_reference_spatial, DevPtr<float> d_moving_means,
                    DevPtr<float> d_reference_means, DevPtr<cuFloatComplex> d_moving_freq,
                    DevPtr<cuFloatComplex> d_reference_freq, DevPtr<size_t> d_reference_batch_map,
                    cudaStream_t stream);

  friend class CrossCorrelation2Factory;

  CrossCorrelation2Settings settings_;
  holoflow::core::TDesc     moving_desc_;
  holoflow::core::TDesc     reference_desc_;
  holoflow::core::TDesc     moving_freq_desc_;
  curaii::CufftHandle       moving_fwd_plan_;
  curaii::CufftHandle       reference_fwd_plan_;
  curaii::CufftHandle       inverse_plan_;

  TensorLayout moving_layout_;
  TensorLayout reference_layout_;
  size_t       h_;
  size_t       w_;

  size_t freq_elems_per_batch_;
  size_t total_moving_freq_elems_;
  size_t total_out_elems_;

  DevPtr<cuFloatComplex> d_moving_spatial_;
  DevPtr<cuFloatComplex> d_reference_spatial_;
  DevPtr<float>          d_moving_means_;
  DevPtr<float>          d_reference_means_;
  DevPtr<cuFloatComplex> d_moving_freq_;
  DevPtr<cuFloatComplex> d_reference_freq_;
  DevPtr<size_t>         d_reference_batch_map_;
  cudaStream_t           stream_;
};

class CrossCorrelation2Factory : public holoflow::core::ISyncTaskFactory {
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