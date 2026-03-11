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

#include <array>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

struct MulSettings {
  bool operator==(const MulSettings &) const { return true; } // Currently empty, so always true
};

void to_json(nlohmann::json &j, const MulSettings &s);
void from_json(const nlohmann::json &j, MulSettings &s);

class Mul : public holoflow::core::ISyncTask {
public:
  Mul(const MulSettings &settings, std::array<holoflow::core::TDesc, 2> idescs, cudaStream_t stream,
      holoflow::core::DType dtype_a, holoflow::core::DType dtype_b, size_t total_out, size_t ndim,
      DevPtr<size_t> d_out_shape, DevPtr<size_t> d_a_strides, DevPtr<size_t> d_b_strides);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const std::array<holoflow::core::TDesc, 2> &get_idescs() const { return idescs_; }
  const MulSettings                          &get_settings() const { return settings_; }
  void update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  MulSettings                          settings_;
  std::array<holoflow::core::TDesc, 2> idescs_;
  cudaStream_t                         stream_;
  holoflow::core::DType                dtype_a_;
  holoflow::core::DType                dtype_b_;
  size_t                               total_out_;
  size_t                               ndim_;

  DevPtr<size_t> d_out_shape_;
  DevPtr<size_t> d_a_strides_;
  DevPtr<size_t> d_b_strides_;
};

class MulFactory : public holoflow::core::ISyncTaskFactory {
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