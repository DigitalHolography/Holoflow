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
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holonp {

struct WhereSettings {};
void to_json(nlohmann::json &j, const WhereSettings &s);
void from_json(const nlohmann::json &j, WhereSettings &s);

class Where : public holoflow::core::ISyncTask {
public:
  Where(cudaStream_t stream, holoflow::core::DType out_dtype, size_t total_out, size_t ndim,
        DevPtr<size_t> d_out_shape, DevPtr<size_t> d_cond_strides, DevPtr<size_t> d_x_strides,
        DevPtr<size_t> d_y_strides);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  cudaStream_t          stream_;
  holoflow::core::DType out_dtype_;
  size_t                total_out_;
  size_t                ndim_;

  DevPtr<size_t> d_out_shape_;
  DevPtr<size_t> d_cond_strides_;
  DevPtr<size_t> d_x_strides_;
  DevPtr<size_t> d_y_strides_;
};

class WhereFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
