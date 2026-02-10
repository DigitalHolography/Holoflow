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

struct SubSettings {};
void to_json(nlohmann::json &j, const SubSettings &s);
void from_json(const nlohmann::json &j, SubSettings &s);

class Sub : public holoflow::core::ISyncTask {
public:
  Sub(cudaStream_t stream, holoflow::core::DType dtype, size_t total_out, size_t ndim,
      DevPtr<size_t> d_out_shape, DevPtr<size_t> d_a_strides, DevPtr<size_t> d_b_strides);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  cudaStream_t          stream_;
  holoflow::core::DType dtype_;
  size_t                total_out_;
  size_t                ndim_;

  DevPtr<size_t> d_out_shape_;
  DevPtr<size_t> d_a_strides_;
  DevPtr<size_t> d_b_strides_;
};

class SubFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
