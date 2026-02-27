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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

namespace holonp {

struct CopySettings {};

void to_json(nlohmann::json &j, const CopySettings &s);
void from_json(const nlohmann::json &j, CopySettings &s);

class Copy : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Copy(size_t ndim, size_t total_elems, size_t elem_size,
       curaii::unique_device_ptr<std::int64_t> d_src_strides,
       curaii::unique_device_ptr<std::int64_t> d_src_shape, cudaStream_t stream);
  friend class CopyFactory;

  size_t ndim_        = 0;
  size_t total_elems_ = 0;
  size_t elem_size_   = 0;

  cudaStream_t stream_ = static_cast<cudaStream_t>(0);

  curaii::unique_device_ptr<std::int64_t> d_src_strides_;
  curaii::unique_device_ptr<std::int64_t> d_src_shape_;
};

class CopyFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
