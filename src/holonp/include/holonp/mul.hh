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

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holonp {

struct MulSettings {};

void to_json(nlohmann::json &j, const MulSettings &s);
void from_json(const nlohmann::json &j, MulSettings &s);

class Mul : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Mul(const MulSettings &settings, cudaStream_t stream, size_t out_ndim, std::int64_t total_out,
      HostPtr<std::int64_t> h_out_strides, DevPtr<std::int64_t> d_out_strides,
      HostPtr<std::int64_t> h_a_strides, DevPtr<std::int64_t> d_a_strides,
      HostPtr<std::int64_t> h_b_strides, DevPtr<std::int64_t> d_b_strides);
  friend class MulFactory;

  MulSettings  settings_;
  cudaStream_t stream_;
  size_t       out_ndim_;
  std::int64_t total_out_;

  HostPtr<std::int64_t> h_out_strides_;
  DevPtr<std::int64_t>  d_out_strides_;
  HostPtr<std::int64_t> h_a_strides_;
  DevPtr<std::int64_t>  d_a_strides_;
  HostPtr<std::int64_t> h_b_strides_;
  DevPtr<std::int64_t>  d_b_strides_;
};

class MulFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holonp
