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
#include <optional>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holonp {

struct ReshapeSettings {
  // Target shape. One dimension can be -1 (inferred).
  std::vector<int64_t> shape;

  // std::nullopt = Auto (View if possible, else Copy)
  // true         = Force Copy
  // false        = Force View (raise error if copy needed)
  std::optional<bool> copy = std::nullopt;

  bool operator==(const ReshapeSettings &other) const {
    return shape == other.shape && copy == other.copy;
  }
};

void to_json(nlohmann::json &j, const ReshapeSettings &s);
void from_json(const nlohmann::json &j, ReshapeSettings &s);

class Reshape : public holoflow::core::ISyncTask {
public:
  // Constructor for the Copy case
  Reshape(const ReshapeSettings &settings, const holoflow::core::TDesc &idesc, size_t ndim,
          size_t total_elems, size_t elem_size, curaii::unique_device_ptr<int64_t> d_src_strides,
          curaii::unique_device_ptr<int64_t> d_src_shape, cudaStream_t stream);

  // Constructor for the View case (No-op)
  Reshape(const ReshapeSettings &settings, const holoflow::core::TDesc &idesc);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const ReshapeSettings       &get_settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  bool                  is_view_ = false;
  ReshapeSettings       settings_;
  holoflow::core::TDesc idesc_;

  // Copy-only params
  size_t       ndim_        = 0;
  size_t       total_elems_ = 0;
  size_t       elem_size_   = 0;
  cudaStream_t stream_      = nullptr;

  curaii::unique_device_ptr<int64_t> d_src_strides_;
  curaii::unique_device_ptr<int64_t> d_src_shape_;
};

class ReshapeFactory : public holoflow::core::ISyncTaskFactory {
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