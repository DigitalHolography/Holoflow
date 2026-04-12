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

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holonp {

// Extracts overlapping 2D sliding windows from the last two spatial dimensions.
//
// Input:  [..., H, W]
// Output: [..., ny_win, nx_win, win_h, win_w]
//
// where:
//   ny_win = (H - win_h) / stride_y + 1
//   nx_win = (W - win_w) / stride_x + 1
//
// Requires contiguous input. Call ascontiguousarray() first if needed.
struct Unfold2DSettings {
  size_t win_h;
  size_t win_w;
  size_t stride_y;
  size_t stride_x;

  bool operator==(const Unfold2DSettings &) const = default;
};

void to_json(nlohmann::json &j, const Unfold2DSettings &s);
void from_json(const nlohmann::json &j, Unfold2DSettings &s);

class Unfold2D : public holoflow::core::ISyncTask {
public:
  Unfold2D(const Unfold2DSettings &settings, const holoflow::core::TDesc &idesc, size_t batch,
           size_t H, size_t W, size_t ny_win, size_t nx_win, size_t elem_size, cudaStream_t stream);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const Unfold2DSettings      &get_settings() const { return settings_; }
  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t s) { stream_ = s; }

private:
  Unfold2DSettings      settings_;
  holoflow::core::TDesc idesc_;
  size_t                batch_;
  size_t                H_, W_;
  size_t                ny_win_, nx_win_;
  size_t                elem_size_;
  cudaStream_t          stream_;
};

class Unfold2DFactory : public holoflow::core::ISyncTaskFactory {
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
