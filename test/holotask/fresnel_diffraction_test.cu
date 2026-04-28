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

#include <gtest/gtest.h>

#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::SyncCreateCtx;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

nlohmann::json fresnel_settings(float z) {
  return holotask::syncs::FresnelDiffractionSettings{
      .lambda           = 852e-9f,
      .dx               = 20e-6f,
      .dy               = 20e-6f,
      .z                = z,
      .axes             = {-2, -1},
      .skip_phase_shift = false,
  };
}

nlohmann::json short_time_fresnel_settings(float z) {
  return holotask::syncs::ShortTimeFresnelDiffractionSettings{
      .lambda           = 852e-9f,
      .dx               = 20e-6f,
      .dy               = 20e-6f,
      .z                = z,
      .win_h            = 4,
      .win_w            = 4,
      .stride_y         = 2,
      .stride_x         = 2,
      .phase_ref        = holotask::syncs::STFDPhaseReference::LOCAL,
      .skip_phase_shift = false,
      .axes             = {-2, -1},
  };
}

} // namespace

TEST(FresnelDiffractionUpdateTest, ReusesTaskWhenOnlyPropagationDistanceChanges) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const std::vector<TDesc>                   input = {device_desc({4, 4}, DType::CF32)};

  curaii::CudaStream  stream;
  const SyncCreateCtx ctx{stream.get()};

  auto task = factory.create(input, fresnel_settings(0.01f), ctx);
  task->bind_logger(spdlog::default_logger());
  auto *raw = task.get();

  task = factory.update(std::move(task), input, fresnel_settings(0.02f), ctx);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  EXPECT_EQ(task.get(), raw);
}

TEST(ShortTimeFresnelDiffractionUpdateTest, ReusesTaskWhenOnlyPropagationDistanceChanges) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const std::vector<TDesc>                            input = {device_desc({8, 8}, DType::CF32)};

  curaii::CudaStream  stream;
  const SyncCreateCtx ctx{stream.get()};

  auto task = factory.create(input, short_time_fresnel_settings(0.01f), ctx);
  task->bind_logger(spdlog::default_logger());
  auto *raw = task.get();

  task = factory.update(std::move(task), input, short_time_fresnel_settings(0.02f), ctx);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  EXPECT_EQ(task.get(), raw);
}
