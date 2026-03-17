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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

#include "holoflow/core/tensor.hh"
#include "holotask/syncs/zernike_phase.hh"

using namespace holoflow::core;
using namespace holotask::syncs;

namespace {

float eval_expected_mode(int noll_index, float x_n, float y_n) {
  const float sqrt8 = std::sqrt(8.0f);

  switch (noll_index) {
  case 7:
    return sqrt8 * y_n * (3.0f * x_n * x_n - y_n * y_n);
  case 8:
    return sqrt8 * y_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f);
  case 9:
    return sqrt8 * x_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f);
  case 10:
    return sqrt8 * x_n * (x_n * x_n - 3.0f * y_n * y_n);
  default:
    return 0.0f;
  }
}

} // namespace

TEST(ZernikePhaseFactory, InferAcceptsThirdOrderModes) {
  ZernikePhaseFactory        factory;
  const std::array<TDesc, 1> inputs{
      TDesc({4}, DType::F32, MemLoc::Host),
  };

  const auto infer = factory.infer(inputs, nlohmann::json(ZernikePhaseSettings{
                                               .indexes = {7, 8, 9, 10},
                                               .ny      = 5,
                                               .nx      = 5,
                                           }));

  ASSERT_EQ(infer.output_descs.size(), 1u);
  EXPECT_EQ(infer.output_descs[0].shape, (std::vector<size_t>{5, 5}));
  EXPECT_EQ(infer.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(infer.output_descs[0].mem_loc, MemLoc::Host);
}

TEST(ZernikePhase, ReconstructsThirdOrderModes) {
  constexpr int ny       = 5;
  constexpr int nx       = 5;
  constexpr int sample_y = 3;
  constexpr int sample_x = 4;

  const float center_y = (static_cast<float>(ny) - 1.0f) * 0.5f;
  const float center_x = (static_cast<float>(nx) - 1.0f) * 0.5f;
  const float radius   = 0.5f * static_cast<float>(std::min(ny, nx));
  const float x_n      = (static_cast<float>(sample_x) - center_x) / radius;
  const float y_n      = (static_cast<float>(sample_y) - center_y) / radius;

  for (int noll_index : {7, 8, 9, 10}) {
    Tensor input(TDesc({1}, DType::F32, MemLoc::Host));
    Tensor output(TDesc({ny, nx}, DType::F32, MemLoc::Host));

    const float coefficient = 1.0f;
    std::memcpy(input.data(), &coefficient, sizeof(coefficient));

    ZernikePhaseFactory        factory;
    const std::array<TDesc, 1> input_descs{input.desc()};
    auto                       task = factory.create(input_descs,
                                                     nlohmann::json(ZernikePhaseSettings{
                                                         .indexes = {noll_index},
                                                         .ny      = ny,
                                                         .nx      = nx,
                               }),
                                                     SyncCreateCtx{});

    std::array<TView, 1> inputs{input.view()};
    std::array<TView, 1> outputs{output.view()};
    std::atomic<bool>    cancelled = false;
    SyncCtx              ctx{
                     .inputs       = std::span<TView>(inputs),
                     .outputs      = std::span<TView>(outputs),
                     .cancelled    = &cancelled,
                     .event_writer = nullptr,
                     .event_reader = nullptr,
    };

    ASSERT_EQ(task->execute(ctx), OpResult::Ok);

    const auto *phase    = reinterpret_cast<const float *>(output.data());
    const float actual   = phase[static_cast<size_t>(sample_y) * static_cast<size_t>(nx) +
                               static_cast<size_t>(sample_x)];
    const float expected = eval_expected_mode(noll_index, x_n, y_n);
    EXPECT_NEAR(actual, expected, 1e-5f) << "Mismatch for Noll index " << noll_index;
  }
}
