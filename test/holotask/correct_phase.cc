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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

#include "holoflow/core/tensor.hh"
#include "holotask/syncs/correct_phase.hh"

using namespace holoflow::core;
using namespace holotask::syncs;

namespace {

constexpr float kPi     = 3.14159265358979323846f;
constexpr float kHalfPi = 0.5f * kPi;

void cuda_check(cudaError_t err, const char *msg = nullptr) {
  if (err != cudaSuccess) {
    if (msg != nullptr) {
      fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    } else {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    std::abort();
  }
}

template <typename T> void upload(Tensor &tensor, const std::vector<T> &host_values) {
  cuda_check(cudaMemcpy(tensor.data(), host_values.data(), host_values.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H->D");
}

template <typename T> std::vector<T> download(const Tensor &tensor, size_t count) {
  std::vector<T> host_values(count);
  cuda_check(
      cudaMemcpy(host_values.data(), tensor.data(), count * sizeof(T), cudaMemcpyDeviceToHost),
      "cudaMemcpy D->H");
  return host_values;
}

} // namespace

TEST(CorrectPhaseFactory, InferBroadcastsSingleMaskImage) {
  CorrectPhaseFactory factory;

  const std::array<TDesc, 2> inputs{
      TDesc({2, 3, 4}, DType::CF32, MemLoc::Device),
      TDesc({1, 3, 4}, DType::F32, MemLoc::Device),
  };

  const auto infer = factory.infer(inputs, nlohmann::json(CorrectPhaseSettings{}));

  ASSERT_EQ(infer.output_descs.size(), 1u);
  EXPECT_EQ(infer.output_descs[0].shape, inputs[0].shape);
  EXPECT_EQ(infer.output_descs[0].dtype, DType::CF32);
}

TEST(CorrectPhase, SubtractsMaskFromFloatPhaseInput) {
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  const std::vector<float> input_values{
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };
  const std::vector<float> mask_values{
      0.5f,
      1.0f,
      1.5f,
      2.0f,
  };

  Tensor input(TDesc({2, 2, 2}, DType::F32, MemLoc::Device));
  Tensor mask(TDesc({1, 2, 2}, DType::F32, MemLoc::Device));
  Tensor output(TDesc({2, 2, 2}, DType::F32, MemLoc::Device));

  upload(input, input_values);
  upload(mask, mask_values);

  CorrectPhaseFactory        factory;
  const std::array<TDesc, 2> input_descs{input.desc(), mask.desc()};
  auto task = factory.create(input_descs, nlohmann::json(CorrectPhaseSettings{}),
                             holoflow::core::SyncCreateCtx{.stream = stream});

  std::array<TView, 2> inputs{input.view(), mask.view()};
  std::array<TView, 1> outputs{output.view()};
  std::atomic<bool>    cancelled = false;
  SyncCtx              ctx{
                   .inputs       = std::span<TView>(inputs),
                   .outputs      = std::span<TView>(outputs),
                   .cancelled    = &cancelled,
                   .event_writer = nullptr,
                   .event_reader = nullptr,
  };

  EXPECT_EQ(task->execute(ctx), OpResult::Ok);
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  const auto               host_output = download<float>(output, input_values.size());
  const std::vector<float> expected{
      0.5f, 1.0f, 1.5f, 2.0f, 4.5f, 5.0f, 5.5f, 6.0f,
  };

  ASSERT_EQ(host_output.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(host_output[i], expected[i]) << "Mismatch at index " << i;
  }

  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

TEST(CorrectPhase, RotatesComplexInputByNegativeMaskPhase) {
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  const std::vector<cuFloatComplex> input_values{
      make_cuFloatComplex(1.0f, 0.0f),
      make_cuFloatComplex(0.0f, 2.0f),
      make_cuFloatComplex(-3.0f, 0.0f),
      make_cuFloatComplex(0.0f, -4.0f),
  };
  const std::vector<float> mask_values{
      0.0f,
      kHalfPi,
      kPi,
      -kHalfPi,
  };

  Tensor input(TDesc({1, 2, 2}, DType::CF32, MemLoc::Device));
  Tensor mask(TDesc({2, 2}, DType::F32, MemLoc::Device));
  Tensor output(TDesc({1, 2, 2}, DType::CF32, MemLoc::Device));

  upload(input, input_values);
  upload(mask, mask_values);

  CorrectPhaseFactory        factory;
  const std::array<TDesc, 2> input_descs{input.desc(), mask.desc()};
  auto task = factory.create(input_descs, nlohmann::json(CorrectPhaseSettings{}),
                             holoflow::core::SyncCreateCtx{.stream = stream});

  std::array<TView, 2> inputs{input.view(), mask.view()};
  std::array<TView, 1> outputs{output.view()};
  std::atomic<bool>    cancelled = false;
  SyncCtx              ctx{
                   .inputs       = std::span<TView>(inputs),
                   .outputs      = std::span<TView>(outputs),
                   .cancelled    = &cancelled,
                   .event_writer = nullptr,
                   .event_reader = nullptr,
  };

  EXPECT_EQ(task->execute(ctx), OpResult::Ok);
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  const auto host_output = download<cuFloatComplex>(output, input_values.size());
  const std::vector<cuFloatComplex> expected{
      make_cuFloatComplex(1.0f, 0.0f),
      make_cuFloatComplex(2.0f, 0.0f),
      make_cuFloatComplex(3.0f, 0.0f),
      make_cuFloatComplex(4.0f, 0.0f),
  };

  ASSERT_EQ(host_output.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(host_output[i].x, expected[i].x, 1e-5f) << "Real mismatch at index " << i;
    EXPECT_NEAR(host_output[i].y, expected[i].y, 1e-5f) << "Imag mismatch at index " << i;
  }

  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
