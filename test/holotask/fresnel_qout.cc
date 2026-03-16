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
#include "holotask/sources/fresnel_qout.hh"

using namespace holoflow::core;
using namespace holotask::sources;

namespace {

constexpr float kPi = 3.14159265358979323846f;

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

float quadratic_r2(size_t row, size_t col, size_t width, size_t height, float dx, float dy) {
  int size     = static_cast<int>(width > height ? width : height);
  int offset_x = (size - static_cast<int>(width)) / 2;
  int offset_y = (size - static_cast<int>(height)) / 2;

  float x = (static_cast<float>(static_cast<int>(col) + offset_x) - size / 2.0f) * dx;
  float y = (static_cast<float>(static_cast<int>(row) + offset_y) - size / 2.0f) * dy;
  return x * x + y * y;
}

cuFloatComplex expected_qout(size_t row, size_t col, size_t width, size_t height, float lambda,
                             float dx, float dy, float z) {
  float r2          = quadratic_r2(row, col, width, height, dx, dy);
  float k           = 2.0f * kPi / lambda;
  float total_phase = k * z + kPi * r2 / (lambda * z);
  float amplitude   = 1.0f / (lambda * z);
  return make_cuFloatComplex(amplitude * std::sin(total_phase),
                             -amplitude * std::cos(total_phase));
}

} // namespace

TEST(FresnelQoutFactory, InferReturnsComplexDeviceMask) {
  FresnelQoutFactory factory;

  const std::array<TDesc, 1> inputs{
      TDesc({1}, DType::F32, MemLoc::Device),
  };

  const auto infer =
      factory.infer(inputs, nlohmann::json(FresnelQoutSettings{.lambda = 1.0f,
                                                               .dx     = 1.0f,
                                                               .dy     = 1.0f,
                                                               .nx     = 4,
                                                               .ny     = 3}));

  ASSERT_EQ(infer.output_descs.size(), 1u);
  EXPECT_EQ(infer.output_descs[0].shape, (std::vector<size_t>{3, 4}));
  EXPECT_EQ(infer.output_descs[0].dtype, DType::CF32);
  EXPECT_EQ(infer.output_descs[0].mem_loc, MemLoc::Device);
}

TEST(FresnelQout, GeneratesQuadraticPhaseShiftWithPrefactor) {
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

  constexpr float lambda = 2.0f;
  constexpr float dx     = 0.5f;
  constexpr float dy     = 0.5f;
  constexpr float z      = 0.25f;
  constexpr size_t nx    = 2;
  constexpr size_t ny    = 2;

  Tensor z_tensor(TDesc({1}, DType::F32, MemLoc::Device));
  Tensor output(TDesc({ny, nx}, DType::CF32, MemLoc::Device));

  upload(z_tensor, std::vector<float>{z});

  FresnelQoutFactory factory;
  const std::array<TDesc, 1> input_descs{z_tensor.desc()};
  auto task = factory.create(input_descs, nlohmann::json(FresnelQoutSettings{
                                           .lambda = lambda,
                                           .dx     = dx,
                                           .dy     = dy,
                                           .nx     = nx,
                                           .ny     = ny,
                                       }),
                             holoflow::core::SyncCreateCtx{.stream = stream});

  std::array<TView, 1> inputs{z_tensor.view()};
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

  const auto host_output = download<cuFloatComplex>(output, nx * ny);
  ASSERT_EQ(host_output.size(), nx * ny);

  for (size_t row = 0; row < ny; ++row) {
    for (size_t col = 0; col < nx; ++col) {
      const size_t         idx      = row * nx + col;
      const cuFloatComplex expected = expected_qout(row, col, nx, ny, lambda, dx, dy, z);
      EXPECT_NEAR(host_output[idx].x, expected.x, 1e-5f)
          << "Real mismatch at (" << row << ", " << col << ")";
      EXPECT_NEAR(host_output[idx].y, expected.y, 1e-5f)
          << "Imag mismatch at (" << row << ", " << col << ")";
    }
  }

  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
