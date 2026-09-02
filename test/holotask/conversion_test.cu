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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuComplex.h>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/conversion.hh"
#include "tensor_test_buffer.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::TDesc;
using Settings = holotask::syncs::ConversionSettings;

template <typename T> std::vector<std::byte> as_bytes(std::span<const T> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), values.size_bytes());
  return bytes;
}

template <typename T> std::vector<T> from_bytes(std::span<const std::byte> bytes) {
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

template <typename T>
void expect_output(const holonp_test::TensorTestBuffer &output, std::span<const T> expected) {
  const auto actual = from_bytes<T>(output.download());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_same_v<T, cuFloatComplex>) {
      EXPECT_FLOAT_EQ(actual[i].x, expected[i].x) << "real element " << i;
      EXPECT_FLOAT_EQ(actual[i].y, expected[i].y) << "imaginary element " << i;
    } else if constexpr (std::is_floating_point_v<T>) {
      EXPECT_NEAR(actual[i], expected[i], 1e-6f) << "element " << i;
    } else {
      EXPECT_EQ(actual[i], expected[i]) << "element " << i;
    }
  }
}

template <typename In, typename Out>
void exercise_single_kernel(DType dtype, const Settings &settings, const std::vector<In> &first,
                            const std::vector<Out> &first_expected, const std::vector<In> &second,
                            const std::vector<Out> &second_expected) {
  const TDesc                        input_desc({first.size()}, dtype, MemLoc::Device);
  holotask::syncs::ConversionFactory factory;
  const auto                         inference = factory.infer(std::array{input_desc}, settings);
  curaii::CudaStream                 stream_a;
  curaii::CudaStream                 stream_b;
  auto task = factory.create(std::array{input_desc}, settings, {.stream = stream_a.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input_a(input_desc);
  holonp_test::TensorTestBuffer input_b(input_desc);
  holonp_test::TensorTestBuffer output_a(inference.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(inference.output_descs[0]);
  std::array                    input_views_a{input_a.view()};
  std::array                    input_views_b{input_b.view()};
  std::array                    output_views_a{output_a.view()};
  std::array                    output_views_b{output_b.view()};
  std::atomic<bool>             cancelled{false};
  holoflow::core::SyncCtx       ctx_a{input_views_a, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_b{input_views_b, output_views_b, &cancelled, nullptr, nullptr};

  input_a.upload(as_bytes<In>(first));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_output<Out>(output_a, first_expected);

  input_a.upload(as_bytes<In>(second));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_output<Out>(output_a, second_expected);

  input_b.upload(as_bytes<In>(first));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_output<Out>(output_b, first_expected);

  task =
      factory.update(std::move(task), std::array{input_desc}, settings, {.stream = stream_b.get()});
  input_b.upload(as_bytes<In>(second));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_b.get()));
  expect_output<Out>(output_b, second_expected);
}

template <typename Out> std::vector<Out> scaled_reference(const std::vector<float> &input) {
  const auto [min_it, max_it] = std::minmax_element(input.begin(), input.end());
  const float      max_output = static_cast<float>(std::numeric_limits<Out>::max());
  std::vector<Out> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const float scaled = (input[i] - *min_it) * max_output / (*max_it - *min_it);
    output[i]          = static_cast<Out>(std::lround(scaled));
  }
  return output;
}

template <typename Out> void exercise_scaled(Settings::Target target) {
  const std::vector<float>           first{-7.0f, -2.0f, 1.0f, 3.0f, 9.0f, 17.0f, 25.0f};
  const std::vector<float>           second{40.0f, 20.0f, 10.0f, 5.0f, 0.0f, -10.0f, -20.0f};
  const TDesc                        input_desc({first.size()}, DType::F32, MemLoc::Device);
  const Settings                     settings{target, Settings::Strategy::Scaled};
  holotask::syncs::ConversionFactory factory;
  const auto                         inference = factory.infer(std::array{input_desc}, settings);
  curaii::CudaStream                 stream;
  auto task = factory.create(std::array{input_desc}, settings, {.stream = stream.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input_a(input_desc);
  holonp_test::TensorTestBuffer input_b(input_desc);
  holonp_test::TensorTestBuffer output_a(inference.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(inference.output_descs[0]);
  std::array                    input_views_a{input_a.view()};
  std::array                    input_views_b{input_b.view()};
  std::array                    output_views_a{output_a.view()};
  std::array                    output_views_b{output_b.view()};
  std::atomic<bool>             cancelled{false};
  holoflow::core::SyncCtx       ctx_a{input_views_a, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_b{input_views_b, output_views_b, &cancelled, nullptr, nullptr};

  input_a.upload(as_bytes<float>(first));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<Out>(output_a, scaled_reference<Out>(first));

  input_a.upload(as_bytes<float>(second));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<Out>(output_a, scaled_reference<Out>(second));

  input_b.upload(as_bytes<float>(second));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<Out>(output_b, scaled_reference<Out>(second));

  input_a.upload(as_bytes<float>(first));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<Out>(output_a, scaled_reference<Out>(first));
}

TEST(ConversionCudaGraph, ReplaysAndUpdatesAllSingleKernelConversions) {
  const std::vector<uint8_t> u8_a{0, 1, 17, 128, 255};
  const std::vector<uint8_t> u8_b{255, 9, 3, 2, 1};
  exercise_single_kernel<uint8_t, float>(
      DType::U8, {Settings::Target::F32, Settings::Strategy::Real}, u8_a,
      {0.0f, 1.0f, 17.0f, 128.0f, 255.0f}, u8_b, {255.0f, 9.0f, 3.0f, 2.0f, 1.0f});
  exercise_single_kernel<uint8_t, cuFloatComplex>(
      DType::U8, {Settings::Target::CF32, Settings::Strategy::Real}, u8_a,
      {{0.0f, 0.0f}, {1.0f, 0.0f}, {17.0f, 0.0f}, {128.0f, 0.0f}, {255.0f, 0.0f}}, u8_b,
      {{255.0f, 0.0f}, {9.0f, 0.0f}, {3.0f, 0.0f}, {2.0f, 0.0f}, {1.0f, 0.0f}});

  const std::vector<uint16_t> u16_a{0, 1, 257, 4096, 65535};
  const std::vector<uint16_t> u16_b{65535, 32, 16, 8, 4};
  exercise_single_kernel<uint16_t, cuFloatComplex>(
      DType::U16, {Settings::Target::CF32, Settings::Strategy::Real}, u16_a,
      {{0.0f, 0.0f}, {1.0f, 0.0f}, {257.0f, 0.0f}, {4096.0f, 0.0f}, {65535.0f, 0.0f}}, u16_b,
      {{65535.0f, 0.0f}, {32.0f, 0.0f}, {16.0f, 0.0f}, {8.0f, 0.0f}, {4.0f, 0.0f}});

  const std::vector<float> f32_a{-4.5f, 0.0f, 2.25f, 11.0f};
  const std::vector<float> f32_b{9.0f, -3.0f, 0.5f, 1.25f};
  exercise_single_kernel<float, cuFloatComplex>(
      DType::F32, {Settings::Target::CF32, Settings::Strategy::Real}, f32_a,
      {{-4.5f, 0.0f}, {0.0f, 0.0f}, {2.25f, 0.0f}, {11.0f, 0.0f}}, f32_b,
      {{9.0f, 0.0f}, {-3.0f, 0.0f}, {0.5f, 0.0f}, {1.25f, 0.0f}});

  const std::vector<cuFloatComplex> complex_a{{3.0f, 4.0f}, {-5.0f, 12.0f}, {0.0f, -2.0f}};
  const std::vector<cuFloatComplex> complex_b{{8.0f, 15.0f}, {1.0f, 0.0f}, {0.0f, 3.0f}};
  exercise_single_kernel<cuFloatComplex, float>(
      DType::CF32, {Settings::Target::F32, Settings::Strategy::Modulus}, complex_a,
      {5.0f, 13.0f, 2.0f}, complex_b, {17.0f, 1.0f, 3.0f});

  const float pi = std::acos(-1.0f);
  exercise_single_kernel<cuFloatComplex, float>(
      DType::CF32, {Settings::Target::F32, Settings::Strategy::Argument}, complex_a,
      {std::atan2(4.0f, 3.0f), std::atan2(12.0f, -5.0f), -pi / 2.0f}, complex_b,
      {std::atan2(15.0f, 8.0f), 0.0f, pi / 2.0f});
}

TEST(ConversionCudaGraph, CachesScaledConversionGraphsByBufferAddress) {
  exercise_scaled<uint8_t>(Settings::Target::U8);
  exercise_scaled<uint16_t>(Settings::Target::U16);
}

TEST(ConversionCudaGraph, FallsBackOnDefaultStreamAndDuringOuterCapture) {
  const std::vector<uint8_t>         input_values{2, 4, 8, 16};
  const TDesc                        input_desc({input_values.size()}, DType::U8, MemLoc::Device);
  const Settings                     settings{Settings::Target::F32, Settings::Strategy::Real};
  holotask::syncs::ConversionFactory factory;
  const auto                         inference = factory.infer(std::array{input_desc}, settings);
  holonp_test::TensorTestBuffer      input(input_desc);
  holonp_test::TensorTestBuffer      output(inference.output_descs[0]);
  input.upload(as_bytes<uint8_t>(input_values));
  std::array              inputs{input.view()};
  std::array              outputs{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};

  auto default_task = factory.create(std::array{input_desc}, settings, {.stream = nullptr});
  ASSERT_EQ(default_task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_output<float>(output, std::vector<float>{2.0f, 4.0f, 8.0f, 16.0f});

  curaii::CudaStream stream;
  auto captured_task = factory.create(std::array{input_desc}, settings, {.stream = stream.get()});
  cudaGraph_t graph  = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  ASSERT_EQ(captured_task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<float>(output, std::vector<float>{2.0f, 4.0f, 8.0f, 16.0f});
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  const std::vector<float> scaled_input{-3.0f, -1.0f, 2.0f, 9.0f};
  const TDesc              scaled_desc({scaled_input.size()}, DType::F32, MemLoc::Device);
  const Settings           scaled_settings{Settings::Target::U8, Settings::Strategy::Scaled};
  const auto scaled_inference = factory.infer(std::array{scaled_desc}, scaled_settings);
  auto       scaled_task =
      factory.create(std::array{scaled_desc}, scaled_settings, {.stream = stream.get()});
  holonp_test::TensorTestBuffer scaled_input_buffer(scaled_desc);
  holonp_test::TensorTestBuffer scaled_output_buffer(scaled_inference.output_descs[0]);
  scaled_input_buffer.upload(as_bytes<float>(scaled_input));
  std::array              scaled_inputs{scaled_input_buffer.view()};
  std::array              scaled_outputs{scaled_output_buffer.view()};
  holoflow::core::SyncCtx scaled_ctx{scaled_inputs, scaled_outputs, &cancelled, nullptr, nullptr};

  graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  ASSERT_EQ(scaled_task->execute(scaled_ctx), OpResult::Ok);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_output<uint8_t>(scaled_output_buffer, scaled_reference<uint8_t>(scaled_input));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

} // namespace
