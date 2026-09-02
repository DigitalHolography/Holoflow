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
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/pct_clip.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::TDesc;

std::vector<std::byte> as_bytes(std::span<const float> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), values.size_bytes());
  return bytes;
}

std::vector<float> from_bytes(std::span<const std::byte> bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<float> reference_clip(const std::vector<float> &input, size_t depth, size_t height,
                                  size_t width, const holotask::syncs::PctClipSettings &settings) {
  constexpr float     pi = 3.14159265358979323846f;
  const float         th = settings.roi.angle * (pi / 180.0f);
  const float         c  = std::cos(th);
  const float         s  = std::sin(th);
  std::vector<size_t> spatial_indices;
  for (size_t y = 0; y < height; ++y) {
    const float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    for (size_t x = 0; x < width; ++x) {
      const float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
      const float dx = xn - settings.roi.cx;
      const float dy = yn - settings.roi.cy;
      const float xr = c * dx + s * dy;
      const float yr = -s * dx + c * dy;
      if ((xr * xr) / (settings.roi.rx * settings.roi.rx) +
              (yr * yr) / (settings.roi.ry * settings.roi.ry) <=
          1.0f) {
        spatial_indices.push_back(y * width + x);
      }
    }
  }

  std::vector<float> roi_values;
  roi_values.reserve(depth * spatial_indices.size());
  const size_t plane_size = height * width;
  for (size_t z = 0; z < depth; ++z) {
    for (const size_t index : spatial_indices) {
      roi_values.push_back(input[z * plane_size + index]);
    }
  }
  std::sort(roi_values.begin(), roi_values.end());
  const size_t min_idx =
      static_cast<size_t>(settings.min_pct / 100.0f * static_cast<float>(roi_values.size() - 1));
  const size_t max_idx =
      static_cast<size_t>(settings.max_pct / 100.0f * static_cast<float>(roi_values.size() - 1));

  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = std::fmin(std::fmax(input[i], roi_values[min_idx]), roi_values[max_idx]);
  }
  return output;
}

void expect_values(const holonp_test::TensorTestBuffer &output,
                   const std::vector<float>            &expected) {
  const auto actual = from_bytes(output.download());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << "element " << i;
  }
}

TEST(PctClip, MatchesReferenceForRotatedRoiAndMultiplePlanes) {
  constexpr size_t depth  = 2;
  constexpr size_t height = 5;
  constexpr size_t width  = 7;
  const TDesc      input_desc({depth, height, width}, DType::F32, MemLoc::Device);
  const holotask::syncs::PctClipSettings settings{
      .min_pct = 20.0f,
      .max_pct = 80.0f,
      .roi     = {.cx = 0.55f, .cy = 0.45f, .rx = 0.38f, .ry = 0.25f, .angle = 31.0f},
  };
  std::vector<float> input(input_desc.num_elements());
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i * 17 + 11) % 53) - 19.0f;
  }

  holotask::syncs::PctClipFactory factory;
  const auto                      run =
      holonp_test::run_sync_factory(factory, std::array{input_desc},
                                    std::vector<std::vector<std::byte>>{as_bytes(input)}, settings);
  const auto expected = reference_clip(input, depth, height, width, settings);
  const auto actual   = from_bytes(run.output_bytes[0]);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << "element " << i;
  }
}

TEST(PctClip, HandlesConstantInputAndEndpointPercentiles) {
  const TDesc                            input_desc({1, 4, 5}, DType::F32, MemLoc::Device);
  const holotask::syncs::PctClipSettings settings{
      .min_pct = 0.0f,
      .max_pct = 100.0f,
      .roi     = {.cx = 0.5f, .cy = 0.5f, .rx = 1.0f, .ry = 1.0f, .angle = 0.0f},
  };
  const std::vector<float> input(input_desc.num_elements(), 4.25f);

  holotask::syncs::PctClipFactory factory;
  const auto                      run =
      holonp_test::run_sync_factory(factory, std::array{input_desc},
                                    std::vector<std::vector<std::byte>>{as_bytes(input)}, settings);
  for (const float value : from_bytes(run.output_bytes[0])) {
    EXPECT_FLOAT_EQ(value, 4.25f);
  }
}

TEST(PctClipCudaGraph, CachesRotatingAddressPairsAndStreams) {
  constexpr size_t depth  = 2;
  constexpr size_t height = 8;
  constexpr size_t width  = 9;
  const TDesc      input_desc({depth, height, width}, DType::F32, MemLoc::Device);
  const holotask::syncs::PctClipSettings settings{
      .min_pct = 10.0f,
      .max_pct = 90.0f,
      .roi     = {.cx = 0.5f, .cy = 0.5f, .rx = 0.45f, .ry = 0.35f, .angle = -17.0f},
  };
  holotask::syncs::PctClipFactory factory;
  const auto                      inference = factory.infer(std::array{input_desc}, settings);
  curaii::CudaStream              stream_a;
  curaii::CudaStream              stream_b;
  auto task = factory.create(std::array{input_desc}, settings, {.stream = stream_a.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input_a(input_desc);
  holonp_test::TensorTestBuffer input_b(input_desc);
  holonp_test::TensorTestBuffer input_c(input_desc);
  holonp_test::TensorTestBuffer output_a(inference.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(inference.output_descs[0]);
  std::array                    input_views_a{input_a.view()};
  std::array                    input_views_b{input_b.view()};
  std::array                    input_views_c{input_c.view()};
  std::array                    output_views_a{output_a.view()};
  std::array                    output_views_b{output_b.view()};
  std::atomic<bool>             cancelled{false};
  holoflow::core::SyncCtx       ctx_a{input_views_a, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_b{input_views_b, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_c{input_views_c, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_new_output{input_views_b, output_views_b, &cancelled, nullptr,
                                               nullptr};
  holoflow::core::SyncCtx ctx_updated_output{input_views_a, output_views_b, &cancelled, nullptr,
                                             nullptr};

  std::vector<float> frame_a(input_desc.num_elements());
  std::vector<float> frame_b(input_desc.num_elements());
  std::vector<float> frame_c(input_desc.num_elements());
  for (size_t i = 0; i < frame_a.size(); ++i) {
    frame_a[i] = static_cast<float>((i * 13 + 5) % 79) - 30.0f;
    frame_b[i] = static_cast<float>((i * 29 + 7) % 97) - 41.0f;
    frame_c[i] = static_cast<float>((i * 37 + 3) % 89) - 23.0f;
  }

  input_a.upload(as_bytes(frame_a));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_a, reference_clip(frame_a, depth, height, width, settings));

  input_b.upload(as_bytes(frame_b));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_a, reference_clip(frame_b, depth, height, width, settings));

  input_c.upload(as_bytes(frame_c));
  ASSERT_EQ(task->execute(ctx_c), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_a, reference_clip(frame_c, depth, height, width, settings));

  ASSERT_EQ(task->execute(ctx_new_output), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_b, reference_clip(frame_b, depth, height, width, settings));

  ASSERT_EQ(task->execute(ctx_updated_output), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_b, reference_clip(frame_a, depth, height, width, settings));

  input_a.upload(as_bytes(frame_c));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_values(output_a, reference_clip(frame_c, depth, height, width, settings));

  task =
      factory.update(std::move(task), std::array{input_desc}, settings, {.stream = stream_b.get()});
  input_b.upload(as_bytes(frame_b));
  ASSERT_EQ(task->execute(ctx_new_output), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_b.get()));
  expect_values(output_b, reference_clip(frame_b, depth, height, width, settings));
}

TEST(PctClipCudaGraph, EvictsLeastRecentlyUsedAddressPair) {
  constexpr size_t depth        = 1;
  constexpr size_t height       = 3;
  constexpr size_t width        = 4;
  constexpr size_t unique_pairs = 129;
  const TDesc      input_desc({depth, height, width}, DType::F32, MemLoc::Device);
  const holotask::syncs::PctClipSettings settings{
      .min_pct = 25.0f,
      .max_pct = 75.0f,
      .roi     = {.cx = 0.5f, .cy = 0.5f, .rx = 1.0f, .ry = 1.0f, .angle = 0.0f},
  };
  holotask::syncs::PctClipFactory factory;
  const auto                      inference = factory.infer(std::array{input_desc}, settings);
  curaii::CudaStream              stream;
  auto task = factory.create(std::array{input_desc}, settings, {.stream = stream.get()});
  task->bind_logger(spdlog::default_logger());

  std::vector<std::unique_ptr<holonp_test::TensorTestBuffer>> inputs;
  std::vector<std::unique_ptr<holonp_test::TensorTestBuffer>> outputs;
  inputs.reserve(unique_pairs);
  outputs.reserve(unique_pairs);
  std::atomic<bool> cancelled{false};

  for (size_t pair = 0; pair < unique_pairs; ++pair) {
    inputs.push_back(std::make_unique<holonp_test::TensorTestBuffer>(input_desc));
    outputs.push_back(std::make_unique<holonp_test::TensorTestBuffer>(inference.output_descs[0]));

    std::vector<float> frame(input_desc.num_elements());
    for (size_t i = 0; i < frame.size(); ++i) {
      frame[i] = static_cast<float>((i * 11 + pair * 7) % 61) - 20.0f;
    }
    inputs.back()->upload(as_bytes(frame));
    std::array              input_views{inputs.back()->view()};
    std::array              output_views{outputs.back()->view()};
    holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};
    ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  std::vector<float> replacement(input_desc.num_elements());
  for (size_t i = 0; i < replacement.size(); ++i) {
    replacement[i] = static_cast<float>((i * 19 + 3) % 47) - 15.0f;
  }
  inputs.front()->upload(as_bytes(replacement));
  std::array              input_views{inputs.front()->view()};
  std::array              output_views{outputs.front()->view()};
  holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};
  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_values(*outputs.front(), reference_clip(replacement, depth, height, width, settings));
}

TEST(PctClipCudaGraph, FallsBackOnDefaultStream) {
  const TDesc                            input_desc({1, 3, 4}, DType::F32, MemLoc::Device);
  const holotask::syncs::PctClipSettings settings{
      .min_pct = 25.0f,
      .max_pct = 75.0f,
      .roi     = {.cx = 0.5f, .cy = 0.5f, .rx = 0.5f, .ry = 0.5f, .angle = 0.0f},
  };
  const std::vector<float>        input{-6.0f, -3.0f, 0.0f,  3.0f,  6.0f,  9.0f,
                                        12.0f, 15.0f, 18.0f, 21.0f, 24.0f, 27.0f};
  holotask::syncs::PctClipFactory factory;
  const auto                      inference = factory.infer(std::array{input_desc}, settings);
  auto task = factory.create(std::array{input_desc}, settings, {.stream = nullptr});
  holonp_test::TensorTestBuffer input_buffer(input_desc);
  holonp_test::TensorTestBuffer output_buffer(inference.output_descs[0]);
  input_buffer.upload(as_bytes(input));
  std::array              input_views{input_buffer.view()};
  std::array              output_views{output_buffer.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};

  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_values(output_buffer, reference_clip(input, 1, 3, 4, settings));

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
  expect_values(output_buffer, reference_clip(input, 1, 3, 4, settings));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

} // namespace
