// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/normalize.hh"
#include "tensor_test_buffer.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::TDesc;

std::vector<std::byte> as_bytes(const std::vector<float> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<float> from_bytes(const std::vector<std::byte> &bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<float> normalize_rows(const std::vector<float> &input, size_t rows, size_t cols,
                                  float lo, float hi) {
  std::vector<float> output(input.size());
  for (size_t row = 0; row < rows; ++row) {
    const auto begin    = input.begin() + static_cast<std::ptrdiff_t>(row * cols);
    const auto end      = begin + static_cast<std::ptrdiff_t>(cols);
    const auto [mn, mx] = std::minmax_element(begin, end);
    for (size_t col = 0; col < cols; ++col) {
      output[row * cols + col] =
          ((input[row * cols + col] - *mn) / ((*mx - *mn) + 1e-7f)) * (hi - lo) + lo;
    }
  }
  return output;
}

void expect_near(const holonp_test::TensorTestBuffer &buffer, const std::vector<float> &expected) {
  const auto actual = from_bytes(buffer.download());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
  }
}

struct NormalizeFixture {
  TDesc                              desc{{3, 5}, DType::F32, MemLoc::Device};
  holotask::syncs::NormalizeSettings settings{.axes = {1}, .lo = -1.0f, .hi = 2.0f};
  holotask::syncs::NormalizeFactory  factory;
};

TEST(NormalizeCudaGraph, CachesRotatingAddressesAndUpdatedStream) {
  NormalizeFixture   fixture;
  const auto         inference = fixture.factory.infer(std::array{fixture.desc}, fixture.settings);
  curaii::CudaStream stream_a;
  curaii::CudaStream stream_b;
  auto               task = fixture.factory.create(std::array{fixture.desc}, fixture.settings,
                                                   {.stream = stream_a.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input_a(fixture.desc);
  holonp_test::TensorTestBuffer input_b(fixture.desc);
  holonp_test::TensorTestBuffer output_a(inference.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(inference.output_descs[0]);
  std::array                    input_views_a{input_a.view()};
  std::array                    input_views_b{input_b.view()};
  std::array                    output_views_a{output_a.view()};
  std::array                    output_views_b{output_b.view()};
  std::atomic<bool>             cancelled{false};
  holoflow::core::SyncCtx       ctx_a{input_views_a, output_views_a, &cancelled, nullptr, nullptr};
  holoflow::core::SyncCtx       ctx_b{input_views_b, output_views_b, &cancelled, nullptr, nullptr};

  std::vector<float> first(fixture.desc.num_elements());
  std::vector<float> second(fixture.desc.num_elements());
  for (size_t i = 0; i < first.size(); ++i) {
    first[i]  = static_cast<float>((i * 7 + 3) % 19) - 6.0f;
    second[i] = static_cast<float>((i * 11 + 5) % 23) - 9.0f;
  }

  input_a.upload(as_bytes(first));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_near(output_a, normalize_rows(first, 3, 5, -1.0f, 2.0f));

  input_b.upload(as_bytes(second));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_near(output_b, normalize_rows(second, 3, 5, -1.0f, 2.0f));

  input_a.upload(as_bytes(second));
  ASSERT_EQ(task->execute(ctx_a), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_a.get()));
  expect_near(output_a, normalize_rows(second, 3, 5, -1.0f, 2.0f));

  task = fixture.factory.update(std::move(task), std::array{fixture.desc}, fixture.settings,
                                {.stream = stream_b.get()});
  input_b.upload(as_bytes(first));
  ASSERT_EQ(task->execute(ctx_b), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream_b.get()));
  expect_near(output_b, normalize_rows(first, 3, 5, -1.0f, 2.0f));
}

TEST(NormalizeCudaGraph, ParticipatesInOuterCapture) {
  NormalizeFixture   fixture;
  const auto         inference = fixture.factory.infer(std::array{fixture.desc}, fixture.settings);
  curaii::CudaStream stream;
  auto               task =
      fixture.factory.create(std::array{fixture.desc}, fixture.settings, {.stream = stream.get()});
  holonp_test::TensorTestBuffer input(fixture.desc);
  holonp_test::TensorTestBuffer output(inference.output_descs[0]);
  std::vector<float>            values(fixture.desc.num_elements());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((i * 13 + 1) % 29) - 10.0f;
  }
  input.upload(as_bytes(values));
  std::array              input_views{input.view()};
  std::array              output_views{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};

  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_near(output, normalize_rows(values, 3, 5, -1.0f, 2.0f));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

TEST(NormalizeCudaGraph, EvictsLeastRecentlyUsedAddressPair) {
  NormalizeFixture   fixture;
  const auto         inference = fixture.factory.infer(std::array{fixture.desc}, fixture.settings);
  curaii::CudaStream stream;
  auto               task =
      fixture.factory.create(std::array{fixture.desc}, fixture.settings, {.stream = stream.get()});
  task->bind_logger(spdlog::default_logger());

  constexpr size_t                                            pair_count = 129;
  std::vector<std::unique_ptr<holonp_test::TensorTestBuffer>> inputs;
  std::vector<std::unique_ptr<holonp_test::TensorTestBuffer>> outputs;
  inputs.reserve(pair_count);
  outputs.reserve(pair_count);
  std::atomic<bool> cancelled{false};

  for (size_t pair = 0; pair < pair_count; ++pair) {
    inputs.push_back(std::make_unique<holonp_test::TensorTestBuffer>(fixture.desc));
    outputs.push_back(std::make_unique<holonp_test::TensorTestBuffer>(inference.output_descs[0]));
    std::vector<float> values(fixture.desc.num_elements());
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>((i * 7 + pair * 3) % 31) - 11.0f;
    }
    inputs.back()->upload(as_bytes(values));
    std::array              input_views{inputs.back()->view()};
    std::array              output_views{outputs.back()->view()};
    holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};
    ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  std::vector<float> replacement(fixture.desc.num_elements());
  for (size_t i = 0; i < replacement.size(); ++i) {
    replacement[i] = static_cast<float>((i * 17 + 4) % 37) - 15.0f;
  }
  inputs.front()->upload(as_bytes(replacement));
  std::array              input_views{inputs.front()->view()};
  std::array              output_views{outputs.front()->view()};
  holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};
  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_near(*outputs.front(), normalize_rows(replacement, 3, 5, -1.0f, 2.0f));
}

TEST(NormalizeCudaGraph, FallsBackOnDefaultStream) {
  NormalizeFixture fixture;
  const auto       inference = fixture.factory.infer(std::array{fixture.desc}, fixture.settings);
  auto             task =
      fixture.factory.create(std::array{fixture.desc}, fixture.settings, {.stream = nullptr});
  holonp_test::TensorTestBuffer input(fixture.desc);
  holonp_test::TensorTestBuffer output(inference.output_descs[0]);
  const std::vector<float>      values{1, 4, 2, 8, 3, 9, 5, 7, 6, 10, -2, 0, 2, 4, 6};
  input.upload(as_bytes(values));
  std::array              input_views{input.view()};
  std::array              output_views{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{input_views, output_views, &cancelled, nullptr, nullptr};

  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  expect_near(output, normalize_rows(values, 3, 5, -1.0f, 2.0f));
}

} // namespace
