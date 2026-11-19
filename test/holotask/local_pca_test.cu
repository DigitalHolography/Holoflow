// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <atomic>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

#include "holotask/syncs/local_pca.hh"
#include "holotask/syncs/holovibe_local_pca.hh"
#include "holotask/syncs/pca.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

std::vector<float> run(int stride, int end, bool automatic = false,
                       holotask::syncs::LocalPcaMap map = holotask::syncs::LocalPcaMap::Energy) {
  constexpr int width = 5, height = 3, frames = 3;
  std::vector<float> values(width * height * frames);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const int pixel = y * width + x;
      values[pixel] = static_cast<float>(x + y);
      values[width * height + pixel] = static_cast<float>(x * x + y);
    }
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  holotask::syncs::LocalPcaFactory factory;
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{TDesc({frames, height, width}, DType::F32, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{bytes},
      holotask::syncs::LocalPcaSettings{.window_size = 3, .stride = stride,
                                       .begin = 0, .end = end,
                                       .automatic = automatic, .map = map});
  std::vector<float> output(result.output_bytes[0].size() / sizeof(float));
  std::memcpy(output.data(), result.output_bytes[0].data(), result.output_bytes[0].size());
  return output;
}

} // namespace

TEST(LocalPcaTest, WrapsBorderAndSelectsLowEnergyModes) {
  // At (0,0), the wrapped x coordinates are 4,0,1 and y coordinates are 2,0,1.
  double a = 0, b = 0, c = 0;
  double mean_a = 0, mean_b = 0;
  for (int y : {2, 0, 1})
    for (int x : {4, 0, 1}) {
      mean_a += x + y;
      mean_b += x * x + y;
    }
  mean_a /= 9;
  mean_b /= 9;
  for (int y : {2, 0, 1})
    for (int x : {4, 0, 1}) {
      const double va = x + y - mean_a;
      const double vb = x * x + y - mean_b;
      a += va * va;
      b += va * vb;
      c += vb * vb;
    }
  const double trace = a + c;
  const double smaller = (trace - std::sqrt((a - c) * (a - c) + 4 * b * b)) / 2;
  const auto filtered = run(1, 2);
  ASSERT_EQ(filtered.size(), 15);
  EXPECT_NEAR(filtered[0], smaller / trace, 1e-4);
  EXPECT_GT(filtered[0], 0.0f);
  const auto automatic = run(1, 1, true);
  EXPECT_NEAR(automatic[0], smaller / trace, 1e-4);
  const auto full = run(1, 3);
  ASSERT_EQ(full.size(), 15);
  for (float value : full) EXPECT_NEAR(value, 1.0f, 1e-4);
}

TEST(LocalPcaTest, SupportsMaximumFrameDepth) {
  constexpr int frames = 32, side = 3;
  std::vector<float> values(frames * side * side);
  for (int t = 0; t < frames; ++t)
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x)
        values[t * side * side + y * side + x] =
            static_cast<float>((t + 1) * (x + 2 * y));
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  holotask::syncs::LocalPcaFactory factory;
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{TDesc({frames, side, side}, DType::F32, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{bytes},
      holotask::syncs::LocalPcaSettings{.window_size = 3, .stride = 1,
                                       .begin = 0, .end = frames - 1});
  std::vector<float> output(side * side);
  std::memcpy(output.data(), result.output_bytes[0].data(), result.output_bytes[0].size());
  for (float value : output) EXPECT_NEAR(value, 0.0f, 1e-4f);
}

TEST(LocalPcaTest, RectangularKernelWritesEveryPixelOnce) {
  constexpr int width = 4, height = 3, frames = 3;
  std::vector<float> values(width * height * frames);
  for (int t = 0; t < frames; ++t)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        values[(t * height + y) * width + x] = static_cast<float>(x * x + t * (y + 1));
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  const std::vector<TDesc> descs{TDesc({frames, height, width}, DType::F32, MemLoc::Device)};
  const holotask::syncs::LocalPcaSettings settings{
      .rectangular = true, .kernel_width = width, .kernel_height = height,
      .step_width = 2, .step_height = 1, .begin = 0, .end = frames};
  holotask::syncs::LocalPcaFactory factory;
  const auto inferred = factory.infer(descs, settings);
  EXPECT_EQ(inferred.output_descs[0].shape, (std::vector<size_t>{1, height, width}));
  const std::vector<std::vector<std::byte>> inputs{bytes};
  const auto result = holonp_test::run_sync_factory(factory, descs, inputs, settings);
  ASSERT_EQ(result.output_bytes[0].size(), static_cast<size_t>(width * height * sizeof(float)));
  std::vector<float> output(width * height);
  std::memcpy(output.data(), result.output_bytes[0].data(), result.output_bytes[0].size());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      EXPECT_NEAR(output[y * width + x], 1.0f, 1e-4f);
}

TEST(LocalPcaTest, RectangularStepMustMatchKernelParity) {
  holotask::syncs::LocalPcaFactory factory;
  const std::vector<TDesc> descs{TDesc({3, 3, 4}, DType::F32, MemLoc::Device)};
  EXPECT_THROW(factory.infer(descs, holotask::syncs::LocalPcaSettings{
      .rectangular = true, .kernel_width = 4, .kernel_height = 3,
      .step_width = 1, .step_height = 1, .begin = 0, .end = 2}),
      std::invalid_argument);
}

TEST(LocalPcaTest, FullFrameComponentsMatchGlobalPcaMagnitude) {
  constexpr int width = 4, height = 3, frames = 3;
  std::vector<float> values(width * height * frames);
  for (int t = 0; t < frames; ++t)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        values[(t * height + y) * width + x] =
            static_cast<float>(10 + x * x + (t + 1) * y + t * t * x);
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  const std::vector<TDesc> descs{TDesc({frames, height, width}, DType::F32, MemLoc::Device)};
  const std::vector<std::vector<std::byte>> inputs{bytes};
  holotask::syncs::LocalPcaFactory local_factory;
  holotask::syncs::PcaFactory global_factory;
  const auto local = holonp_test::run_sync_factory(
      local_factory, descs, inputs,
      holotask::syncs::LocalPcaSettings{
          .rectangular = true, .kernel_width = width, .kernel_height = height,
          .step_width = width, .step_height = height, .begin = 0, .end = frames,
          .map = holotask::syncs::LocalPcaMap::Components});
  const auto global = holonp_test::run_sync_factory(
      global_factory, descs, inputs, holotask::syncs::PcaSettings{0, frames});
  ASSERT_EQ(local.output_bytes[0].size(), global.output_bytes[0].size());
  for (size_t i = 0; i < values.size(); ++i) {
    float a = 0.0f, b = 0.0f;
    std::memcpy(&a, local.output_bytes[0].data() + i * sizeof(float), sizeof(float));
    std::memcpy(&b, global.output_bytes[0].data() + i * sizeof(float), sizeof(float));
    EXPECT_NEAR(std::abs(a), std::abs(b), 2e-2f);
  }
}

TEST(LocalPcaTest, DynamicMapSubtractsLocalStaticContrast) {
  using holotask::syncs::LocalPcaMap;
  const auto original = run(1, 2, true, LocalPcaMap::Original);
  const auto static_map = run(1, 2, true, LocalPcaMap::Static);
  const auto dynamic = run(1, 2, true, LocalPcaMap::Dynamic);
  ASSERT_EQ(original.size(), 15);
  ASSERT_EQ(static_map.size(), 15);
  ASSERT_EQ(dynamic.size(), 15);
  bool changed = false;
  for (size_t i = 0; i < dynamic.size(); ++i) {
    EXPECT_NEAR(dynamic[i], original[i] - static_map[i], 1e-4f);
    changed |= std::abs(dynamic[i]) > 1e-4f;
  }
  EXPECT_TRUE(changed);
}

TEST(LocalPcaTest, HonorsCancellationBeforeSubmittingPatches) {
  holotask::syncs::LocalPcaFactory factory;
  const std::vector<TDesc> input_descs{TDesc({2, 3, 3}, DType::F32, MemLoc::Device)};
  const holotask::syncs::LocalPcaSettings settings{
      .window_size = 3, .stride = 1, .begin = 0, .end = 1};
  const auto inferred = factory.infer(input_descs, settings);
  auto task = factory.create(input_descs, settings, {});
  task->bind_logger(spdlog::default_logger());
  holonp_test::TensorTestBuffer input(input_descs[0]);
  holonp_test::TensorTestBuffer output(inferred.output_descs[0]);
  std::array inputs{input.view()};
  std::array outputs{output.view()};
  std::atomic<bool> cancelled{true};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};
  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Cancelled);
}

TEST(HolovibeLocalPcaTest, RankFactorZeroCopiesComplexFramesAndOneRemovesRankOneSignal) {
  std::vector<float2> input(2 * 2 * 2);
  for (int t = 0; t < 2; ++t)
    for (int p = 0; p < 4; ++p)
      input[t * 4 + p] = make_float2(static_cast<float>((t + 1) * (p + 1)),
                                     static_cast<float>((t + 1) * (p + 1) * 0.5));
  std::vector<std::byte> bytes(input.size() * sizeof(float2));
  std::memcpy(bytes.data(), input.data(), bytes.size());
  holotask::syncs::HolovibeLocalPcaFactory factory;
  const std::vector<TDesc> desc{TDesc({2, 2, 2}, DType::CF32, MemLoc::Device)};
  for (float factor : {0.f, 1.f}) {
    const holotask::syncs::HolovibeLocalPcaSettings settings{
        .width_x = 2, .width_y = 2, .stride_x = 1, .stride_y = 1,
        .rank_factor = factor};
    const auto result = holonp_test::run_sync_factory(
        factory, desc, std::vector<std::vector<std::byte>>{bytes}, settings);
    ASSERT_EQ(result.output_bytes[0].size(), bytes.size());
    std::vector<float2> output(input.size());
    std::memcpy(output.data(), result.output_bytes[0].data(), bytes.size());
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_NEAR(output[i].x, factor == 0.f ? input[i].x : 0.f, 1e-3f);
      EXPECT_NEAR(output[i].y, factor == 0.f ? input[i].y : 0.f, 1e-3f);
    }
  }
}

TEST(HolovibeLocalPcaTest, RemovesRankOneSignalAcrossTilesAtThirtyTwoFrames) {
  constexpr int frames = 32, width = 4, height = 4;
  std::vector<float2> input(frames * width * height);
  for (int t = 0; t < frames; ++t)
    for (int p = 0; p < width * height; ++p)
      input[t * width * height + p] = make_float2(static_cast<float>((t + 1) * (p + 1)),
                                                  static_cast<float>((t + 1) * (p + 1) * 0.5));
  std::vector<std::byte> bytes(input.size() * sizeof(float2));
  std::memcpy(bytes.data(), input.data(), bytes.size());
  holotask::syncs::HolovibeLocalPcaFactory factory;
  const std::vector<TDesc> desc{TDesc({frames, height, width}, DType::CF32, MemLoc::Device)};
  const auto result = holonp_test::run_sync_factory(
      factory, desc, std::vector<std::vector<std::byte>>{bytes},
      holotask::syncs::HolovibeLocalPcaSettings{
          .width_x = 2, .width_y = 2, .stride_x = 2, .stride_y = 2,
          .rank_factor = 1.f});
  std::vector<float2> output(input.size());
  ASSERT_EQ(result.output_bytes[0].size(), bytes.size());
  std::memcpy(output.data(), result.output_bytes[0].data(), bytes.size());
  for (const auto &value : output) {
    EXPECT_NEAR(value.x, 0.f, 1e-2f);
    EXPECT_NEAR(value.y, 0.f, 1e-2f);
  }
}

TEST(HolovibeLocalPcaTest, BatchedTilesMatchIndependentTiles) {
  constexpr int frames = 3, width = 4, height = 4, tile_side = 2;
  std::vector<float2> input(frames * width * height);
  for (int t = 0; t < frames; ++t)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const int pixel = y * width + x;
        input[t * width * height + pixel] = make_float2(
            static_cast<float>(1 + x + t * (y + 1) + t * t * (x + 2)),
            static_cast<float>(y - x * t + t * t * (y + 1)));
      }
  const auto as_bytes = [](const std::vector<float2> &values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float2));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
  };
  holotask::syncs::HolovibeLocalPcaFactory factory;
  const holotask::syncs::HolovibeLocalPcaSettings settings{
      .width_x = tile_side, .width_y = tile_side,
      .stride_x = tile_side, .stride_y = tile_side, .rank_factor = 0.7f};
  const auto batched = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{TDesc({frames, height, width}, DType::CF32, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{as_bytes(input)}, settings);
  ASSERT_EQ(batched.output_bytes[0].size(), input.size() * sizeof(float2));

  for (int tile_y = 0; tile_y < height; tile_y += tile_side)
    for (int tile_x = 0; tile_x < width; tile_x += tile_side) {
      std::vector<float2> tile(frames * tile_side * tile_side);
      for (int t = 0; t < frames; ++t)
        for (int y = 0; y < tile_side; ++y)
          for (int x = 0; x < tile_side; ++x)
            tile[t * tile_side * tile_side + y * tile_side + x] =
                input[t * width * height + (tile_y + y) * width + tile_x + x];
      const auto independent = holonp_test::run_sync_factory(
          factory,
          std::vector<TDesc>{TDesc({frames, tile_side, tile_side}, DType::CF32, MemLoc::Device)},
          std::vector<std::vector<std::byte>>{as_bytes(tile)}, settings);
      for (int t = 0; t < frames; ++t)
        for (int y = 0; y < tile_side; ++y)
          for (int x = 0; x < tile_side; ++x) {
            const size_t src = t * tile_side * tile_side + y * tile_side + x;
            const size_t dst = t * width * height + (tile_y + y) * width + tile_x + x;
            float2 actual{}, expected{};
            std::memcpy(&actual, batched.output_bytes[0].data() + dst * sizeof(float2), sizeof(float2));
            std::memcpy(&expected, independent.output_bytes[0].data() + src * sizeof(float2), sizeof(float2));
            EXPECT_NEAR(actual.x, expected.x, 1e-3f);
            EXPECT_NEAR(actual.y, expected.y, 1e-3f);
          }
    }
}
