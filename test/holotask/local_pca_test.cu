// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "holotask/syncs/local_pca.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

TEST(LocalPcaTest, RankFactorZeroCopiesComplexFramesAndOneRemovesRankOneSignal) {
  std::vector<float2> input(2 * 2 * 2);
  for (int t = 0; t < 2; ++t)
    for (int p = 0; p < 4; ++p)
      input[t * 4 + p] = make_float2(static_cast<float>((t + 1) * (p + 1)),
                                     static_cast<float>((t + 1) * (p + 1) * 0.5));
  std::vector<std::byte> bytes(input.size() * sizeof(float2));
  std::memcpy(bytes.data(), input.data(), bytes.size());
  holotask::syncs::LocalPcaFactory factory;
  const std::vector<TDesc> desc{TDesc({2, 2, 2}, DType::CF32, MemLoc::Device)};
  for (float factor : {0.f, 1.f}) {
    const holotask::syncs::LocalPcaSettings settings{
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

TEST(LocalPcaTest, RemovesRankOneSignalAcrossTilesAtThirtyTwoFrames) {
  constexpr int frames = 32, width = 10, height = 8;
  std::vector<float2> input(frames * width * height);
  for (int t = 0; t < frames; ++t)
    for (int p = 0; p < width * height; ++p)
      input[t * width * height + p] = make_float2(static_cast<float>((t + 1) * (p + 1)),
                                                  static_cast<float>((t + 1) * (p + 1) * 0.5));
  std::vector<std::byte> bytes(input.size() * sizeof(float2));
  std::memcpy(bytes.data(), input.data(), bytes.size());
  holotask::syncs::LocalPcaFactory factory;
  const std::vector<TDesc> desc{TDesc({frames, height, width}, DType::CF32, MemLoc::Device)};
  const auto result = holonp_test::run_sync_factory(
      factory, desc, std::vector<std::vector<std::byte>>{bytes},
      holotask::syncs::LocalPcaSettings{
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

TEST(LocalPcaTest, BatchedTilesMatchIndependentTiles) {
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
  holotask::syncs::LocalPcaFactory factory;
  const holotask::syncs::LocalPcaSettings settings{
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
