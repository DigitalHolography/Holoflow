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
#include "holotask/syncs/pca_lsci.hh"
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

TEST(LocalPcaTest, FullImageKernelMatchesPcaLsciStaticMap) {
  constexpr int frames = 3, side = 3;
  std::vector<float> values(frames * side * side);
  for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x) {
      const int pixel = y * side + x;
      values[pixel] = static_cast<float>(10 + x + y);
      values[side * side + pixel] = static_cast<float>(10 + x * x + y);
      values[2 * side * side + pixel] = static_cast<float>(10 + x + y * y);
    }
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  const std::vector<TDesc> descs{TDesc({frames, side, side}, DType::F32, MemLoc::Device)};
  const std::vector<std::vector<std::byte>> inputs{bytes};
  holotask::syncs::LocalPcaFactory local_factory;
  holotask::syncs::PcaLsciFactory global_factory;
  const auto local = holonp_test::run_sync_factory(
      local_factory, descs, inputs,
      holotask::syncs::LocalPcaSettings{.window_size = side, .stride = 1,
                                       .begin = 0, .end = 2, .automatic = true,
                                       .map = holotask::syncs::LocalPcaMap::Static});
  const auto global = holonp_test::run_sync_factory(
      global_factory, descs, inputs,
      holotask::syncs::PcaLsciSettings{.window_size = side,
                                      .map = holotask::syncs::PcaLsciMap::Static});
  for (int i = 0; i < side * side; ++i) {
    float local_value = 0, global_value = 0;
    std::memcpy(&local_value, local.output_bytes[0].data() + i * sizeof(float), sizeof(float));
    std::memcpy(&global_value, global.output_bytes[0].data() + i * sizeof(float), sizeof(float));
    EXPECT_NEAR(local_value, global_value, 1e-3f);
  }
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
