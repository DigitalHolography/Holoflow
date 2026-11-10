// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

#include "holotask/syncs/pca_lsci.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<float> run_map(holotask::syncs::PcaLsciMap map, int window_size = 1) {
  // The centered frames have orthogonal spatial modes of energies 2 and 0.5.
  // The threshold is 1.25, so only the first mode is retained in the static reconstruction.
  const std::vector<float> frames{11.0f, 9.0f, 10.0f, 10.0f,
                                  10.0f, 10.0f, 10.5f, 9.5f};
  holotask::syncs::PcaLsciFactory factory;
  const TDesc desc({2, 2, 2}, DType::F32, MemLoc::Device);
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(frames)},
      holotask::syncs::PcaLsciSettings{.window_size = window_size, .map = map});
  std::vector<float> output(4);
  std::memcpy(output.data(), result.output_bytes[0].data(), 4 * sizeof(float));
  return output;
}

} // namespace

TEST(PcaLsciTest, SelectsModesAndSubtractsContrastMaps) {
  const auto original = run_map(holotask::syncs::PcaLsciMap::Original);
  const auto static_map = run_map(holotask::syncs::PcaLsciMap::Static);
  const auto dynamic = run_map(holotask::syncs::PcaLsciMap::Dynamic);
  ASSERT_EQ(original.size(), 4);
  EXPECT_NEAR(original[2], 0.25f / 10.25f, 1e-5f);
  EXPECT_NEAR(static_map[2], 0.0f, 1e-5f);
  EXPECT_NEAR(dynamic[2], original[2] - static_map[2], 1e-5f);
  EXPECT_NEAR(dynamic[0], 0.0f, 1e-5f);
}

TEST(PcaLsciTest, PoolsSpatialAndTemporalMomentsWithClippedBorders) {
  const auto original = run_map(holotask::syncs::PcaLsciMap::Original, 3);
  const auto static_map = run_map(holotask::syncs::PcaLsciMap::Static, 3);
  const auto dynamic = run_map(holotask::syncs::PcaLsciMap::Dynamic, 3);
  const float expected_original = std::sqrt(2.5f / 8.0f) / 10.0f;
  for (size_t p = 0; p < 4; ++p) {
    EXPECT_NEAR(original[p], expected_original, 1e-5f);
    EXPECT_NEAR(static_map[p], 0.05f, 1e-5f);
    EXPECT_NEAR(dynamic[p], original[p] - static_map[p], 1e-5f);
  }
}

TEST(PcaLsciTest, RetainsSmallU16ContrastOnHighBaseline) {
  const std::vector<uint16_t> frames{60000, 60001, 60002, 60003,
                                     60004, 60005, 60006, 60007};
  holotask::syncs::PcaLsciFactory factory;
  const TDesc desc({2, 2, 2}, DType::U16, MemLoc::Device);
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(frames)},
      holotask::syncs::PcaLsciSettings{.window_size = 3,
                                      .map = holotask::syncs::PcaLsciMap::Original});
  std::vector<float> output(4);
  std::memcpy(output.data(), result.output_bytes[0].data(), 4 * sizeof(float));
  const float expected = static_cast<float>(std::sqrt(5.25) / 60003.5);
  for (float value : output) EXPECT_NEAR(value, expected, 1e-9f);
}

TEST(PcaLsciTest, ReusesTaskWhenMapAndSpatialWindowChange) {
  const std::vector<float> frames{11.0f, 9.0f, 10.0f, 10.0f,
                                  10.0f, 10.0f, 10.5f, 9.5f};
  holotask::syncs::PcaLsciFactory factory;
  const std::array input_descs{TDesc({2, 2, 2}, DType::F32, MemLoc::Device)};
  const auto output_desc = factory.infer(input_descs, holotask::syncs::PcaLsciSettings{});
  auto task = factory.create(input_descs, holotask::syncs::PcaLsciSettings{}, {});
  task->bind_logger(spdlog::default_logger());
  const auto *original_task = task.get();
  holonp_test::TensorTestBuffer input(input_descs[0]);
  holonp_test::TensorTestBuffer output(output_desc.output_descs[0]);
  input.upload(as_bytes(frames));
  std::array inputs{input.view()};
  std::array outputs{output.view()};
  std::atomic<bool> cancelled{false};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};

  task = factory.update(std::move(task), input_descs,
      holotask::syncs::PcaLsciSettings{.window_size = 1,
                                       .map = holotask::syncs::PcaLsciMap::Original}, {});
  ASSERT_EQ(task.get(), original_task);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> original(4);
  const auto original_bytes = output.download();
  std::memcpy(original.data(), original_bytes.data(), original_bytes.size());
  EXPECT_NEAR(original[2], 0.25f / 10.25f, 1e-5f);

  task = factory.update(std::move(task), input_descs,
      holotask::syncs::PcaLsciSettings{.window_size = 3,
                                       .map = holotask::syncs::PcaLsciMap::Static}, {});
  ASSERT_EQ(task.get(), original_task);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> static_map(4);
  const auto static_bytes = output.download();
  std::memcpy(static_map.data(), static_bytes.data(), static_bytes.size());
  for (float value : static_map) EXPECT_NEAR(value, 0.05f, 1e-5f);
}
