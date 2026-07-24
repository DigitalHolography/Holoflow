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

#include <cstring>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/angular_spectrum.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

struct CF32 {
  float re;
  float im;
};

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

nlohmann::json zero_distance_settings() {
  return holotask::syncs::AngularSpectrumSettings{
      .lambda = 852e-9f,
      .dx     = 20e-6f,
      .dy     = 20e-6f,
      .z      = 0.0f,
      .filter = std::nullopt,
  };
}

nlohmann::json zero_distance_padded_settings(int width = 4, int height = 4) {
  return holotask::syncs::AngularSpectrumSettings{
      .lambda = 852e-9f,
      .dx     = 20e-6f,
      .dy     = 20e-6f,
      .z      = 0.0f,
      .filter = std::nullopt,
      .padding =
          holotask::syncs::AngularSpectrumSettings::Padding{
              .width  = width,
              .height = height,
          },
  };
}

void expect_cf32_near(const std::vector<std::byte> &actual, const std::vector<CF32> &expected,
                      float atol = 1e-3f) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(CF32));
  const auto *a = reinterpret_cast<const CF32 *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(a[i].re, expected[i].re, atol);
    EXPECT_NEAR(a[i].im, expected[i].im, atol);
  }
}

} // namespace

class AngularSpectrumInferTest : public ::testing::Test {
protected:
  holotask::syncs::AngularSpectrumFactory factory;
};

TEST_F(AngularSpectrumInferTest, AcceptsRealInputAndOutputsComplex) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};
  const auto               r  = factory.infer(in, zero_distance_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AngularSpectrumInferTest, KeepsComplexInputInPlace) {
  const std::vector<TDesc> in = {device_desc({2, 2, 2, 2}, DType::CF32)};
  const auto               r  = factory.infer(in, zero_distance_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2, 2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  ASSERT_EQ(r.in_place.size(), 1);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
}

TEST_F(AngularSpectrumInferTest, PaddingChangesTrailingShapeAndDisablesInPlace) {
  const std::vector<TDesc> in = {device_desc({2, 3, 2, 2}, DType::CF32)};
  const auto               r  = factory.infer(in, zero_distance_padded_settings(6, 4));

  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3, 4, 6}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AngularSpectrumInferTest, RejectsPaddingSmallerThanInput) {
  const std::vector<TDesc> in = {device_desc({4, 4}, DType::F32)};

  EXPECT_THROW(factory.infer(in, zero_distance_padded_settings(2, 4)), std::invalid_argument);
  EXPECT_THROW(factory.infer(in, zero_distance_padded_settings(4, 2)), std::invalid_argument);
}

TEST_F(AngularSpectrumInferTest, RejectsOddPaddingMargins) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};

  EXPECT_THROW(factory.infer(in, zero_distance_padded_settings(3, 4)), std::invalid_argument);
  EXPECT_THROW(factory.infer(in, zero_distance_padded_settings(4, 3)), std::invalid_argument);
}

class AngularSpectrumExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::AngularSpectrumFactory factory;
};

TEST_F(AngularSpectrumExecuteTest, RealInputUsesComplexOutput) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                 std::vector<std::vector<std::byte>>{data},
                                                 zero_distance_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::CF32);

  expect_cf32_near(run.output_bytes[0],
                   std::vector<CF32>{{4.0f, 0.0f}, {8.0f, 0.0f}, {12.0f, 0.0f}, {16.0f, 0.0f}});
}

TEST_F(AngularSpectrumExecuteTest, CentersRealInputInZeroPaddedOutput) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                 std::vector<std::vector<std::byte>>{data},
                                                 zero_distance_padded_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{4, 4}));
  expect_cf32_near(run.output_bytes[0], std::vector<CF32>{
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {16.0f, 0.0f},
                                            {32.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {48.0f, 0.0f},
                                            {64.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                            {0.0f, 0.0f},
                                        });
}

TEST_F(AngularSpectrumExecuteTest, CentersBatchedComplexInputInZeroPaddedOutput) {
  const TDesc d    = device_desc({2, 2, 2}, DType::CF32);
  const auto  data = as_bytes(std::vector<CF32>{{1.0f, 1.0f},
                                                {2.0f, 2.0f},
                                                {3.0f, 3.0f},
                                                {4.0f, 4.0f},
                                                {5.0f, 5.0f},
                                                {6.0f, 6.0f},
                                                {7.0f, 7.0f},
                                                {8.0f, 8.0f}});

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                 std::vector<std::vector<std::byte>>{data},
                                                 zero_distance_padded_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 4, 4}));
  const auto *actual = reinterpret_cast<const CF32 *>(run.output_bytes[0].data());
  for (int batch = 0; batch < 2; ++batch) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        const int output_idx = batch * 16 + y * 4 + x;
        if (x >= 1 && x <= 2 && y >= 1 && y <= 2) {
          const int input_idx = batch * 4 + (y - 1) * 2 + x - 1;
          EXPECT_NEAR(actual[output_idx].re, (input_idx + 1) * 16.0f, 1e-3f);
          EXPECT_NEAR(actual[output_idx].im, (input_idx + 1) * 16.0f, 1e-3f);
        } else {
          EXPECT_NEAR(actual[output_idx].re, 0.0f, 1e-3f);
          EXPECT_NEAR(actual[output_idx].im, 0.0f, 1e-3f);
        }
      }
    }
  }
}

TEST_F(AngularSpectrumExecuteTest, CentersBatchedRealInputInZeroPaddedOutput) {
  const TDesc d    = device_desc({2, 2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

  const auto run = holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                                 std::vector<std::vector<std::byte>>{data},
                                                 zero_distance_padded_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 4, 4}));
  const auto *actual = reinterpret_cast<const CF32 *>(run.output_bytes[0].data());
  for (int batch = 0; batch < 2; ++batch) {
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        const int output_idx = batch * 16 + y * 4 + x;
        if (x >= 1 && x <= 2 && y >= 1 && y <= 2) {
          const int input_idx = batch * 4 + (y - 1) * 2 + x - 1;
          EXPECT_NEAR(actual[output_idx].re, (input_idx + 1) * 16.0f, 1e-3f);
        } else {
          EXPECT_NEAR(actual[output_idx].re, 0.0f, 1e-3f);
        }
        EXPECT_NEAR(actual[output_idx].im, 0.0f, 1e-3f);
      }
    }
  }
}

TEST_F(AngularSpectrumExecuteTest, AllPassFilterPreservesPropagationTransferFunction) {
  const TDesc       d = device_desc({4, 4}, DType::CF32);
  std::vector<CF32> input(16, CF32{0.0f, 0.0f});
  input[5]        = CF32{1.0f, 0.0f};
  const auto data = as_bytes(input);

  auto unfiltered_settings    = zero_distance_settings();
  unfiltered_settings["z"]    = 0.01f;
  auto filtered_settings      = unfiltered_settings;
  filtered_settings["filter"] = holotask::syncs::AngularSpectrumSettings::Filter{
      .r_inner = 0,
      .r_outer = 100,
      .s_inner = 0,
      .s_outer = 0,
  };

  const auto unfiltered =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                    std::vector<std::vector<std::byte>>{data}, unfiltered_settings);
  const auto filtered = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, filtered_settings);

  ASSERT_EQ(unfiltered.output_bytes[0].size(), filtered.output_bytes[0].size());
  const auto *expected = reinterpret_cast<const CF32 *>(unfiltered.output_bytes[0].data());
  const auto *actual   = reinterpret_cast<const CF32 *>(filtered.output_bytes[0].data());
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(actual[i].re, expected[i].re, 1e-3f);
    EXPECT_NEAR(actual[i].im, expected[i].im, 1e-3f);
  }
}
