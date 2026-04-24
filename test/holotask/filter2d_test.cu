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

#include <cmath>
#include <cstring>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/filter2d.hh"

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

nlohmann::json all_pass_settings() {
  return holotask::syncs::Filter2DSettings{
      .r_inner = 0,
      .r_outer = 100,
      .s_inner = 0,
      .s_outer = 0,
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

class Filter2DInferTest : public ::testing::Test {
protected:
  holotask::syncs::Filter2DFactory factory;
};

TEST_F(Filter2DInferTest, AcceptsRealInputAndOutputsComplex) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};
  const auto               r  = factory.infer(in, all_pass_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(Filter2DInferTest, KeepsComplexInputInPlace) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::CF32)};
  const auto               r  = factory.infer(in, all_pass_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  ASSERT_EQ(r.in_place.size(), 1);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
}

class Filter2DExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::Filter2DFactory factory;
};

TEST_F(Filter2DExecuteTest, RealInputUsesComplexOutput) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  const auto run =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                    std::vector<std::vector<std::byte>>{data}, all_pass_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::CF32);

  expect_cf32_near(run.output_bytes[0],
                   std::vector<CF32>{{4.0f, 4.0f}, {8.0f, 8.0f}, {12.0f, 12.0f}, {16.0f, 16.0f}});
}
