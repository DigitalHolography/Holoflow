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
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"
#include "holonp/matmul.hh"
#include "holonp/sum.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

std::vector<std::byte> float_bytes(const std::vector<float> &values) {
  std::vector<std::byte> result(values.size() * sizeof(float));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

std::vector<float> floats(const std::vector<std::byte> &bytes) {
  EXPECT_EQ(bytes.size() % sizeof(float), 0u);
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

} // namespace

TEST(HolonpGeneralCaseTest, RankThreeReductionAlongMiddleAxis) {
  const TDesc input({2, 3, 4}, DType::F32, MemLoc::Device);
  std::vector<float> values(24);
  for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i + 1);

  holonp::SumFactory factory;
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> payloads{float_bytes(values)};
  const auto run = holonp_test::run_sync_factory(
      factory, descriptors, payloads, nlohmann::json{{"axis", -2}});

  ASSERT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 4}));
  EXPECT_EQ(floats(run.output_bytes[0]),
            (std::vector<float>{15.f, 18.f, 21.f, 24.f, 51.f, 54.f, 57.f, 60.f}));
}

TEST(HolonpComplexityTest, SumHandlesIncreasingInputSizes) {
  holonp::SumFactory factory;
  for (const size_t count : {1024u, 65536u, 1048576u}) {
    SCOPED_TRACE(count);
    const TDesc input({count}, DType::F32, MemLoc::Device);
    const std::vector<float> values(count, 1.f);
    const std::vector<TDesc> descriptors{input};
    const std::vector<std::vector<std::byte>> payloads{float_bytes(values)};

    const auto run = holonp_test::run_sync_factory(
        factory, descriptors, payloads, nlohmann::json::object());
    ASSERT_EQ(run.output_descs[0].shape, (std::vector<size_t>{}));
    ASSERT_EQ(run.output_bytes[0].size(), sizeof(float));
    EXPECT_FLOAT_EQ(floats(run.output_bytes[0])[0], static_cast<float>(count));
  }
}

TEST(HolonpComplexityTest, MatmulHandlesIncreasingSquareMatrices) {
  holonp::MatmulFactory factory;
  for (const size_t side : {4u, 16u, 64u}) {
    SCOPED_TRACE(side);
    const TDesc input({side, side}, DType::F32, MemLoc::Device);
    const std::vector<TDesc> descriptors{input, input};
    const std::vector<float> values(side * side, 1.f);
    const auto payload = float_bytes(values);
    const std::vector<std::vector<std::byte>> payloads{payload, payload};

    const auto run = holonp_test::run_sync_factory(
        factory, descriptors, payloads, nlohmann::json::object());
    ASSERT_EQ(run.output_descs[0].shape, (std::vector<size_t>{side, side}));
    const auto output = floats(run.output_bytes[0]);
    ASSERT_EQ(output.size(), side * side);
    for (const float value : output) EXPECT_FLOAT_EQ(value, static_cast<float>(side));
  }
}
