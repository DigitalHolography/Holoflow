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
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/flatfield.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

nlohmann::json settings(float sigma = 1.0f) {
  return holotask::syncs::FlatfieldSettings{.sigma = sigma};
}

void expect_f32_near(const std::vector<std::byte> &actual, const std::vector<float> &expected,
                     float atol = 1e-4f) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(float));
  const auto *a = reinterpret_cast<const float *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(a[i], expected[i], atol);
  }
}

} // namespace

class FlatfieldInferTest : public ::testing::Test {
protected:
  holotask::syncs::FlatfieldFactory factory;
};

TEST_F(FlatfieldInferTest, KeepsF32ShapeAndDtype) {
  const std::vector<TDesc> in = {device_desc({2, 3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3, 4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(FlatfieldInferTest, RejectsNonPositiveSigma) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};
  EXPECT_THROW(factory.infer(in, settings(0.0f)), std::invalid_argument);
}

class FlatfieldExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::FlatfieldFactory factory;
};

TEST_F(FlatfieldExecuteTest, ConstantInputSubtractsToZero) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f});

  const auto run = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{d}, std::vector<std::vector<std::byte>>{data}, settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::F32);
  expect_f32_near(run.output_bytes[0], std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
}
