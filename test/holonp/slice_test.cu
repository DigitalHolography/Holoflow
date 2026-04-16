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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/slice.hh"

#include "python_oracle.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

static TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

static void expect_near_oracle(const std::vector<std::byte> &actual,
                               const std::vector<std::byte> &expected, DType dtype,
                               float rtol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t n = actual.size() / holoflow::core::size_of(dtype);
  if (dtype == DType::U8) {
    const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]);
  } else if (dtype == DType::U16) {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]);
  } else if (dtype == DType::F32) {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol);
    }
  } else {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n * 2; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol);
    }
  }
}

class SliceInferTest : public ::testing::Test {
protected:
  holonp::SliceFactory factory;
};

TEST_F(SliceInferTest, ComputesShapeStridesAndOffsetForMixedSliceItems) {
  const TDesc in = device_desc({3, 4, 5}, DType::F32);
  const auto  j  = nlohmann::json{
        {"slices",
         {
           {{"start", 1}, {"stop", 3}, {"step", 1}},
           2,
           {{"start", nullptr}, {"stop", nullptr}, {"step", 2}},
       }},
  };

  const auto r = factory.infer(std::vector<TDesc>{in}, j);
  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  const auto &out = r.output_descs[0];
  EXPECT_EQ(out.shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(out.strides, (std::vector<size_t>{80, 8}));
  EXPECT_EQ(out.offset, 120u);
}

TEST_F(SliceInferTest, RejectsOutOfRangeIndex) {
  const TDesc in = device_desc({3, 4}, DType::F32);
  const auto  j  = nlohmann::json{
        {"slices",
         {
           3,
           {{"start", nullptr}, {"stop", nullptr}, {"step", 1}},
       }},
  };
  EXPECT_THROW(factory.infer(std::vector<TDesc>{in}, j), std::out_of_range);
}

class SliceOracleTest : public ::testing::Test {
protected:
  holonp::SliceFactory factory;
};

TEST_F(SliceOracleTest, InferredDescriptorMaterializationMatchesNumpySlice) {
  const TDesc in = device_desc({3, 4}, DType::F32);
  const auto  j  = nlohmann::json{
        {"slices",
         {
           {{"start", 1}, {"stop", 3}, {"step", 1}},
           {{"start", 0}, {"stop", 4}, {"step", 2}},
       }},
  };
  const auto input_bytes =
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});

  const auto infer    = factory.infer(std::vector<TDesc>{in}, j);
  const auto out_desc = infer.output_descs[0];

  holonp_test::OracleInput materialize;
  materialize.op          = "ascontiguousarray";
  materialize.n_outputs   = 1;
  materialize.input_descs = {out_desc};
  materialize.input_bytes = {input_bytes};
  const auto view_bytes   = holonp_test::invoke_oracle(materialize, kOracleScript);

  holonp_test::OracleInput expected;
  expected.op          = "slice";
  expected.n_outputs   = 1;
  expected.input_descs = {in};
  expected.input_bytes = {input_bytes};
  expected.settings    = j;
  const auto oracle    = holonp_test::invoke_oracle(expected, kOracleScript);

  expect_near_oracle(view_bytes.output_bytes[0], oracle.output_bytes[0], DType::F32);
}
