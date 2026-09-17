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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"
#include "holonp/abs.hh"
#include "holonp/add.hh"
#include "holonp/argmax.hh"
#include "holonp/arange.hh"
#include "holonp/concatenate.hh"
#include "holonp/equal.hh"
#include "holonp/fft.hh"
#include "holonp/fftshift.hh"
#include "holonp/meshgrid.hh"
#include "holonp/multiply.hh"
#include "holonp/reshape.hh"
#include "holonp/rfft.hh"
#include "holonp/slice.hh"
#include "holonp/subtract.hh"
#include "holonp/where.hh"
#include "holonp/zeros.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T> std::vector<T> from_bytes(const std::vector<std::byte> &bytes) {
  EXPECT_EQ(bytes.size() % sizeof(T), 0u);
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

} // namespace

TEST(HolonpEdgeCaseTest, AbsHandlesSpecialFloatValues) {
  holonp::AbsFactory factory;
  const TDesc        desc = device_desc({4}, DType::F32);
  const float        nan  = std::numeric_limits<float>::quiet_NaN();
  const auto         data = as_bytes(std::vector<float>{-INFINITY, -0.F, nan, INFINITY});

  const auto run = holonp_test::run_sync_factory(factory, {&desc, 1}, {&data, 1}, {});
  const auto actual = from_bytes<float>(run.output_bytes[0]);

  ASSERT_EQ(actual.size(), 4u);
  EXPECT_EQ(actual[0], INFINITY);
  EXPECT_EQ(actual[1], 0.F);
  EXPECT_FALSE(std::signbit(actual[1]));
  EXPECT_TRUE(std::isnan(actual[2]));
  EXPECT_EQ(actual[3], INFINITY);
}

TEST(HolonpEdgeCaseTest, UnsignedAddAndSubtractWrapAtTheDtypeWidth) {
  const TDesc desc = device_desc({3}, DType::U8);
  const auto  a    = as_bytes(std::vector<std::uint8_t>{255, 0, 128});
  const auto  b    = as_bytes(std::vector<std::uint8_t>{1, 1, 128});
  const std::array<TDesc, 2>                 input_descs{desc, desc};
  const std::array<std::vector<std::byte>, 2> input_data{a, b};

  holonp::AddFactory add;
  const auto         add_run = holonp_test::run_sync_factory(add, input_descs, input_data, {});
  EXPECT_EQ(from_bytes<std::uint8_t>(add_run.output_bytes[0]),
            (std::vector<std::uint8_t>{0, 1, 0}));

  holonp::SubtractFactory subtract;
  const auto               subtract_run =
      holonp_test::run_sync_factory(subtract, input_descs, input_data, {});
  EXPECT_EQ(from_bytes<std::uint8_t>(subtract_run.output_bytes[0]),
            (std::vector<std::uint8_t>{254, 255, 0}));
}

TEST(HolonpEdgeCaseTest, UnsignedMultiplyWrapsAtTheDtypeWidth) {
  holonp::MultiplyFactory factory;
  const TDesc             desc = device_desc({2}, DType::U16);
  const auto              a    = as_bytes(std::vector<std::uint16_t>{65535, 32768});
  const auto              b    = as_bytes(std::vector<std::uint16_t>{2, 3});
  const std::array<TDesc, 2>                 input_descs{desc, desc};
  const std::array<std::vector<std::byte>, 2> input_data{a, b};

  const auto run = holonp_test::run_sync_factory(factory, input_descs, input_data, {});
  EXPECT_EQ(from_bytes<std::uint16_t>(run.output_bytes[0]),
            (std::vector<std::uint16_t>{65534, 32768}));
}

TEST(HolonpEdgeCaseTest, EqualTreatsNaNAsUnequal) {
  holonp::EqualFactory factory;
  const TDesc          desc = device_desc({3}, DType::F32);
  const float          nan  = std::numeric_limits<float>::quiet_NaN();
  const auto           a    = as_bytes(std::vector<float>{nan, 1.F, 2.F});
  const auto           b    = as_bytes(std::vector<float>{nan, 1.F, nan});
  const std::array<TDesc, 2>                 input_descs{desc, desc};
  const std::array<std::vector<std::byte>, 2> input_data{a, b};

  const auto run = holonp_test::run_sync_factory(factory, input_descs, input_data, {});
  EXPECT_EQ(from_bytes<std::uint8_t>(run.output_bytes[0]),
            (std::vector<std::uint8_t>{0, 1, 0}));
}

TEST(HolonpEdgeCaseTest, WhereUsesAnyNonzeroConditionAsTrue) {
  holonp::WhereFactory factory;
  const TDesc          condition_desc = device_desc({4}, DType::U8);
  const TDesc          value_desc     = device_desc({4}, DType::F32);
  const auto           condition      = as_bytes(std::vector<std::uint8_t>{0, 2, 255, 1});
  const auto           x              = as_bytes(std::vector<float>{1.F, 2.F, 3.F, 4.F});
  const auto           y              = as_bytes(std::vector<float>{10.F, 20.F, 30.F, 40.F});

  const std::array<TDesc, 3>                 input_descs{condition_desc, value_desc, value_desc};
  const std::array<std::vector<std::byte>, 3> input_data{condition, x, y};
  const auto run = holonp_test::run_sync_factory(factory, input_descs, input_data, {});
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]),
            (std::vector<float>{10.F, 2.F, 3.F, 4.F}));
}

TEST(HolonpEdgeCaseTest, ArgmaxChoosesTheFirstIndexOnTies) {
  holonp::ArgmaxFactory factory;
  const TDesc           desc = device_desc({2, 4}, DType::F32);
  const auto            data = as_bytes(std::vector<float>{5.F, 5.F, 1.F, 0.F,
                                                             -1.F, -2.F, -1.F, -3.F});
  const auto             run = holonp_test::run_sync_factory(
      factory, {&desc, 1}, {&data, 1}, nlohmann::json{{"axis", 1}});

  EXPECT_EQ(from_bytes<std::uint16_t>(run.output_bytes[0]),
            (std::vector<std::uint16_t>{0, 0}));
}

TEST(HolonpEdgeCaseTest, ArangeSupportsDescendingFractionalRanges) {
  holonp::ArangeFactory factory;
  const auto            settings = nlohmann::json{{"start", 1.0}, {"stop", 0.0}, {"step", -0.25}};

  const auto run = holonp_test::run_sync_factory(factory, {}, {}, settings);
  const auto actual = from_bytes<float>(run.output_bytes[0]);
  ASSERT_EQ(actual.size(), 4u);
  EXPECT_FLOAT_EQ(actual[0], 1.F);
  EXPECT_FLOAT_EQ(actual[1], 0.75F);
  EXPECT_FLOAT_EQ(actual[2], 0.5F);
  EXPECT_FLOAT_EQ(actual[3], 0.25F);
}

TEST(HolonpEdgeCaseTest, ZerosPreservesZeroElementShapesDuringInference) {
  holonp::ZerosFactory factory;
  const auto           result = factory.infer({}, nlohmann::json{{"shape", {2, 0, 3}}});

  ASSERT_EQ(result.output_descs.size(), 1u);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 0, 3}));
  EXPECT_EQ(result.output_descs[0].num_elements(), 0u);
}

TEST(HolonpEdgeCaseTest, ReshapeInfersOneNegativeDimension) {
  holonp::ReshapeFactory factory;
  const TDesc             input = device_desc({2, 3, 4}, DType::F32);
  const auto              result = factory.infer({&input, 1}, nlohmann::json{{"shape", {-1, 4}}});

  ASSERT_EQ(result.output_descs.size(), 1u);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{6, 4}));
}

TEST(HolonpEdgeCaseTest, SliceRejectsNonPositiveSteps) {
  holonp::SliceFactory factory;
  const TDesc          input = device_desc({4}, DType::F32);

  for (const auto step : {0, -1}) {
    const auto settings = nlohmann::json{
        {"slices", {{{"start", nullptr}, {"stop", nullptr}, {"step", step}}}}};
    EXPECT_THROW(factory.infer({&input, 1}, settings), std::invalid_argument) << step;
  }
}

TEST(HolonpEdgeCaseTest, ConcatenateAcceptsOneInput) {
  holonp::ConcatenateFactory factory;
  const TDesc                input = device_desc({2, 2}, DType::F32);
  const auto                 data  = as_bytes(std::vector<float>{1.F, 2.F, 3.F, 4.F});

  const auto run = holonp_test::run_sync_factory(
      factory, {&input, 1}, {&data, 1}, nlohmann::json{{"axis", 0}});
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]),
            (std::vector<float>{1.F, 2.F, 3.F, 4.F}));
}

TEST(HolonpEdgeCaseTest, MeshgridWithOneInputIsTheInputVector) {
  holonp::MeshgridFactory factory;
  const TDesc             input = device_desc({3}, DType::F32);
  const auto              data  = as_bytes(std::vector<float>{1.F, 2.F, 3.F});

  const auto run = holonp_test::run_sync_factory(
      factory, {&input, 1}, {&data, 1}, nlohmann::json{{"indexing", "ij"}});
  ASSERT_EQ(run.output_bytes.size(), 1u);
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]), (std::vector<float>{1.F, 2.F, 3.F}));
}

TEST(HolonpEdgeCaseTest, FftLengthOneIsAnIdentityTransform) {
  holonp::FFTFactory factory;
  const TDesc        input = device_desc({2, 1}, DType::F32);
  const auto         data  = as_bytes(std::vector<float>{3.F, -4.F});

  const auto run = holonp_test::run_sync_factory(
      factory, {&input, 1}, {&data, 1}, nlohmann::json{{"axis", -1}});
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]),
            (std::vector<float>{3.F, 0.F, -4.F, 0.F}));
}

TEST(HolonpEdgeCaseTest, RfftOddLengthIncludesTheNyquistBin) {
  holonp::RFFTFactory factory;
  const TDesc         input = device_desc({5}, DType::F32);
  const auto          data  = as_bytes(std::vector<float>{1.F, 0.F, 0.F, 0.F, 0.F});

  const auto run = holonp_test::run_sync_factory(
      factory, {&input, 1}, {&data, 1}, nlohmann::json{{"axis", 0}});
  ASSERT_EQ(run.output_descs[0].shape, (std::vector<size_t>{3}));
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]),
            (std::vector<float>{1.F, 0.F, 1.F, 0.F, 1.F, 0.F}));
}

TEST(HolonpEdgeCaseTest, FftshiftUsesNumpyCenterForOddLength) {
  holonp::FFTShiftFactory factory;
  const TDesc            input = device_desc({5}, DType::F32);
  const auto             data  = as_bytes(std::vector<float>{0.F, 1.F, 2.F, 3.F, 4.F});

  const auto run = holonp_test::run_sync_factory(factory, {&input, 1}, {&data, 1}, {});
  EXPECT_EQ(from_bytes<float>(run.output_bytes[0]),
            (std::vector<float>{3.F, 4.F, 0.F, 1.F, 2.F}));
}
