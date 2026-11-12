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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <vector>

#include <cuComplex.h>
#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"
#include "holonp/exp.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void expect_near(const std::vector<std::byte> &actual, const std::vector<std::byte> &expected,
                 DType dtype) {
  ASSERT_EQ(actual.size(), expected.size());

  if (dtype == DType::F32) {
    const auto  *actual_values   = reinterpret_cast<const float *>(actual.data());
    const auto  *expected_values = reinterpret_cast<const float *>(expected.data());
    const size_t total_out       = actual.size() / sizeof(float);
    for (size_t i = 0; i < total_out; ++i) {
      const float tolerance = 2e-6f * std::max(std::abs(expected_values[i]), 1.0f);
      EXPECT_NEAR(actual_values[i], expected_values[i], tolerance) << "at index " << i;
    }
    return;
  }

  ASSERT_EQ(dtype, DType::CF32);
  const auto  *actual_values   = reinterpret_cast<const cuFloatComplex *>(actual.data());
  const auto  *expected_values = reinterpret_cast<const cuFloatComplex *>(expected.data());
  const size_t total_out       = actual.size() / sizeof(cuFloatComplex);
  for (size_t i = 0; i < total_out; ++i) {
    const float real_tolerance = 2e-6f * std::max(std::abs(expected_values[i].x), 1.0f);
    const float imag_tolerance = 2e-6f * std::max(std::abs(expected_values[i].y), 1.0f);
    EXPECT_NEAR(actual_values[i].x, expected_values[i].x, real_tolerance)
        << "real component at index " << i;
    EXPECT_NEAR(actual_values[i].y, expected_values[i].y, imag_tolerance)
        << "imaginary component at index " << i;
  }
}

template <typename T>
void expect_matches_numpy(DType dtype, const std::vector<size_t> &shape,
                          const std::vector<T> &values, bool update = false) {
  holonp::ExpFactory                        factory;
  const auto                                idesc       = device_desc(shape, dtype);
  const auto                                input_bytes = as_bytes(values);
  const std::vector<TDesc>                  input_descs{idesc};
  const std::vector<std::vector<std::byte>> inputs{input_bytes};
  const auto                                settings = nlohmann::json::object();

  const auto run =
      update ? holonp_test::run_sync_factory_update(factory, input_descs, inputs, settings)
             : holonp_test::run_sync_factory(factory, input_descs, inputs, settings);

  holonp_test::OracleInput oracle_input;
  oracle_input.op          = "exp";
  oracle_input.n_outputs   = 1;
  oracle_input.input_descs = input_descs;
  oracle_input.input_bytes = inputs;
  const auto oracle        = holonp_test::invoke_oracle(oracle_input, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_near(run.output_bytes[0], oracle.output_bytes[0], dtype);
}

} // namespace

// -------------------------------------------------------------------------------------------------
// ExpFactory: inference tests
// -------------------------------------------------------------------------------------------------

TEST(ExpInferTest, PreservesF32Descriptor) {
  holonp::ExpFactory       factory;
  const std::vector<TDesc> inputs{device_desc({2, 3}, DType::F32)};
  const auto               result = factory.infer(inputs, nlohmann::json::object());

  EXPECT_EQ(result.kind, TaskKind::Sync);
  ASSERT_EQ(result.output_descs.size(), 1u);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(result.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_TRUE(result.in_place.empty());
}

TEST(ExpInferTest, PreservesCF32Descriptor) {
  holonp::ExpFactory       factory;
  const std::vector<TDesc> inputs{device_desc({4}, DType::CF32)};
  const auto               result = factory.infer(inputs, nlohmann::json::object());

  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::CF32);
}

TEST(ExpInferTest, RejectsUnsupportedInputs) {
  holonp::ExpFactory factory;

  EXPECT_THROW(factory.infer({}, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> integer_input{device_desc({4}, DType::U16)};
  EXPECT_THROW(factory.infer(integer_input, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> host_input{TDesc({4}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(host_input, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> strided_input{
      TDesc({4}, DType::F32, MemLoc::Device, std::vector<size_t>{8})};
  EXPECT_THROW(factory.infer(strided_input, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> empty_input{device_desc({0}, DType::F32)};
  EXPECT_THROW(factory.infer(empty_input, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ExpFactory: execution tests
// -------------------------------------------------------------------------------------------------

TEST(ExpOracleTest, F32MatchesNumpy) {
  expect_matches_numpy(DType::F32, {2, 3},
                       std::vector<float>{-4.0f, -1.0f, 0.0f, 0.5f, 1.0f, 4.0f});
}

TEST(ExpOracleTest, CF32MatchesNumpy) {
  const std::vector<cuFloatComplex> values{
      make_cuFloatComplex(0.0f, 0.0f),
      make_cuFloatComplex(0.0f, 1.57079632679f),
      make_cuFloatComplex(1.0f, -1.0f),
      make_cuFloatComplex(-2.0f, 3.0f),
  };
  expect_matches_numpy(DType::CF32, {2, 2}, values);
}

TEST(ExpUpdateTest, ReusesTaskAndMatchesNumpy) {
  expect_matches_numpy(DType::CF32, {3},
                       std::vector<cuFloatComplex>{make_cuFloatComplex(0.0f, -1.0f),
                                                   make_cuFloatComplex(0.5f, 2.0f),
                                                   make_cuFloatComplex(-1.0f, 0.25f)},
                       true);
}
