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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <cuComplex.h>
#include <nlohmann/json.hpp>

#include "curaii/cuda.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/square.hh"

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

TDesc device_desc(std::vector<size_t> shape, DType dtype, std::vector<size_t> strides) {
  return TDesc(std::move(shape), dtype, MemLoc::Device, std::move(strides));
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T>
std::vector<std::byte> make_strided_2d_bytes(const std::vector<T> &logical, size_t rows,
                                             size_t cols, size_t row_stride_bytes) {
  std::vector<std::byte> bytes(rows * row_stride_bytes, std::byte{0});
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      const size_t src_idx = row * cols + col;
      const size_t dst_off = row * row_stride_bytes + col * sizeof(T);
      std::memcpy(bytes.data() + dst_off, &logical[src_idx], sizeof(T));
    }
  }
  return bytes;
}

void expect_matches(const std::vector<std::byte> &actual, const std::vector<std::byte> &expected,
                    DType dtype) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t total_out = actual.size() / holoflow::core::size_of(dtype);

  if (dtype == DType::U8) {
    const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
    for (size_t i = 0; i < total_out; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    return;
  }

  if (dtype == DType::U16) {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.data());
    for (size_t i = 0; i < total_out; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    return;
  }

  const auto  *a               = reinterpret_cast<const float *>(actual.data());
  const auto  *e               = reinterpret_cast<const float *>(expected.data());
  const size_t component_count = actual.size() / sizeof(float);
  for (size_t i = 0; i < component_count; ++i) {
    const float tolerance = 1e-6f * std::max(std::abs(e[i]), 1.0f);
    EXPECT_NEAR(a[i], e[i], tolerance) << "at component " << i;
  }
}

void expect_matches_numpy(const TDesc &idesc, const std::vector<std::byte> &input_bytes) {
  holonp::SquareFactory factory;
  const auto            run = holonp_test::run_sync_factory(factory, {&idesc, 1}, {&input_bytes, 1},
                                                            nlohmann::json::object());

  holonp_test::OracleInput oracle_input;
  oracle_input.op          = "square";
  oracle_input.n_outputs   = 1;
  oracle_input.input_descs = {idesc};
  oracle_input.input_bytes = {input_bytes};
  const auto oracle        = holonp_test::invoke_oracle(oracle_input, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_matches(run.output_bytes[0], oracle.output_bytes[0], idesc.dtype);
}

} // namespace

// -------------------------------------------------------------------------------------------------
// SquareFactory: inference tests
// -------------------------------------------------------------------------------------------------

TEST(SquareInferTest, PreservesShapeAndDtype) {
  holonp::SquareFactory factory;

  for (const auto dtype : {DType::U8, DType::U16, DType::F32, DType::CF32}) {
    const std::vector<TDesc> inputs{device_desc({2, 3}, dtype)};
    const auto               result = factory.infer(inputs, nlohmann::json::object());

    EXPECT_EQ(result.kind, TaskKind::Sync);
    ASSERT_EQ(result.input_descs.size(), 1u);
    ASSERT_EQ(result.output_descs.size(), 1u);
    EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(result.output_descs[0].dtype, dtype);
    EXPECT_EQ(result.output_descs[0].mem_loc, MemLoc::Device);
    ASSERT_EQ(result.owned_inputs.size(), 1u);
    EXPECT_FALSE(result.owned_inputs[0]);
    EXPECT_TRUE(result.in_place.empty());
  }
}

TEST(SquareInferTest, AcceptsStridedInputAndProducesContiguousOutput) {
  holonp::SquareFactory    factory;
  const std::vector<TDesc> inputs{device_desc({2, 2}, DType::F32, {16, 4})};
  const auto               result = factory.infer(inputs, nlohmann::json::object());

  EXPECT_EQ(result.output_descs[0].strides, (std::vector<size_t>{8, 4}));
}

TEST(SquareInferTest, RejectsInvalidInputs) {
  holonp::SquareFactory factory;

  EXPECT_THROW(factory.infer({}, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> host_input{TDesc({4}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(host_input, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> empty_input{device_desc({0}, DType::F32)};
  EXPECT_THROW(factory.infer(empty_input, nlohmann::json::object()), std::invalid_argument);

  const std::vector<TDesc> invalid_stride{device_desc({2}, DType::F32, {5})};
  EXPECT_THROW(factory.infer(invalid_stride, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// SquareFactory: execution tests
// -------------------------------------------------------------------------------------------------

TEST(SquareOracleTest, U8MatchesNumpyIncludingOverflow) {
  const auto idesc = device_desc({5}, DType::U8);
  expect_matches_numpy(idesc, as_bytes(std::vector<std::uint8_t>{0, 2, 15, 16, 255}));
}

TEST(SquareOracleTest, U16MatchesNumpyIncludingOverflow) {
  const auto idesc = device_desc({4}, DType::U16);
  expect_matches_numpy(idesc, as_bytes(std::vector<std::uint16_t>{0, 10, 255, 300}));
}

TEST(SquareOracleTest, F32MatchesNumpy) {
  const auto idesc = device_desc({2, 3}, DType::F32);
  expect_matches_numpy(idesc, as_bytes(std::vector<float>{-4.0f, -0.5f, 0.0f, 1.5f, 3.0f, 10.0f}));
}

TEST(SquareOracleTest, CF32MatchesNumpy) {
  const auto idesc = device_desc({4}, DType::CF32);
  expect_matches_numpy(idesc,
                       as_bytes(std::vector<cuFloatComplex>{
                           make_cuFloatComplex(1.0f, 2.0f), make_cuFloatComplex(-3.0f, 4.0f),
                           make_cuFloatComplex(0.0f, -2.0f), make_cuFloatComplex(-1.5f, -0.5f)}));
}

TEST(SquareOracleTest, StridedF32MatchesNumpy) {
  const auto idesc       = device_desc({2, 2}, DType::F32, {16, 4});
  const auto input_bytes = make_strided_2d_bytes<float>({-1.0f, 2.0f, -3.0f, 4.0f}, 2, 2, 16);
  expect_matches_numpy(idesc, input_bytes);
}

// -------------------------------------------------------------------------------------------------
// SquareFactory: update tests
// -------------------------------------------------------------------------------------------------

TEST(SquareUpdateTest, ReusesTaskWithSameDescriptor) {
  holonp::SquareFactory    factory;
  curaii::CudaStream       stream;
  const auto               idesc = device_desc({4}, DType::F32);
  const std::vector<TDesc> inputs{idesc};
  const auto               settings = nlohmann::json::object();

  auto  task     = factory.create(inputs, settings, {stream.get()});
  auto *original = task.get();
  task           = factory.update(std::move(task), inputs, settings, {stream.get()});

  EXPECT_EQ(task.get(), original);
}

TEST(SquareUpdateTest, RecreatesTaskWithChangedDescriptor) {
  holonp::SquareFactory    factory;
  curaii::CudaStream       stream;
  const std::vector<TDesc> initial_inputs{device_desc({4}, DType::F32)};
  const std::vector<TDesc> updated_inputs{device_desc({2, 2}, DType::F32)};
  const auto               settings = nlohmann::json::object();

  auto  task     = factory.create(initial_inputs, settings, {stream.get()});
  auto *original = task.get();
  task           = factory.update(std::move(task), updated_inputs, settings, {stream.get()});

  EXPECT_NE(task.get(), original);
}
