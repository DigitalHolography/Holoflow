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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/concatenate.hh"

#include "python_oracle.hh"
#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

// Absolute path to oracle.py, baked in at compile time.
static const std::filesystem::path kOracleScript{HOLONP_TEST_ORACLE_SCRIPT};

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

static TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> static std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

static nlohmann::json axis_settings(std::optional<int> axis) {
  if (axis.has_value()) {
    return nlohmann::json{{"axis", *axis}};
  }
  return nlohmann::json{{"axis", nullptr}};
}

// Element-wise comparison: exact for integer types, toleranced for F32/CF32.
static void expect_near_oracle(const std::vector<std::byte> &actual,
                               const std::vector<std::byte> &expected, DType dtype,
                               float rtol = 1e-5f) {
  ASSERT_EQ(actual.size(), expected.size());
  const size_t n = actual.size() / holoflow::core::size_of(dtype);

  switch (dtype) {
  case DType::U8: {
    const auto *a = reinterpret_cast<const std::uint8_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint8_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    break;
  }
  case DType::U16: {
    const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
    const auto *e = reinterpret_cast<const std::uint16_t *>(expected.data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
    break;
  }
  case DType::F32: {
    const auto *a = reinterpret_cast<const float *>(actual.data());
    const auto *e = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol) << "at index " << i;
    }
    break;
  }
  case DType::CF32: {
    const size_t n_floats = actual.size() / sizeof(float);
    const auto  *a        = reinterpret_cast<const float *>(actual.data());
    const auto  *e        = reinterpret_cast<const float *>(expected.data());
    for (size_t i = 0; i < n_floats; ++i) {
      const float tol = rtol * std::max(std::abs(e[i]), 1.0f);
      EXPECT_NEAR(a[i], e[i], tol) << "at float component " << i;
    }
    break;
  }
  }
}

// -------------------------------------------------------------------------------------------------
// ConcatenateFactory: inference tests
// -------------------------------------------------------------------------------------------------

class ConcatenateInferTest : public ::testing::Test {
protected:
  holonp::ConcatenateFactory factory;
};

TEST_F(ConcatenateInferTest, DefaultAxis0) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({1, 2}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(ConcatenateInferTest, Axis1) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 1}, DType::F32)};
  const auto               r  = factory.infer(in, axis_settings(1));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
}

TEST_F(ConcatenateInferTest, NegativeAxis) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::U8), device_desc({2, 1}, DType::U8)};
  const auto               r  = factory.infer(in, axis_settings(-1));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
}

TEST_F(ConcatenateInferTest, AxisNullFlattensInputs) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::U16), device_desc({3}, DType::U16)};
  const auto               r  = factory.infer(in, axis_settings(std::nullopt));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{7}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ConcatenateInferTest, RejectsNoInputs) {
  const std::vector<TDesc> in;
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConcatenateInferTest, RejectsHostInput) {
  const std::vector<TDesc> in = {TDesc({2, 2}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConcatenateInferTest, RejectsMismatchedDtypes) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 2}, DType::U16)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(ConcatenateInferTest, AcceptsNonContiguousInput) {
  const TDesc              a({2, 2}, DType::F32, MemLoc::Device, std::vector<size_t>{16, 4});
  const std::vector<TDesc> in = {a, device_desc({2, 2}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4, 2}));
}

TEST_F(ConcatenateInferTest, RejectsAxisOutOfRange) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 2}, DType::F32)};
  EXPECT_THROW(factory.infer(in, axis_settings(2)), std::invalid_argument);
}

TEST_F(ConcatenateInferTest, RejectsRankMismatchWhenAxisIsSet) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, axis_settings(0)), std::invalid_argument);
}

TEST_F(ConcatenateInferTest, RejectsShapeMismatchOutsideAxis) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({1, 3}, DType::F32)};
  EXPECT_THROW(factory.infer(in, axis_settings(0)), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ConcatenateFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class ConcatenateOracleTest : public ::testing::Test {
protected:
  holonp::ConcatenateFactory factory;

  void check(DType dtype, const std::vector<TDesc> &input_descs,
             const std::vector<std::vector<std::byte>> &input_bytes,
             const nlohmann::json                      &settings) {
    const auto run = holonp_test::run_sync_factory(factory, input_descs, input_bytes, settings);

    holonp_test::OracleInput oi;
    oi.op             = "concatenate";
    oi.n_outputs      = 1;
    oi.input_descs    = input_descs;
    oi.input_bytes    = input_bytes;
    oi.settings       = settings;
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

    ASSERT_EQ(run.output_bytes.size(), 1u);
    ASSERT_EQ(oracle.output_bytes.size(), 1u);
    expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], dtype);
  }
};

TEST_F(ConcatenateOracleTest, F32Axis0) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({1, 2}, DType::F32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f}),
      as_bytes(std::vector<float>{5.f, 6.f}),
  };
  check(DType::F32, in, data, axis_settings(0));
}

TEST_F(ConcatenateOracleTest, F32Axis1) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 1}, DType::F32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f}),
      as_bytes(std::vector<float>{10.f, 20.f}),
  };
  check(DType::F32, in, data, axis_settings(1));
}

TEST_F(ConcatenateOracleTest, F32Axis1StridedInput) {
  const TDesc              a({2, 2}, DType::F32, MemLoc::Device, std::vector<size_t>{16, 4});
  const TDesc              b                     = device_desc({2, 1}, DType::F32);
  const std::vector<TDesc> in                    = {a, b};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, -1.f, -1.f, 3.f, 4.f, -1.f, -1.f}),
      as_bytes(std::vector<float>{10.f, 20.f}),
  };
  check(DType::F32, in, data, axis_settings(1));
}

TEST_F(ConcatenateOracleTest, F32Axis1OffsetStridedInput) {
  const TDesc              a({2, 2}, DType::F32, MemLoc::Device, std::vector<size_t>{16, 4}, 8);
  const TDesc              b                     = device_desc({2, 1}, DType::F32);
  const std::vector<TDesc> in                    = {a, b};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, -1.f, -1.f, 3.f, 4.f, -1.f, -1.f}),
      as_bytes(std::vector<float>{10.f, 20.f}),
  };

  const auto run = holonp_test::run_sync_factory(factory, in, data, axis_settings(1));

  expect_near_oracle(run.output_bytes[0],
                     as_bytes(std::vector<float>{1.f, 2.f, 10.f, 3.f, 4.f, 20.f}), DType::F32);
}

TEST_F(ConcatenateOracleTest, F32AxisNullFlatten) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({3}, DType::F32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f}),
      as_bytes(std::vector<float>{5.f, 6.f, 7.f}),
  };
  check(DType::F32, in, data, axis_settings(std::nullopt));
}

TEST_F(ConcatenateOracleTest, U8Axis0) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::U8), device_desc({1, 3}, DType::U8)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}),
      as_bytes(std::vector<std::uint8_t>{7, 8, 9}),
  };
  check(DType::U8, in, data, axis_settings(0));
}

TEST_F(ConcatenateOracleTest, CF32Axis1) {
  struct CF32 {
    float re, im;
  };

  const std::vector<TDesc>                  in   = {device_desc({2, 1}, DType::CF32),
                                                    device_desc({2, 2}, DType::CF32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<CF32>{{1.f, 2.f}, {3.f, 4.f}}),
      as_bytes(std::vector<CF32>{{5.f, -1.f}, {6.f, -2.f}, {7.f, -3.f}, {8.f, -4.f}}),
  };
  check(DType::CF32, in, data, axis_settings(1));
}

TEST_F(ConcatenateOracleTest, OutputDescMatchesInfer) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::U16), device_desc({2, 1}, DType::U16)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint16_t>{1, 2, 3, 4}),
      as_bytes(std::vector<std::uint16_t>{5, 6}),
  };
  const auto run = holonp_test::run_sync_factory(factory, in, data, axis_settings(1));

  ASSERT_EQ(run.output_descs.size(), 1u);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(run.output_descs[0].mem_loc, MemLoc::Device);
}

// -------------------------------------------------------------------------------------------------
// ConcatenateFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class ConcatenateUpdateTest : public ::testing::Test {
protected:
  holonp::ConcatenateFactory factory;
};

TEST_F(ConcatenateUpdateTest, ReusesConcatenateTask) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({1, 2}, DType::F32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f}),
      as_bytes(std::vector<float>{5.f, 6.f}),
  };

  const auto run = holonp_test::run_sync_factory_update(factory, in, data, axis_settings(0));

  holonp_test::OracleInput oi;
  oi.op             = "concatenate";
  oi.n_outputs      = 1;
  oi.input_descs    = in;
  oi.input_bytes    = data;
  oi.settings       = axis_settings(0);
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(ConcatenateUpdateTest, RecreatesOnChangedAxis) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32), device_desc({2, 2}, DType::F32)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f}),
      as_bytes(std::vector<float>{10.f, 20.f, 30.f, 40.f}),
  };

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  const auto                          j_old = axis_settings(0);
  const auto                          j_new = axis_settings(1);

  auto task = factory.create(in, j_old, create_ctx);
  task      = factory.update(std::move(task), in, j_new, create_ctx);

  const auto infer = factory.infer(in, j_new);

  std::vector<holonp_test::TensorTestBuffer> in_bufs;
  for (size_t i = 0; i < in.size(); ++i) {
    in_bufs.emplace_back(in[i]);
    in_bufs.back().upload(data[i]);
  }
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);

  std::vector<holoflow::core::TView> in_views;
  for (auto &buf : in_bufs) {
    in_views.push_back(buf.view());
  }
  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = in_views,
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  holonp_test::OracleInput oi;
  oi.op             = "concatenate";
  oi.n_outputs      = 1;
  oi.input_descs    = in;
  oi.input_bytes    = data;
  oi.settings       = j_new;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  const auto actual = out_buf.download();
  expect_near_oracle(actual, oracle.output_bytes[0], DType::F32);
}

TEST_F(ConcatenateUpdateTest, RecreatesOnWrongTaskType) {
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const std::vector<TDesc> in = {device_desc({2, 2}, DType::U8), device_desc({1, 2}, DType::U8)};
  const std::vector<std::vector<std::byte>> data = {
      as_bytes(std::vector<std::uint8_t>{1, 2, 3, 4}),
      as_bytes(std::vector<std::uint8_t>{5, 6}),
  };

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};
  const auto                          settings = axis_settings(0);

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), in, settings, create_ctx);

  const auto infer = factory.infer(in, settings);

  std::vector<holonp_test::TensorTestBuffer> in_bufs;
  for (size_t i = 0; i < in.size(); ++i) {
    in_bufs.emplace_back(in[i]);
    in_bufs.back().upload(data[i]);
  }
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);

  std::vector<holoflow::core::TView> in_views;
  for (auto &buf : in_bufs) {
    in_views.push_back(buf.view());
  }
  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = in_views,
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  holonp_test::OracleInput oi;
  oi.op             = "concatenate";
  oi.n_outputs      = 1;
  oi.input_descs    = in;
  oi.input_bytes    = data;
  oi.settings       = settings;
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  const auto actual = out_buf.download();
  expect_near_oracle(actual, oracle.output_bytes[0], DType::U8);
}
