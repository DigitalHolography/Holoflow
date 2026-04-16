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
#include "holonp/add.hh"

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
    // CF32 stores interleaved (real, imag) float pairs; compare each component.
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
// AddFactory: inference tests
// -------------------------------------------------------------------------------------------------

class AddInferTest : public ::testing::Test {
protected:
  holonp::AddFactory factory;
};

TEST_F(AddInferTest, F32SameShape) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32), device_desc({4}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.input_descs.size(), 2u);
  ASSERT_EQ(r.output_descs.size(), 1u);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
  ASSERT_EQ(r.owned_inputs.size(), 2u);
  EXPECT_EQ(r.owned_inputs[0], false);
  EXPECT_EQ(r.owned_inputs[1], false);
  ASSERT_EQ(r.owned_outputs.size(), 1u);
  EXPECT_EQ(r.owned_outputs[0], false);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(AddInferTest, U8SameShape) {
  const std::vector<TDesc> in = {device_desc({8}, DType::U8), device_desc({8}, DType::U8)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::U8);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{8}));
}

TEST_F(AddInferTest, U16SameShape) {
  const std::vector<TDesc> in = {device_desc({6}, DType::U16), device_desc({6}, DType::U16)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{6}));
}

TEST_F(AddInferTest, CF32SameShape) {
  const std::vector<TDesc> in = {device_desc({5}, DType::CF32), device_desc({5}, DType::CF32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{5}));
  EXPECT_EQ(r.output_descs[0].mem_loc, MemLoc::Device);
}

TEST_F(AddInferTest, F32TwoDimSameShape) {
  const std::vector<TDesc> in = {device_desc({2, 3}, DType::F32), device_desc({2, 3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(AddInferTest, BroadcastScalarToVector) {
  // (4,) + (1,) → (4,)
  const std::vector<TDesc> in = {device_desc({4}, DType::F32), device_desc({1}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
}

TEST_F(AddInferTest, BroadcastRowToMatrix) {
  // (3,3) + (1,3) → (3,3)
  const std::vector<TDesc> in = {device_desc({3, 3}, DType::F32), device_desc({1, 3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3, 3}));
}

TEST_F(AddInferTest, BroadcastLowerRankToHigher) {
  // (3,) broadcast into (2,3): a has ndim=1 padded to (1,3), b has shape (2,3) → output (2,3)
  const std::vector<TDesc> in = {device_desc({3}, DType::F32), device_desc({2, 3}, DType::F32)};
  const auto               r  = factory.infer(in, nlohmann::json::object());

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 3}));
}

TEST_F(AddInferTest, RejectsZeroInputs) {
  const std::vector<TDesc> in;
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AddInferTest, RejectsOneInput) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AddInferTest, RejectsThreeInputs) {
  const TDesc              d  = device_desc({4}, DType::F32);
  const std::vector<TDesc> in = {d, d, d};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AddInferTest, RejectsMismatchedDtypes) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32), device_desc({4}, DType::U16)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AddInferTest, RejectsInput0InHostMemory) {
  const std::vector<TDesc> in = {TDesc({4}, DType::F32, MemLoc::Host),
                                 device_desc({4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

TEST_F(AddInferTest, RejectsInput1InHostMemory) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32),
                                 TDesc({4}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(in, nlohmann::json::object()), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// AddFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class AddOracleTest : public ::testing::Test {
protected:
  holonp::AddFactory factory;

  // Helper: run factory + oracle and compare two same-shape contiguous device tensors.
  template <typename T>
  void check(DType dtype, const std::vector<size_t> &shape_a, const std::vector<T> &host_a,
             const std::vector<size_t> &shape_b, const std::vector<T> &host_b) {
    const TDesc                               da          = device_desc(shape_a, dtype);
    const TDesc                               db          = device_desc(shape_b, dtype);
    const auto                                ibytes_a    = as_bytes(host_a);
    const auto                                ibytes_b    = as_bytes(host_b);
    const std::vector<TDesc>                  input_descs = {da, db};
    const std::vector<std::vector<std::byte>> input_data  = {ibytes_a, ibytes_b};

    const auto run =
        holonp_test::run_sync_factory(factory, input_descs, input_data, nlohmann::json::object());

    holonp_test::OracleInput oi;
    oi.op             = "add";
    oi.n_outputs      = 1;
    oi.input_descs    = {da, db};
    oi.input_bytes    = {ibytes_a, ibytes_b};
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

    ASSERT_EQ(run.output_bytes.size(), 1u);
    ASSERT_EQ(oracle.output_bytes.size(), 1u);
    expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], dtype);
  }
};

TEST_F(AddOracleTest, F32SameShape) {
  check(DType::F32, {4}, std::vector<float>{1.0f, -2.0f, 3.5f, 0.0f}, {4},
        std::vector<float>{0.5f, 1.0f, -1.5f, 2.0f});
}

TEST_F(AddOracleTest, F32TwoDim) {
  check(DType::F32, {2, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3},
        std::vector<float>{-1.f, -2.f, -3.f, -4.f, -5.f, -6.f});
}

TEST_F(AddOracleTest, F32BroadcastScalarToVector) {
  check(DType::F32, {4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {1},
        std::vector<float>{10.0f});
}

TEST_F(AddOracleTest, F32BroadcastRowToMatrix) {
  check(DType::F32, {3, 3}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {1, 3},
        std::vector<float>{10.f, 20.f, 30.f});
}

TEST_F(AddOracleTest, U8SameShape) {
  check(DType::U8, {4}, std::vector<std::uint8_t>{0, 10, 100, 200}, {4},
        std::vector<std::uint8_t>{1, 20, 50, 55});
}

TEST_F(AddOracleTest, U16SameShape) {
  check(DType::U16, {4}, std::vector<std::uint16_t>{0, 100, 1000, 60000}, {4},
        std::vector<std::uint16_t>{1, 200, 500, 5000});
}

TEST_F(AddOracleTest, CF32SameShape) {
  struct CF32 {
    float re, im;
  };
  const std::vector<CF32> a_data = {{1.f, 2.f}, {-3.f, 4.f}, {0.f, 0.f}, {5.f, -1.f}};
  const std::vector<CF32> b_data = {{2.f, -1.f}, {1.f, 1.f}, {3.f, 3.f}, {-2.f, 2.f}};

  const TDesc da       = device_desc({4}, DType::CF32);
  const TDesc db       = device_desc({4}, DType::CF32);
  const auto  ibytes_a = as_bytes(a_data);
  const auto  ibytes_b = as_bytes(b_data);

  const std::vector<TDesc>                  input_descs = {da, db};
  const std::vector<std::vector<std::byte>> input_data  = {ibytes_a, ibytes_b};

  const auto run =
      holonp_test::run_sync_factory(factory, input_descs, input_data, nlohmann::json::object());

  holonp_test::OracleInput oi;
  oi.op             = "add";
  oi.n_outputs      = 1;
  oi.input_descs    = {da, db};
  oi.input_bytes    = {ibytes_a, ibytes_b};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  ASSERT_EQ(oracle.output_bytes.size(), 1u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::CF32);
}

// -------------------------------------------------------------------------------------------------
// AddFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class AddUpdateTest : public ::testing::Test {
protected:
  holonp::AddFactory factory;
};

TEST_F(AddUpdateTest, ReusesAddTask) {
  // update() with an existing Add task and same descriptors should reuse it.
  const std::vector<float>                  a_data      = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float>                  b_data      = {4.0f, 3.0f, 2.0f, 1.0f};
  const TDesc                               idesc       = device_desc({4}, DType::F32);
  const auto                                abytes      = as_bytes(a_data);
  const auto                                bbytes      = as_bytes(b_data);
  const std::vector<TDesc>                  input_descs = {idesc, idesc};
  const std::vector<std::vector<std::byte>> input_data  = {abytes, bbytes};
  const nlohmann::json                      settings    = nlohmann::json::object();

  const auto run = holonp_test::run_sync_factory_update(factory, input_descs, input_data, settings);

  holonp_test::OracleInput oi;
  oi.op             = "add";
  oi.n_outputs      = 1;
  oi.input_descs    = {idesc, idesc};
  oi.input_bytes    = {abytes, bbytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  ASSERT_EQ(run.output_bytes.size(), 1u);
  expect_near_oracle(run.output_bytes[0], oracle.output_bytes[0], DType::F32);
}

TEST_F(AddUpdateTest, RecreatesOnWrongTaskType) {
  // Passing a non-Add ISyncTask to update() must trigger the create path and still work.
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const std::vector<float> a_data   = {1.0f, -2.0f, 3.0f};
  const std::vector<float> b_data   = {-1.0f, 2.0f, -3.0f};
  const TDesc              idesc    = device_desc({3}, DType::F32);
  const auto               abytes   = as_bytes(a_data);
  const auto               bbytes   = as_bytes(b_data);
  const std::vector<TDesc> descs    = {idesc, idesc};
  const nlohmann::json     settings = nlohmann::json::object();

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});
  auto task = factory.update(std::move(fake), descs, settings, create_ctx);

  const auto infer = factory.infer(descs, settings);

  holonp_test::TensorTestBuffer in_a(idesc);
  holonp_test::TensorTestBuffer in_b(idesc);
  in_a.upload(abytes);
  in_b.upload(bbytes);
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);

  auto                                 va       = in_a.view();
  auto                                 vb       = in_b.view();
  auto                                 ov       = out_buf.view();
  std::array<holoflow::core::TView, 2> in_views = {va, vb};
  std::atomic<bool>                    cancelled{false};
  holoflow::core::SyncCtx              ctx{
      .inputs       = {in_views.data(), in_views.size()},
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  const auto actual = out_buf.download();

  holonp_test::OracleInput oi;
  oi.op             = "add";
  oi.n_outputs      = 1;
  oi.input_descs    = {idesc, idesc};
  oi.input_bytes    = {abytes, bbytes};
  const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

  expect_near_oracle(actual, oracle.output_bytes[0], DType::F32);
}
