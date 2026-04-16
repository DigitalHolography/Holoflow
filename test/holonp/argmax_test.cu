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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holonp/argmax.hh"

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

// Build ArgmaxSettings JSON: axis=null (reduce all), optional keepdims.
static nlohmann::json jsettings_all(bool keepdims = false) {
  return nlohmann::json{{"axis", nullptr}, {"keepdims", keepdims}};
}

// Build ArgmaxSettings JSON: single integer axis.
static nlohmann::json jsettings_axis(int axis, bool keepdims = false) {
  return nlohmann::json{{"axis", axis}, {"keepdims", keepdims}};
}

// Build ArgmaxSettings JSON: multiple axes.
static nlohmann::json jsettings_axes(std::vector<int> axes, bool keepdims = false) {
  return nlohmann::json{{"axis", axes}, {"keepdims", keepdims}};
}

// Compare actual bytes against expected U16 values (argmax always outputs U16).
static void expect_eq_u16(const std::vector<std::byte>     &actual,
                          const std::vector<std::uint16_t> &expected) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(std::uint16_t));
  const auto *a = reinterpret_cast<const std::uint16_t *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(a[i], expected[i]) << "at index " << i;
}

// Brute-force flat argmax for CF32 using the same lexicographic ordering as the kernel
// (compare real parts first, then imaginary parts on a tie).
struct CF32 {
  float re, im;
};
static std::uint16_t flat_argmax_cf32(const std::vector<CF32> &v) {
  std::uint16_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    const bool greater = (v[i].re > v[best].re) || (v[i].re == v[best].re && v[i].im > v[best].im);
    if (greater)
      best = static_cast<std::uint16_t>(i);
  }
  return best;
}

// -------------------------------------------------------------------------------------------------
// ArgmaxFactory: inference tests
// -------------------------------------------------------------------------------------------------

class ArgmaxInferTest : public ::testing::Test {
protected:
  holonp::ArgmaxFactory factory;
};

TEST_F(ArgmaxInferTest, RejectsRank0OutputForFlatReduceF32) {
  const std::vector<TDesc> in = {device_desc({4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsRank0OutputForFlatReduceU8) {
  const std::vector<TDesc> in = {device_desc({8}, DType::U8)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsRank0OutputForFlatReduceU16) {
  const std::vector<TDesc> in = {device_desc({6}, DType::U16)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsRank0OutputForFlatReduceCF32) {
  const std::vector<TDesc> in = {device_desc({5}, DType::CF32)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, Axis0of2D) {
  // Reduce along rows: (3,4) → {4}
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_axis(0));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{4}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ArgmaxInferTest, Axis1of2D) {
  // Reduce along columns: (3,4) → {3}
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_axis(1));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ArgmaxInferTest, KeepDimsAllAxes) {
  // (3,4), all axes, keepdims → {1,1}
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_all(/*keepdims=*/true));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1, 1}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ArgmaxInferTest, KeepDimsAxis0) {
  // (3,4), axis=0, keepdims → {1,4}
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_axis(0, /*keepdims=*/true));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{1, 4}));
}

TEST_F(ArgmaxInferTest, NegativeAxisNormalized) {
  // axis=-1 on (3,4) is equivalent to axis=1 → output {3}
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_axis(-1));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3}));
}

TEST_F(ArgmaxInferTest, MultiAxisReduction) {
  // (2,3,4), axes=[0,2] → {3}
  const std::vector<TDesc> in = {device_desc({2, 3, 4}, DType::F32)};
  const auto               r  = factory.infer(in, jsettings_axes({0, 2}));

  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{3}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::U16);
}

TEST_F(ArgmaxInferTest, RejectsRank0OutputWhenReducingAllAxesOf2D) {
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsRank0OutputWhenReducingAllAxesExplicitly) {
  const std::vector<TDesc> in = {device_desc({2, 3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_axes({0, 1, 2})), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsZeroInputs) {
  EXPECT_THROW(factory.infer({}, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsTwoInputs) {
  const TDesc              d  = device_desc({4}, DType::F32);
  const std::vector<TDesc> in = {d, d};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsHostMemory) {
  const std::vector<TDesc> in = {TDesc({4}, DType::F32, MemLoc::Host)};
  EXPECT_THROW(factory.infer(in, jsettings_all()), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsAxisOutOfRange) {
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_axis(2)), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsNegativeAxisOutOfRange) {
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_axis(-3)), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsDuplicateAxes) {
  const std::vector<TDesc> in = {device_desc({3, 4}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_axes({0, 0})), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsZeroElementReduction) {
  // Axis 1 has size 0 → zero-element reduction
  const std::vector<TDesc> in = {device_desc({3, 0}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_axis(1)), std::invalid_argument);
}

TEST_F(ArgmaxInferTest, RejectsReductionTooLarge) {
  // 70000 elements exceeds U16 max (65535)
  const std::vector<TDesc> in = {device_desc({70000}, DType::F32)};
  EXPECT_THROW(factory.infer(in, jsettings_all(/*keepdims=*/true)), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ArgmaxFactory: execution tests via NumPy oracle
// -------------------------------------------------------------------------------------------------

class ArgmaxOracleTest : public ::testing::Test {
protected:
  holonp::ArgmaxFactory factory;

  // Run factory + oracle and compare for a single contiguous device tensor.
  template <typename T>
  void check(DType dtype, const std::vector<size_t> &shape, const std::vector<T> &host_data,
             const nlohmann::json &jsettings) {
    const TDesc                               idesc       = device_desc(shape, dtype);
    const auto                                ibytes      = as_bytes(host_data);
    const std::vector<TDesc>                  input_descs = {idesc};
    const std::vector<std::vector<std::byte>> input_data  = {ibytes};

    const auto run = holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings);

    holonp_test::OracleInput oi;
    oi.op             = "argmax";
    oi.n_outputs      = 1;
    oi.input_descs    = {idesc};
    oi.input_bytes    = {ibytes};
    oi.settings       = jsettings;
    const auto oracle = holonp_test::invoke_oracle(oi, kOracleScript);

    ASSERT_EQ(run.output_bytes.size(), 1u);
    ASSERT_EQ(oracle.output_bytes.size(), 1u);
    ASSERT_EQ(run.output_bytes[0].size(), oracle.output_bytes[0].size());

    const size_t n = run.output_bytes[0].size() / sizeof(std::uint16_t);
    const auto  *a = reinterpret_cast<const std::uint16_t *>(run.output_bytes[0].data());
    const auto  *e = reinterpret_cast<const std::uint16_t *>(oracle.output_bytes[0].data());
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(a[i], e[i]) << "at index " << i;
  }
};

TEST_F(ArgmaxOracleTest, F32Axis0) {
  // Column-wise argmax: (3,4), axis=0 → {4}
  check(DType::F32, {3, 4},
        std::vector<float>{// row 0
                           1.0f, 8.0f, 3.0f, 2.0f,
                           // row 1
                           5.0f, 2.0f, 9.0f, 1.0f,
                           // row 2
                           4.0f, 6.0f, 0.0f, 7.0f},
        jsettings_axis(0));
}

TEST_F(ArgmaxOracleTest, F32Axis1) {
  // Row-wise argmax: (3,4), axis=1 → {3}
  check(DType::F32, {3, 4},
        std::vector<float>{1.0f, 8.0f, 3.0f, 2.0f, 5.0f, 2.0f, 9.0f, 1.0f, 4.0f, 6.0f, 0.0f, 7.0f},
        jsettings_axis(1));
}

TEST_F(ArgmaxOracleTest, F32NegativeAxis) {
  // axis=-1 on (3,4) is the same as axis=1.
  check(DType::F32, {3, 4},
        std::vector<float>{1.0f, 8.0f, 3.0f, 2.0f, 5.0f, 2.0f, 9.0f, 1.0f, 4.0f, 6.0f, 0.0f, 7.0f},
        jsettings_axis(-1));
}

TEST_F(ArgmaxOracleTest, F32KeepDimsAllAxes) {
  // (3,4) with keepdims → output shape {1,1}
  check(DType::F32, {3, 4},
        std::vector<float>{1.0f, 8.0f, 3.0f, 2.0f, 5.0f, 2.0f, 9.0f, 1.0f, 4.0f, 6.0f, 0.0f, 7.0f},
        jsettings_all(/*keepdims=*/true));
}

TEST_F(ArgmaxOracleTest, F32KeepDimsAxis1) {
  // (3,4), axis=1, keepdims → output shape {3,1}
  check(DType::F32, {3, 4},
        std::vector<float>{1.0f, 8.0f, 3.0f, 2.0f, 5.0f, 2.0f, 9.0f, 1.0f, 4.0f, 6.0f, 0.0f, 7.0f},
        jsettings_axis(1, /*keepdims=*/true));
}

TEST_F(ArgmaxOracleTest, RejectsRank0OutputForF32Flat1D) {
  const TDesc              idesc       = device_desc({5}, DType::F32);
  const auto               ibytes      = as_bytes(std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f});
  const std::vector<TDesc> input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data = {ibytes};

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings_all()),
      std::invalid_argument);
}

TEST_F(ArgmaxOracleTest, RejectsRank0OutputForF32Flat2D) {
  const TDesc idesc  = device_desc({2, 3}, DType::F32);
  const auto  ibytes = as_bytes(std::vector<float>{1.0f, 9.0f, 3.0f, 4.0f, 2.0f, 7.0f});
  const std::vector<TDesc>                  input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data  = {ibytes};

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings_all()),
      std::invalid_argument);
}

TEST_F(ArgmaxOracleTest, RejectsRank0OutputForU8Flat) {
  const TDesc              idesc  = device_desc({6}, DType::U8);
  const auto               ibytes = as_bytes(std::vector<std::uint8_t>{10, 50, 30, 200, 100, 150});
  const std::vector<TDesc> input_descs                 = {idesc};
  const std::vector<std::vector<std::byte>> input_data = {ibytes};

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings_all()),
      std::invalid_argument);
}

TEST_F(ArgmaxOracleTest, RejectsRank0OutputForU16Flat) {
  const TDesc              idesc  = device_desc({5}, DType::U16);
  const auto               ibytes = as_bytes(std::vector<std::uint16_t>{100, 500, 300, 1000, 200});
  const std::vector<TDesc> input_descs                 = {idesc};
  const std::vector<std::vector<std::byte>> input_data = {ibytes};

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings_all()),
      std::invalid_argument);
}

TEST_F(ArgmaxOracleTest, RejectsRank0OutputForCF32FlatManual) {
  const std::vector<CF32>  data        = {{1.f, 0.f}, {3.f, 1.f}, {3.f, 0.f}, {-2.f, 5.f}};
  const TDesc              idesc       = device_desc({4}, DType::CF32);
  const auto               ibytes      = as_bytes(data);
  const std::vector<TDesc> input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data = {ibytes};

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory(factory, input_descs, input_data, jsettings_all()),
      std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// ArgmaxFactory: update-path tests
// -------------------------------------------------------------------------------------------------

class ArgmaxUpdateTest : public ::testing::Test {
protected:
  holonp::ArgmaxFactory factory;
};

TEST_F(ArgmaxUpdateTest, RejectsRank0OutputInsteadOfReusingTask) {
  const std::vector<float>                  data        = {1.0f, 4.0f, 2.0f, 3.0f};
  const TDesc                               idesc       = device_desc({4}, DType::F32);
  const auto                                ibytes      = as_bytes(data);
  const std::vector<TDesc>                  input_descs = {idesc};
  const std::vector<std::vector<std::byte>> input_data  = {ibytes};
  const nlohmann::json                      settings    = jsettings_all();

  EXPECT_THROW(
      (void)holonp_test::run_sync_factory_update(factory, input_descs, input_data, settings),
      std::invalid_argument);
}

TEST_F(ArgmaxUpdateTest, RecreatesOnChangedAxis) {
  // Changing axis must discard the old task and produce results for the new configuration.
  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx ctx{stream.get()};

  const std::vector<float> data = {
      1.0f, 8.0f, 3.0f, 5.0f, 2.0f, 9.0f,
  };
  const TDesc idesc = device_desc({2, 3}, DType::F32);

  const std::vector<TDesc> descs = {idesc};

  // Create with axis=0, then update to axis=1.
  auto task = factory.create(descs, jsettings_axis(0), ctx);
  task      = factory.update(std::move(task), descs, jsettings_axis(1), ctx);

  const auto infer = factory.infer(descs, jsettings_axis(1));

  holonp_test::TensorTestBuffer in_buf(idesc);
  in_buf.upload(as_bytes(data));
  holonp_test::TensorTestBuffer out_buf(infer.output_descs[0]);

  auto                    iv = in_buf.view();
  auto                    ov = out_buf.view();
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx exec_ctx{
      .inputs       = {&iv, 1},
      .outputs      = {&ov, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  task->bind_logger(spdlog::default_logger());
  EXPECT_NO_THROW((void)task->execute(exec_ctx));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  // row 0: [1,8,3] → argmax=1;  row 1: [5,2,9] → argmax=2
  expect_eq_u16(out_buf.download(), {1u, 2u});
}

TEST_F(ArgmaxUpdateTest, RejectsRank0OutputWhenUpdatingWrongTaskType) {
  // Passing a non-Argmax ISyncTask to update() must still reject invalid rank-0 output
  // configurations.
  struct FakeTask : holoflow::core::ISyncTask {
    holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
      return holoflow::core::OpResult::Ok;
    }
  };

  const TDesc              idesc    = device_desc({5}, DType::F32);
  const std::vector<TDesc> descs    = {idesc};
  const nlohmann::json     settings = jsettings_all();

  curaii::CudaStream                  stream;
  const holoflow::core::SyncCreateCtx create_ctx{stream.get()};

  auto fake = std::unique_ptr<holoflow::core::ISyncTask>(new FakeTask{});

  EXPECT_THROW((void)factory.update(std::move(fake), descs, settings, create_ctx),
               std::invalid_argument);
}