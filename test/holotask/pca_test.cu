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
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/pca.hh"

#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype = DType::F32) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<float> as_floats(const std::vector<std::byte> &bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

nlohmann::json settings(int begin, int end, int window_size = 0) {
  return holotask::syncs::PcaSettings{
      .begin       = begin,
      .end         = end,
      .window_size = window_size,
  };
}

float dot(const float *lhs, const float *rhs, size_t size) {
  float result = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    result += lhs[i] * rhs[i];
  }
  return result;
}

std::vector<float> covariance_two_by_two_input(float scale = 1.0f) {
  const float sqrt_two         = std::sqrt(2.0f);
  const float inverse_sqrt_two = 1.0f / sqrt_two;
  const float sqrt_three_half  = std::sqrt(1.5f);
  return {
      scale * sqrt_two,
      0.0f,
      scale * inverse_sqrt_two,
      scale * sqrt_three_half,
  };
}

std::vector<float> diagonal_covariance_input(size_t depth, float scale = 1.0f) {
  std::vector<float> input(depth * depth, 0.0f);
  for (size_t feature = 0; feature < depth; ++feature) {
    input[feature * depth + feature] = scale * std::sqrt(static_cast<float>(feature + 1));
  }
  return input;
}

std::vector<float> execute_once(holoflow::core::ISyncTask     &task,
                                holonp_test::TensorTestBuffer &input,
                                holonp_test::TensorTestBuffer &output, cudaStream_t stream) {
  std::vector<holoflow::core::TView> inputs  = {input.view()};
  std::vector<holoflow::core::TView> outputs = {output.view()};
  std::atomic<bool>                  cancelled{false};
  holoflow::core::SyncCtx            ctx{
      .inputs       = inputs,
      .outputs      = outputs,
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  EXPECT_EQ(task.execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return as_floats(output.download());
}

} // namespace

class PcaInferTest : public ::testing::Test {
protected:
  holotask::syncs::PcaFactory factory;
};

TEST_F(PcaInferTest, AcceptsF32AndPreservesBatchShape) {
  const std::vector<TDesc> inputs = {device_desc({3, 4, 5, 6})};
  const auto               result = factory.infer(inputs, settings(1, 3));

  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{3, 2, 5, 6}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
}

TEST_F(PcaInferTest, RejectsComplexInput) {
  const std::vector<TDesc> inputs = {device_desc({2, 1, 2}, DType::CF32)};
  EXPECT_THROW(factory.infer(inputs, settings(0, 2)), std::invalid_argument);
}

TEST_F(PcaInferTest, LocalPcaRejectsEvenSpatialWindow) {
  const std::vector<TDesc> inputs = {device_desc({2, 3, 3})};
  EXPECT_THROW(factory.infer(inputs, settings(0, 2, 4)), std::invalid_argument);
}

class PcaExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::PcaFactory factory;
};

TEST_F(PcaExecuteTest, LocalPcaWrapsBordersAndProjectsCenterPixel) {
  // A 9x9 window on this 3x3 image covers every pixel nine times by wrapping the borders.
  // The covariance has eigenvalues 45 and 81. Eigenvector signs are arbitrary, so compare squares.
  std::vector<float> input(2 * 3 * 3, 0.0f);
  input[0]     = 1.0f; // T0 at top-left
  input[4]     = 2.0f; // T0 at center
  input[9 + 8] = 3.0f; // T1 at bottom-right

  const TDesc input_desc = device_desc({2, 3, 3});
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(0, 2, 9));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 18);
  EXPECT_NEAR(output[0] * output[0], 1.0f, 2e-2f);
  EXPECT_NEAR(output[9] * output[9], 0.0f, 2e-2f);
  EXPECT_NEAR(output[4] * output[4], 4.0f, 2e-2f);
  EXPECT_NEAR(output[9 + 4] * output[9 + 4], 0.0f, 2e-2f);
}

TEST_F(PcaExecuteTest, LocalPcaPreservesLeadingBatches) {
  const TDesc input_desc = device_desc({2, 2, 1, 1});
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc},
      std::vector<std::vector<std::byte>>{as_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 2.0f})},
      settings(1, 2, 3));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 2);
  EXPECT_NEAR(output[0] * output[0], 1.0f, 2e-2f);
  EXPECT_NEAR(output[1] * output[1], 4.0f, 2e-2f);
}

TEST_F(PcaExecuteTest, LocalPcaCancelledExecutionCanRetrySameInput) {
  const TDesc input_desc = device_desc({2, 3, 3});
  const auto config = settings(0, 2, 3);
  const auto inference = factory.infer(std::array{input_desc}, config);
  auto task = factory.create(std::array{input_desc}, config, {.stream = nullptr});
  task->bind_logger(spdlog::default_logger());
  holonp_test::TensorTestBuffer input_buffer(input_desc);
  holonp_test::TensorTestBuffer output_buffer(inference.output_descs[0]);
  input_buffer.upload(as_bytes(std::vector<float>(18, 1.0f)));
  std::array inputs{input_buffer.view()};
  std::array outputs{output_buffer.view()};
  std::atomic<bool> cancelled{true};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};
  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Cancelled);
  cancelled.store(false);
  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
}

TEST_F(PcaExecuteTest, ProducesOrthogonalComponentsWithExpectedEigenvalueEnergy) {
  const TDesc input_desc = device_desc({2, 1, 2});
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc},
      std::vector<std::vector<std::byte>>{as_bytes(covariance_two_by_two_input())}, settings(0, 2));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 4);
  EXPECT_NEAR(dot(output.data(), output.data(), 2), 1.0f, 2e-2f);
  EXPECT_NEAR(dot(output.data() + 2, output.data() + 2, 2), 3.0f, 2e-2f);
  EXPECT_NEAR(dot(output.data(), output.data() + 2, 2), 0.0f, 2e-2f);
}

TEST_F(PcaExecuteTest, SelectsNonzeroBeginAcrossBatchedAndTailKernels) {
  constexpr size_t   batches = 33;
  std::vector<float> input;
  input.reserve(batches * 4);
  for (size_t batch = 0; batch < batches; ++batch) {
    const auto values = covariance_two_by_two_input(batch + 1 == batches ? 2.0f : 1.0f);
    input.insert(input.end(), values.begin(), values.end());
  }

  const TDesc input_desc = device_desc({batches, 2, 1, 2});
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(1, 2));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), batches * 2);
  for (size_t batch = 0; batch + 1 < batches; ++batch) {
    EXPECT_NEAR(dot(output.data() + batch * 2, output.data() + batch * 2, 2), 3.0f, 2e-2f);
  }
  EXPECT_NEAR(dot(output.data() + (batches - 1) * 2, output.data() + (batches - 1) * 2, 2), 12.0f,
              5e-2f);
}

TEST_F(PcaExecuteTest, ProducesOrthogonalComponentsForEveryBatch) {
  constexpr size_t   batches = 3;
  std::vector<float> input;
  input.reserve(batches * 4);
  for (size_t batch = 0; batch < batches; ++batch) {
    const auto values = covariance_two_by_two_input(static_cast<float>(batch + 1));
    input.insert(input.end(), values.begin(), values.end());
  }

  const TDesc input_desc = device_desc({batches, 2, 1, 2});
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(0, 2));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), batches * 4);
  for (size_t batch = 0; batch < batches; ++batch) {
    const auto *batch_output = output.data() + batch * 4;
    const auto  scale_sq     = static_cast<float>((batch + 1) * (batch + 1));
    EXPECT_NEAR(dot(batch_output, batch_output, 2), scale_sq, 2e-2f * scale_sq);
    EXPECT_NEAR(dot(batch_output + 2, batch_output + 2, 2), 3.0f * scale_sq, 2e-2f * scale_sq);
    EXPECT_NEAR(dot(batch_output, batch_output + 2, 2), 0.0f, 2e-2f * scale_sq);
  }
}

TEST_F(PcaExecuteTest, SupportsDepthsAtAndAboveCusolverFallbackBoundary) {
  for (const size_t depth : {size_t{256}, size_t{512}}) {
    const TDesc input_desc = device_desc({depth, 1, depth});
    const auto  result     = holonp_test::run_sync_factory(
        factory, std::vector<TDesc>{input_desc},
        std::vector<std::vector<std::byte>>{as_bytes(diagonal_covariance_input(depth))},
        settings(static_cast<int>(depth - 1), static_cast<int>(depth)));

    const auto output = as_floats(result.output_bytes[0]);
    ASSERT_EQ(output.size(), depth);
    EXPECT_NEAR(dot(output.data(), output.data(), output.size()), static_cast<float>(depth),
                5e-2f * static_cast<float>(depth));
  }
}

TEST_F(PcaExecuteTest, RecomputesEigenvectorsOnEveryExecution) {
  const TDesc input_desc = device_desc({2, 1, 2});
  const auto  inferred   = factory.infer(std::vector<TDesc>{input_desc}, settings(0, 2));

  curaii::CudaStream stream;
  auto task = factory.create(std::vector<TDesc>{input_desc}, settings(0, 2), {stream.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(inferred.output_descs[0]);

  input.upload(as_bytes(covariance_two_by_two_input()));
  (void)execute_once(*task, input, output, stream.get());

  input.upload(as_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 1.0f}));
  const auto updated = execute_once(*task, input, output, stream.get());
  ASSERT_EQ(updated.size(), 4);
  EXPECT_NEAR(dot(updated.data(), updated.data(), 2), 1.0f, 2e-2f);
  EXPECT_NEAR(dot(updated.data() + 2, updated.data() + 2, 2), 9.0f, 3e-2f);
}

TEST_F(PcaExecuteTest, CudaGraphReplaysCusolverFallbackWithCurrentData) {
  constexpr size_t depth      = 256;
  const TDesc      input_desc = device_desc({depth, 1, depth});
  const auto       inferred   = factory.infer(std::array{input_desc}, settings(depth - 1, depth));

  curaii::CudaStream stream;
  auto task = factory.create(std::array{input_desc}, settings(depth - 1, depth), {stream.get()});
  task->bind_logger(spdlog::default_logger());
  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(inferred.output_descs[0]);

  input.upload(as_bytes(diagonal_covariance_input(depth)));
  (void)execute_once(*task, input, output, stream.get());

  input.upload(as_bytes(diagonal_covariance_input(depth, 2.0f)));
  const auto replayed = execute_once(*task, input, output, stream.get());
  EXPECT_NEAR(dot(replayed.data(), replayed.data(), replayed.size()), 4.0f * depth, 5e-2f * depth);
}

TEST_F(PcaExecuteTest, CudaGraphCachesRotatingInputAndOutputBuffers) {
  const TDesc input_desc = device_desc({2, 1, 2});
  const auto  inferred   = factory.infer(std::array{input_desc}, settings(0, 2));

  curaii::CudaStream stream;
  auto               task = factory.create(std::array{input_desc}, settings(0, 2), {stream.get()});
  task->bind_logger(spdlog::default_logger());
  holonp_test::TensorTestBuffer input_a(input_desc);
  holonp_test::TensorTestBuffer input_b(input_desc);
  holonp_test::TensorTestBuffer output_a(inferred.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(inferred.output_descs[0]);

  input_a.upload(as_bytes(covariance_two_by_two_input()));
  input_b.upload(as_bytes(covariance_two_by_two_input(2.0f)));
  (void)execute_once(*task, input_a, output_a, stream.get());
  (void)execute_once(*task, input_b, output_b, stream.get());

  input_a.upload(as_bytes(std::vector<float>{3.0f, 0.0f, 0.0f, 1.0f}));
  input_b.upload(as_bytes(std::vector<float>{4.0f, 0.0f, 0.0f, 2.0f}));
  const auto replayed_a = execute_once(*task, input_a, output_a, stream.get());
  const auto replayed_b = execute_once(*task, input_b, output_b, stream.get());
  EXPECT_NEAR(dot(replayed_a.data() + 2, replayed_a.data() + 2, 2), 9.0f, 3e-2f);
  EXPECT_NEAR(dot(replayed_b.data() + 2, replayed_b.data() + 2, 2), 16.0f, 5e-2f);
}

TEST_F(PcaExecuteTest, ParticipatesInOuterCaptureForDxAndCusolverPaths) {
  for (const size_t depth : {size_t{2}, size_t{256}}) {
    const TDesc        input_desc = device_desc({depth, 1, depth});
    const auto         config     = settings(static_cast<int>(depth - 1), static_cast<int>(depth));
    const auto         inferred   = factory.infer(std::array{input_desc}, config);
    curaii::CudaStream stream;
    auto               task = factory.create(std::array{input_desc}, config, {stream.get()});
    task->bind_logger(spdlog::default_logger());
    holonp_test::TensorTestBuffer input(input_desc);
    holonp_test::TensorTestBuffer output(inferred.output_descs[0]);
    input.upload(as_bytes(diagonal_covariance_input(depth)));
    std::array              inputs{input.view()};
    std::array              outputs{output.view()};
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
    CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    const auto captured = as_floats(output.download());
    EXPECT_NEAR(dot(captured.data(), captured.data(), captured.size()), static_cast<float>(depth),
                5e-2f * depth);
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }
}

TEST_F(PcaExecuteTest, FallsBackOnDefaultStream) {
  constexpr size_t depth      = 256;
  const TDesc      input_desc = device_desc({depth, 1, depth});
  const auto       config     = settings(depth - 1, depth);
  const auto       inferred   = factory.infer(std::array{input_desc}, config);
  auto             task       = factory.create(std::array{input_desc}, config, {.stream = nullptr});
  task->bind_logger(spdlog::default_logger());
  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(inferred.output_descs[0]);
  input.upload(as_bytes(diagonal_covariance_input(depth)));
  std::array              inputs{input.view()};
  std::array              outputs{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};

  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto direct = as_floats(output.download());
  EXPECT_NEAR(dot(direct.data(), direct.data(), direct.size()), static_cast<float>(depth),
              5e-2f * depth);
}

TEST_F(PcaExecuteTest, ReusesCompiledTaskWhenComponentSelectionChanges) {
  const std::vector<TDesc> input_descs = {device_desc({2, 1, 2})};
  const auto               selected    = factory.infer(input_descs, settings(1, 2));

  curaii::CudaStream stream;
  auto               task = factory.create(input_descs, settings(0, 2), {stream.get()});
  task->bind_logger(spdlog::default_logger());
  const auto *original_task = task.get();

  task = factory.update(std::move(task), input_descs, settings(1, 2), {stream.get()});
  ASSERT_EQ(task.get(), original_task);

  holonp_test::TensorTestBuffer input(input_descs[0]);
  holonp_test::TensorTestBuffer output(selected.output_descs[0]);
  input.upload(as_bytes(covariance_two_by_two_input()));

  const auto projected = execute_once(*task, input, output, stream.get());
  ASSERT_EQ(projected.size(), 2);
  EXPECT_NEAR(dot(projected.data(), projected.data(), projected.size()), 3.0f, 2e-2f);

  task = factory.update(std::move(task), input_descs, settings(0, 1), {stream.get()});
  EXPECT_EQ(task.get(), original_task);
  const auto reconfigured = execute_once(*task, input, output, stream.get());
  ASSERT_EQ(reconfigured.size(), 2);
  EXPECT_NEAR(dot(reconfigured.data(), reconfigured.data(), reconfigured.size()), 1.0f, 2e-2f);
}
