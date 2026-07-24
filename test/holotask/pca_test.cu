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

nlohmann::json settings(int begin, int end) {
  return holotask::syncs::PcaSettings{
      .begin = begin,
      .end   = end,
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

TEST_F(PcaInferTest, AcceptsU8AndProducesF32WithPreservedBatchShape) {
  const std::vector<TDesc> inputs = {device_desc({3, 4, 5, 6}, DType::U8)};
  const auto               result = factory.infer(inputs, settings(1, 3));

  ASSERT_EQ(result.output_descs.size(), 1);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<size_t>{3, 2, 5, 6}));
  EXPECT_EQ(result.output_descs[0].dtype, DType::F32);
}

TEST_F(PcaInferTest, RejectsComplexInput) {
  const std::vector<TDesc> inputs = {device_desc({2, 1, 2}, DType::CF32)};
  EXPECT_THROW(factory.infer(inputs, settings(0, 2)), std::invalid_argument);
}

TEST_F(PcaInferTest, RejectsU16Input) {
  const std::vector<TDesc> inputs = {device_desc({2, 1, 2}, DType::U16)};
  EXPECT_THROW(factory.infer(inputs, settings(0, 2)), std::invalid_argument);
}

class PcaExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::PcaFactory factory;
};

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

TEST_F(PcaExecuteTest, U8FusedPathMatchesF32ReferenceProjectionGramMatrix) {
  constexpr size_t          batches  = 3;
  constexpr size_t          features = 8;
  constexpr size_t          samples  = 128;
  std::vector<std::uint8_t> input_u8(batches * features * samples);
  std::vector<float>        input_f32(input_u8.size());
  for (size_t batch = 0; batch < batches; ++batch) {
    for (size_t feature = 0; feature < features; ++feature) {
      for (size_t sample = 0; sample < samples; ++sample) {
        const auto index = (batch * features + feature) * samples + sample;
        input_u8[index]  = static_cast<std::uint8_t>(
            (sample * (3 + 2 * feature) + batch * (5 + 3 * feature) + 7 * feature + 1) % 251);
        input_f32[index] = static_cast<float>(input_u8[index]);
      }
    }
  }

  const auto fused = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_desc({batches, features, 1, samples}, DType::U8)},
      std::vector<std::vector<std::byte>>{as_bytes(input_u8)},
      settings(0, static_cast<int>(features)));
  const auto reference = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_desc({batches, features, 1, samples}, DType::F32)},
      std::vector<std::vector<std::byte>>{as_bytes(input_f32)},
      settings(0, static_cast<int>(features)));

  const auto fused_output     = as_floats(fused.output_bytes[0]);
  const auto reference_output = as_floats(reference.output_bytes[0]);
  ASSERT_EQ(fused_output.size(), reference_output.size());

  for (size_t batch = 0; batch < batches; ++batch) {
    double error_sq = 0.0;
    double norm_sq  = 0.0;
    for (size_t lhs_sample = 0; lhs_sample < samples; ++lhs_sample) {
      for (size_t rhs_sample = 0; rhs_sample < samples; ++rhs_sample) {
        double actual_gram   = 0.0;
        double expected_gram = 0.0;
        for (size_t component = 0; component < features; ++component) {
          const auto offset = (batch * features + component) * samples;
          actual_gram += static_cast<double>(fused_output[offset + lhs_sample]) *
                         fused_output[offset + rhs_sample];
          expected_gram += static_cast<double>(reference_output[offset + lhs_sample]) *
                           reference_output[offset + rhs_sample];
        }
        const double diff = actual_gram - expected_gram;
        error_sq += diff * diff;
        norm_sq += expected_gram * expected_gram;
      }
    }
    EXPECT_LT(std::sqrt(error_sq / norm_sq), 1e-3);
  }
}

TEST_F(PcaExecuteTest, U8FusedPathHandlesSpatialAndComponentTails) {
  const std::vector<std::uint8_t> input = {
      1, 2, 3, 4, 5, 2, 1, 4, 3, 7,
  };
  const TDesc input_desc = device_desc({2, 1, 5}, DType::U8);
  const auto  result     = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      settings(1, 2));

  const auto output = as_floats(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 5);
  EXPECT_GT(dot(output.data(), output.data(), output.size()), 0.0f);
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
}
