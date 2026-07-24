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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <cuComplex.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"

#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::SyncCreateCtx;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

nlohmann::json fresnel_settings(float z) {
  return holotask::syncs::FresnelDiffractionSettings{
      .lambda           = 852e-9f,
      .dx               = 20e-6f,
      .dy               = 20e-6f,
      .z                = z,
      .axes             = {-2, -1},
      .skip_phase_shift = false,
  };
}

nlohmann::json short_time_fresnel_settings(float z) {
  return holotask::syncs::ShortTimeFresnelDiffractionSettings{
      .lambda           = 852e-9f,
      .dx               = 20e-6f,
      .dy               = 20e-6f,
      .z                = z,
      .win_h            = 4,
      .win_w            = 4,
      .stride_y         = 2,
      .stride_x         = 2,
      .phase_ref        = holotask::syncs::STFDPhaseReference::LOCAL,
      .skip_phase_shift = false,
      .axes             = {-2, -1},
  };
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

struct TaskResult {
  TDesc                  desc;
  std::vector<std::byte> bytes;
};

TaskResult run_task(const holoflow::core::ISyncTaskFactory &factory, const TDesc &input_desc,
                    std::span<const std::byte> input_bytes, const nlohmann::json &settings) {
  curaii::CudaStream stream;
  const std::vector  input_descs = {input_desc};
  const auto         infer       = factory.infer(input_descs, settings);
  auto               task        = factory.create(input_descs, settings, {stream.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output(infer.output_descs[0]);
  input.upload(input_bytes);

  auto              input_view  = input.view();
  auto              output_view = output.view();
  std::atomic<bool> cancelled{false};
  holoflow::core::SyncCtx execute_ctx{
      .inputs       = {&input_view, 1},
      .outputs      = {&output_view, 1},
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };
  EXPECT_EQ(task->execute(execute_ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  return {infer.output_descs[0], output.download()};
}

void expect_magnitude_matches(const TaskResult &complex_result, const TaskResult &magnitude_result) {
  ASSERT_EQ(complex_result.desc.dtype, DType::CF32);
  ASSERT_EQ(magnitude_result.desc.dtype, DType::F32);
  ASSERT_EQ(complex_result.desc.shape, magnitude_result.desc.shape);

  const auto count = magnitude_result.bytes.size() / sizeof(float);
  ASSERT_EQ(complex_result.bytes.size(), count * sizeof(cuFloatComplex));

  std::vector<cuFloatComplex> complex_values(count);
  std::vector<float>          magnitude_values(count);
  std::memcpy(complex_values.data(), complex_result.bytes.data(), complex_result.bytes.size());
  std::memcpy(magnitude_values.data(), magnitude_result.bytes.data(), magnitude_result.bytes.size());

  for (size_t i = 0; i < count; ++i) {
    const float expected = std::hypot(complex_values[i].x, complex_values[i].y);
    EXPECT_NEAR(magnitude_values[i], expected, 1e-4f * std::max(1.0f, expected)) << "index " << i;
  }
}

} // namespace

TEST(FresnelDiffractionMagnitudeTest, DefaultsToComplexAndSerializesOptIn) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const std::vector<TDesc> input = {device_desc({2, 4, 4}, DType::CF32)};

  auto settings = fresnel_settings(0.01f);
  EXPECT_FALSE(settings.get<holotask::syncs::FresnelDiffractionSettings>().output_magnitude);
  EXPECT_EQ(factory.infer(input, settings).output_descs[0].dtype, DType::CF32);

  settings["output_magnitude"] = true;
  EXPECT_TRUE(settings.get<holotask::syncs::FresnelDiffractionSettings>().output_magnitude);
  EXPECT_EQ(factory.infer(input, settings).output_descs[0].dtype, DType::F32);
}

TEST(FresnelDiffractionMagnitudeTest, StoreCallbackMatchesComplexMagnitude) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const auto input_desc = device_desc({2, 3, 32, 48}, DType::CF32);
  std::vector<cuFloatComplex> input_values(input_desc.num_elements());
  for (size_t i = 0; i < input_values.size(); ++i)
    input_values[i] = make_cuFloatComplex(static_cast<float>(i % 7) - 3.0f,
                                          static_cast<float>(i % 5) - 2.0f);

  auto complex_settings = fresnel_settings(0.01f);
  auto magnitude_settings = complex_settings;
  magnitude_settings["output_magnitude"] = true;

  const auto input_bytes = as_bytes(input_values);
  expect_magnitude_matches(run_task(factory, input_desc, input_bytes, complex_settings),
                           run_task(factory, input_desc, input_bytes, magnitude_settings));
}

TEST(ShortTimeFresnelDiffractionMagnitudeTest, StoreCallbackMatchesComplexMagnitude) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const auto input_desc = device_desc({2, 3, 32, 32}, DType::CF32);
  std::vector<cuFloatComplex> input_values(input_desc.num_elements());
  for (size_t i = 0; i < input_values.size(); ++i)
    input_values[i] = make_cuFloatComplex(static_cast<float>(i % 11) - 5.0f,
                                          static_cast<float>(i % 3) - 1.0f);

  auto complex_settings = short_time_fresnel_settings(0.01f);
  EXPECT_EQ(factory.infer({&input_desc, 1}, complex_settings).output_descs[0].dtype, DType::CF32);
  auto magnitude_settings = complex_settings;
  magnitude_settings["output_magnitude"] = true;
  EXPECT_EQ(factory.infer({&input_desc, 1}, magnitude_settings).output_descs[0].dtype, DType::F32);

  const auto input_bytes = as_bytes(input_values);
  expect_magnitude_matches(run_task(factory, input_desc, input_bytes, complex_settings),
                           run_task(factory, input_desc, input_bytes, magnitude_settings));
}

TEST(FresnelDiffractionUpdateTest, ReusesTaskWhenOnlyPropagationDistanceChanges) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const std::vector<TDesc>                   input = {device_desc({4, 4}, DType::CF32)};

  for (const bool output_magnitude : {false, true}) {
    SCOPED_TRACE(output_magnitude);
    curaii::CudaStream  stream;
    const SyncCreateCtx ctx{stream.get()};

    auto initial_settings = fresnel_settings(0.01f);
    initial_settings["output_magnitude"] = output_magnitude;
    auto updated_settings = fresnel_settings(0.02f);
    updated_settings["output_magnitude"] = output_magnitude;

    auto task = factory.create(input, initial_settings, ctx);
    task->bind_logger(spdlog::default_logger());
    auto *raw = task.get();

    task = factory.update(std::move(task), input, updated_settings, ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    EXPECT_EQ(task.get(), raw);
  }
}

TEST(ShortTimeFresnelDiffractionUpdateTest, ReusesTaskWhenOnlyPropagationDistanceChanges) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const std::vector<TDesc>                            input = {device_desc({8, 8}, DType::CF32)};

  for (const bool output_magnitude : {false, true}) {
    SCOPED_TRACE(output_magnitude);
    curaii::CudaStream  stream;
    const SyncCreateCtx ctx{stream.get()};

    auto initial_settings = short_time_fresnel_settings(0.01f);
    initial_settings["output_magnitude"] = output_magnitude;
    auto updated_settings = short_time_fresnel_settings(0.02f);
    updated_settings["output_magnitude"] = output_magnitude;

    auto task = factory.create(input, initial_settings, ctx);
    task->bind_logger(spdlog::default_logger());
    auto *raw = task.get();

    task = factory.update(std::move(task), input, updated_settings, ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    EXPECT_EQ(task.get(), raw);
  }
}
