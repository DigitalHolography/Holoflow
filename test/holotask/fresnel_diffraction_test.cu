#include <chrono>
#include <cstdio>
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
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../../src/holotask/src/syncs/fresnel_diffraction/fused_f16.hh"
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"

#include "tensor_test_buffer.hh"
#include "sync_task_runner.hh"
#include "holotask/syncs/correct_phase.hh"

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

  auto                    input_view  = input.view();
  auto                    output_view = output.view();
  std::atomic<bool>       cancelled{false};
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

void expect_magnitude_output_rebinding(const holoflow::core::ISyncTaskFactory &factory,
                                       const TDesc                            &input_desc,
                                       std::span<const std::byte>              input_bytes,
                                       const nlohmann::json                   &settings) {
  curaii::CudaStream stream;
  const std::vector  input_descs = {input_desc};
  const auto         infer       = factory.infer(input_descs, settings);
  auto               task        = factory.create(input_descs, settings, {stream.get()});
  task->bind_logger(spdlog::default_logger());

  holonp_test::TensorTestBuffer input(input_desc);
  holonp_test::TensorTestBuffer output_a(infer.output_descs[0]);
  holonp_test::TensorTestBuffer output_b(infer.output_descs[0]);
  input.upload(input_bytes);

  auto              input_view    = input.view();
  auto              output_view_a = output_a.view();
  auto              output_view_b = output_b.view();
  std::atomic<bool> cancelled{false};
  auto              execute = [&](holoflow::core::TView &output) {
    holoflow::core::SyncCtx execute_ctx{
        .inputs       = {&input_view, 1},
        .outputs      = {&output, 1},
        .cancelled    = &cancelled,
        .event_writer = nullptr,
        .event_reader = nullptr,
    };
    EXPECT_EQ(task->execute(execute_ctx), holoflow::core::OpResult::Ok);
  };

  execute(output_view_a);
  execute(output_view_a); // Stable pointer: caller info should already be current.
  execute(output_view_b); // New graph storage: caller info must be updated.
  execute(output_view_a); // Rebinding must also work when returning to an earlier buffer.
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  EXPECT_EQ(output_a.download(), output_b.download());
}

void expect_magnitude_matches(const TaskResult &complex_result,
                              const TaskResult &magnitude_result) {
  ASSERT_EQ(complex_result.desc.dtype, DType::CF32);
  ASSERT_EQ(magnitude_result.desc.dtype, DType::F32);
  ASSERT_EQ(complex_result.desc.shape, magnitude_result.desc.shape);

  const auto count = magnitude_result.bytes.size() / sizeof(float);
  ASSERT_EQ(complex_result.bytes.size(), count * sizeof(cuFloatComplex));

  std::vector<cuFloatComplex> complex_values(count);
  std::vector<float>          magnitude_values(count);
  std::memcpy(complex_values.data(), complex_result.bytes.data(), complex_result.bytes.size());
  std::memcpy(magnitude_values.data(), magnitude_result.bytes.data(),
              magnitude_result.bytes.size());

  for (size_t i = 0; i < count; ++i) {
    const float expected = std::hypot(complex_values[i].x, complex_values[i].y);
    EXPECT_NEAR(magnitude_values[i], expected, 1e-4f * std::max(1.0f, expected)) << "index " << i;
  }
}

} // namespace

TEST(FresnelDiffractionMagnitudeTest, DefaultsToComplexAndSerializesOptIn) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const std::vector<TDesc>                   input = {device_desc({2, 4, 4}, DType::CF32)};

  auto settings = fresnel_settings(0.01f);
  EXPECT_FALSE(settings.get<holotask::syncs::FresnelDiffractionSettings>().output_magnitude);
  EXPECT_EQ(factory.infer(input, settings).output_descs[0].dtype, DType::CF32);

  settings["output_magnitude"] = true;
  EXPECT_TRUE(settings.get<holotask::syncs::FresnelDiffractionSettings>().output_magnitude);
  EXPECT_EQ(factory.infer(input, settings).output_descs[0].dtype, DType::F32);
}

TEST(FresnelDiffractionMagnitudeTest, StoreCallbackMatchesComplexMagnitude) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const auto                                 input_desc = device_desc({2, 3, 32, 48}, DType::CF32);
  std::vector<cuFloatComplex>                input_values(input_desc.num_elements());
  for (size_t i = 0; i < input_values.size(); ++i)
    input_values[i] =
        make_cuFloatComplex(static_cast<float>(i % 7) - 3.0f, static_cast<float>(i % 5) - 2.0f);

  auto complex_settings                  = fresnel_settings(0.01f);
  auto magnitude_settings                = complex_settings;
  magnitude_settings["output_magnitude"] = true;

  const auto input_bytes = as_bytes(input_values);
  expect_magnitude_matches(run_task(factory, input_desc, input_bytes, complex_settings),
                           run_task(factory, input_desc, input_bytes, magnitude_settings));
}

TEST(ShortTimeFresnelDiffractionMagnitudeTest, StoreCallbackMatchesComplexMagnitude) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const auto                  input_desc = device_desc({2, 3, 32, 32}, DType::CF32);
  std::vector<cuFloatComplex> input_values(input_desc.num_elements());
  for (size_t i = 0; i < input_values.size(); ++i)
    input_values[i] =
        make_cuFloatComplex(static_cast<float>(i % 11) - 5.0f, static_cast<float>(i % 3) - 1.0f);

  auto complex_settings = short_time_fresnel_settings(0.01f);
  EXPECT_EQ(factory.infer({&input_desc, 1}, complex_settings).output_descs[0].dtype, DType::CF32);
  auto magnitude_settings                = complex_settings;
  magnitude_settings["output_magnitude"] = true;
  EXPECT_EQ(factory.infer({&input_desc, 1}, magnitude_settings).output_descs[0].dtype, DType::F32);

  const auto input_bytes = as_bytes(input_values);
  expect_magnitude_matches(run_task(factory, input_desc, input_bytes, complex_settings),
                           run_task(factory, input_desc, input_bytes, magnitude_settings));
}

TEST(FresnelDiffractionMagnitudeTest, LazilyUpdatesChangingOutputPointers) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const auto                                 input_desc = device_desc({2, 3, 32, 48}, DType::CF32);
  std::vector<cuFloatComplex>                input_values(input_desc.num_elements(),
                                                          make_cuFloatComplex(1.0f, -0.5f));
  auto                                       settings = fresnel_settings(0.01f);
  settings["output_magnitude"]                        = true;

  expect_magnitude_output_rebinding(factory, input_desc, as_bytes(input_values), settings);
}

TEST(ShortTimeFresnelDiffractionMagnitudeTest, LazilyUpdatesChangingOutputPointers) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const auto                  input_desc = device_desc({2, 3, 32, 32}, DType::CF32);
  std::vector<cuFloatComplex> input_values(input_desc.num_elements(),
                                           make_cuFloatComplex(1.0f, -0.5f));
  auto                        settings = short_time_fresnel_settings(0.01f);
  settings["output_magnitude"]         = true;

  expect_magnitude_output_rebinding(factory, input_desc, as_bytes(input_values), settings);
}

void expect_outer_graph_distance_update(const holoflow::core::ISyncTaskFactory &factory,
                                        const TDesc &desc, nlohmann::json initial,
                                        nlohmann::json updated) {
  using namespace holoflow::core;
  curaii::CudaStream          stream;
  const std::array            descriptors{desc};
  const auto                  infer = factory.infer(descriptors, initial);
  auto                        task = factory.create(descriptors, initial, {.stream = stream.get()});
  std::vector<cuFloatComplex> values(desc.num_elements());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = make_cuFloatComplex(float(i % 7) + 1.F, float(i % 3) - 1.F);
  auto bytes = as_bytes(values);
  if (desc.dtype == DType::F32) {
    std::vector<float> real(values.size());
    for (size_t i = 0; i < real.size(); ++i)
      real[i] = values[i].x;
    bytes = as_bytes(real);
  } else if (desc.dtype == DType::F16) {
    std::vector<__half> real(values.size());
    for (size_t i = 0; i < real.size(); ++i)
      real[i] = __float2half(values[i].x);
    bytes = as_bytes(real);
  }
  holonp_test::TensorTestBuffer input(desc), output(infer.output_descs[0]);
  input.upload(bytes);
  std::array  inputs{input.view()}, outputs{output.view()};
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  CudaGraphCtx recording{inputs, outputs, stream.get(), nullptr};
  task->record_cuda_graph(recording);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  const auto            original = output.download();
  ExecutionInvalidation invalidation{[&]() noexcept {
    CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
    executable = nullptr;
  }};
  task = factory.update(std::move(task), descriptors, updated,
                        {.stream = stream.get(), .execution_invalidation = &invalidation});
  ASSERT_FALSE(invalidation.invalidated);
  CUDA_CHECK(cudaGraphLaunch(executable, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  const auto actual   = output.download();
  const auto expected = run_task(factory, desc, bytes, updated).bytes;
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size() / sizeof(float); ++i) {
    float a, e;
    std::memcpy(&a, actual.data() + i * sizeof(float), sizeof(float));
    std::memcpy(&e, expected.data() + i * sizeof(float), sizeof(float));
    EXPECT_NEAR(a, e, 1e-4F * std::max(1.F, std::abs(e)));
  }
  EXPECT_NE(actual, original);
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

TEST(FresnelDiffractionUpdateTest, OuterGraphReplaysUpdatedDistanceWithoutRecapture) {
  holotask::syncs::FresnelDiffractionFactory factory;
  for (bool magnitude : {false, true}) {
    auto initial = fresnel_settings(0.01F), updated = fresnel_settings(0.02F);
    initial["output_magnitude"] = updated["output_magnitude"] = magnitude;
    expect_outer_graph_distance_update(factory, device_desc({4, 4}, DType::CF32), initial, updated);
  }
}

TEST(ShortTimeFresnelDiffractionUpdateTest, OuterGraphReplaysUpdatedDistanceWithoutRecapture) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  for (bool magnitude : {false, true}) {
    auto initial = short_time_fresnel_settings(0.01F), updated = short_time_fresnel_settings(0.02F);
    initial["output_magnitude"] = updated["output_magnitude"] = magnitude;
    expect_outer_graph_distance_update(factory, device_desc({8, 8}, DType::CF32), initial, updated);
  }
}

TEST(FresnelDiffractionUpdateTest, ReusesTaskWhenOnlyPropagationDistanceChanges) {
  holotask::syncs::FresnelDiffractionFactory factory;
  const std::vector<TDesc>                   input = {device_desc({4, 4}, DType::CF32)};

  for (const bool output_magnitude : {false, true}) {
    SCOPED_TRACE(output_magnitude);
    curaii::CudaStream  stream;
    const SyncCreateCtx ctx{stream.get()};

    auto initial_settings                = fresnel_settings(0.01f);
    initial_settings["output_magnitude"] = output_magnitude;
    auto updated_settings                = fresnel_settings(0.02f);
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

    auto initial_settings                = short_time_fresnel_settings(0.01f);
    initial_settings["output_magnitude"] = output_magnitude;
    auto updated_settings                = short_time_fresnel_settings(0.02f);
    updated_settings["output_magnitude"] = output_magnitude;

    auto task = factory.create(input, initial_settings, ctx);
    task->bind_logger(spdlog::default_logger());
    auto *raw = task.get();

    task = factory.update(std::move(task), input, updated_settings, ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    EXPECT_EQ(task.get(), raw);
  }
}

TEST(FresnelF16Production, ScaledMagnitudeMeanRemainsFiniteAndReplaysNewInput) {
  holotask::syncs::FresnelDiffractionSettings settings{.lambda           = 851e-9f,
                                                       .dx               = 20e-6f,
                                                       .dy               = 20e-6f,
                                                       .z                = .49f,
                                                       .axes             = {-2, -1},
                                                       .skip_phase_shift = true,
                                                       .output_magnitude = true,
                                                       .use_fused_f16    = true,
                                                       .reduce_magnitude = true};
  const TDesc                                 desc({1, 16, 320, 512}, DType::F16, MemLoc::Device);
  holotask::syncs::FresnelDiffractionFactory  factory;
  auto                                        inference = factory.infer(std::array{desc}, settings);
  ASSERT_EQ(inference.output_descs[0].shape, (std::vector<size_t>{1, 1, 320, 512}));
  ASSERT_EQ(inference.output_descs[0].dtype, DType::F32);
  curaii::CudaStream            stream;
  holonp_test::TensorTestBuffer input(desc), output(inference.output_descs[0]);
  std::vector<__half>           values(desc.num_elements(), __float2half(60000.f));
  std::vector<std::byte>        bytes(values.size() * sizeof(__half));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  input.upload(bytes);
  auto task = holotask::syncs::create_fused_f16_fresnel(desc, settings, {.stream = stream.get()});
  std::array              inputs{input.view()}, outputs{output.view()};
  std::atomic<bool>       cancelled{false};
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, nullptr};
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  auto               high = output.download();
  std::vector<float> actual(high.size() / sizeof(float));
  std::memcpy(actual.data(), high.data(), high.size());
  ASSERT_TRUE(std::all_of(actual.begin(), actual.end(),
                          [](float v) { return std::isfinite(v) && v >= 0; }));
  EXPECT_GT(*std::max_element(actual.begin(), actual.end()), 65504.f);
  cudaGraph_t     graph{};
  cudaGraphExec_t exec{};
  CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  (void)task->execute(ctx);
  CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
  std::fill(bytes.begin(), bytes.end(), std::byte{0});
  input.upload(bytes);
  CUDA_CHECK(cudaGraphLaunch(exec, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  auto zero = output.download();
  std::memcpy(actual.data(), zero.data(), zero.size());
  EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float v) { return v == 0; }));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  input.upload(bytes);
  CUDA_CHECK(cudaGraphLaunch(exec, stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  EXPECT_EQ(output.download(), high);
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
}

TEST(FresnelF16Production, UnsupportedSizeFallsBackWithReducedF32Output) {
  auto settings = fresnel_settings(.49f).get<holotask::syncs::FresnelDiffractionSettings>();
  settings.output_magnitude = true;
  settings.use_fused_f16    = true;
  settings.reduce_magnitude = true;
  const auto desc           = device_desc({1, 3, 64, 64}, DType::F16);
  const auto bytes          = as_bytes(std::vector<__half>(desc.num_elements(), __float2half(1.f)));
  const auto result = run_task(holotask::syncs::FresnelDiffractionFactory{}, desc, bytes, settings);
  ASSERT_EQ(result.desc.dtype, DType::F32);
  ASSERT_EQ(result.desc.shape, (std::vector<size_t>{1, 1, 64, 64}));
  std::vector<float> values(result.bytes.size() / sizeof(float));
  std::memcpy(values.data(), result.bytes.data(), result.bytes.size());
  EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                          [](float v) { return std::isfinite(v) && v >= 0; }));
  EXPECT_GT(*std::max_element(values.begin(), values.end()), 0.f);
}


TEST(FresnelF16Production, DistanceUpdatesReuseTaskAndOuterGraph) {
  using namespace holoflow::core;
  holotask::syncs::FresnelDiffractionFactory factory;
  for (auto dtype : {DType::F16, DType::F32}) {
    for (bool reduce : {false, true}) {
      auto settings = fresnel_settings(.49f);
      settings["use_fused_f16"] = true;
      settings["output_magnitude"] = true;
      settings["reduce_magnitude"] = reduce;
      const TDesc desc({1, 4, 320, 512}, dtype, MemLoc::Device);
      curaii::CudaStream stream;
      auto task = holotask::syncs::create_fused_f16_fresnel(
          desc, settings.get<holotask::syncs::FresnelDiffractionSettings>(), {stream.get()});
      auto *original = task.get();
      for (int i = 0; i < 5; ++i) {
        settings["z"] = .50f + .01f * i;
        auto start = std::chrono::steady_clock::now();
        task = factory.update(std::move(task), std::array{desc}, settings, {stream.get()});
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        EXPECT_EQ(task.get(), original);
        std::fprintf(stderr, "Fresnel synchronized distance update: %.3f ms\n",
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start).count());
      }
      // The shared helper compares a replayed outer graph with a fresh task at the new distance.
      auto updated = settings;
      updated["z"] = .65f;
      expect_outer_graph_distance_update(factory, desc, settings, updated);
    }
  }
}

TEST(AutofocusF16, SensorMatchesWidenedInput) {
  holotask::syncs::ShortTimeFresnelDiffractionFactory factory;
  const auto input = device_desc({2, 3, 32, 32}, DType::F16);
  const auto widened = device_desc(input.shape, DType::F32);
  std::vector<float> values(input.num_elements());
  std::vector<__half> half(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = float(int(i % 31) - 15) / 8;
    half[i] = __float2half(values[i]);
  }
  auto settings = short_time_fresnel_settings(0.01f);
  settings["output_magnitude"] = true;
  settings["phase_ref"] = holotask::syncs::STFDPhaseReference::GLOBAL;
  const auto reference = run_task(factory, widened, as_bytes(values), settings);
  const auto result = run_task(factory, input, as_bytes(half), settings);
  EXPECT_EQ(result.bytes, reference.bytes);
}

TEST(AutofocusF16, ComplexHalfCorrectionAndPropagation) {
  constexpr size_t height = 320, width = 512;
  const auto input = device_desc({1, 3, height, width}, DType::F16);
  const auto mask = device_desc({1, height, width}, DType::F32);
  std::vector<float> phase(height * width);
  std::vector<__half> values(input.num_elements());
  for (size_t i = 0; i < phase.size(); ++i) {
    const float x = float(i % width) / width - 0.5f;
    const float y = float(i / width) / height - 0.5f;
    phase[i] = 4 * x * x + 2 * y * y;
  }
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = __float2half(i < height * width ? 0.f : float(30000 + i % 201));
  holotask::syncs::CorrectPhaseFactory correction;
  const auto correct = [&](bool cf16) {
    return holonp_test::run_sync_factory(
        correction, std::vector<TDesc>{input, mask},
        std::vector<std::vector<std::byte>>{as_bytes(values), as_bytes(phase)},
        holotask::syncs::CorrectPhaseSettings{.output_cf16 = cf16});
  };
  const auto reference = correct(false);
  const auto candidate = correct(true);
  const auto cf32 = device_desc(input.shape, DType::CF32);
  const auto cf16 = device_desc(input.shape, DType::CF16);
  std::vector<__half2> field(values.size());
  ASSERT_EQ(candidate.output_bytes[0].size(), field.size() * sizeof(__half2));
  std::memcpy(field.data(), candidate.output_bytes[0].data(), candidate.output_bytes[0].size());
  double phase_error = 0;
  for (size_t i = phase.size(); i < field.size(); ++i) {
    const auto v = __half22float2(field[i]);
    const auto delta = std::atan2(v.y, v.x) + phase[i % phase.size()];
    phase_error += double(delta) * delta;
  }
  EXPECT_LT(std::sqrt(phase_error / (field.size() - phase.size())), 0.02);
  holotask::syncs::FresnelDiffractionFactory factory;
  auto settings = fresnel_settings(0.01f);
  settings["output_magnitude"] = true;
  settings["reduce_magnitude"] = true;
  const auto expected = run_task(factory, cf32, reference.output_bytes[0], settings);
  for (bool native : {false, true}) {
    settings["use_fused_f16"] = native;
    if (native && !holotask::syncs::supports_fused_f16_fresnel(
                      cf16, settings.get<holotask::syncs::FresnelDiffractionSettings>()))
      continue;
    // Force the native backend in this branch so a setup failure cannot pass via fallback.
    class NativeFactory : public holotask::syncs::FresnelDiffractionFactory {
    public:
      std::unique_ptr<holoflow::core::ISyncTask>
      create(std::span<const TDesc> inputs, const nlohmann::json &configuration,
             const SyncCreateCtx &ctx) const override {
        return holotask::syncs::create_fused_f16_fresnel(
            inputs[0], configuration.get<holotask::syncs::FresnelDiffractionSettings>(), ctx);
      }
    } native_factory;
    const auto &backend = native ? native_factory : factory;
    const auto result = run_task(backend, cf16, candidate.output_bytes[0], settings);
    ASSERT_EQ(result.desc.shape, expected.desc.shape);
    ASSERT_EQ(result.bytes.size(), expected.bytes.size());
    double error = 0, energy = 0;
    for (size_t i = 0; i < result.bytes.size(); i += sizeof(float)) {
      float actual, target;
      std::memcpy(&actual, result.bytes.data() + i, sizeof(float));
      std::memcpy(&target, expected.bytes.data() + i, sizeof(float));
      ASSERT_TRUE(std::isfinite(actual));
      const double delta = double(actual) - target;
      error += delta * delta;
      energy += double(target) * target;
    }
    EXPECT_LT(std::sqrt(error / energy), 0.01);
  }
}
