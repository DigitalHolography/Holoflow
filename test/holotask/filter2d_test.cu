#include "../../src/holotask/src/syncs/filter2d/fused_f16.hh"
#include "curaii/cuda.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "tensor_test_buffer.hh"
#include <chrono>
#include <cstdio>
#include <cuda_fp16.h>
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

#include <cmath>
#include <cstring>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/filter2d.hh"

#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

namespace {

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &v) {
  std::vector<std::byte> out(v.size() * sizeof(T));
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

nlohmann::json all_pass_settings() {
  return holotask::syncs::Filter2DSettings{
      .r_inner = 0,
      .r_outer = 100,
      .s_inner = 0,
      .s_outer = 0,
  };
}

void expect_f32_near(const std::vector<std::byte> &actual, const std::vector<float> &expected,
                     float atol = 1e-3f) {
  ASSERT_EQ(actual.size(), expected.size() * sizeof(float));
  const auto *a = reinterpret_cast<const float *>(actual.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(a[i], expected[i], atol);
  }
}

} // namespace

class Filter2DInferTest : public ::testing::Test {
protected:
  holotask::syncs::Filter2DFactory factory;
};

TEST_F(Filter2DInferTest, AcceptsRealInputAndOutputsReal) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::F32)};
  const auto               r  = factory.infer(in, all_pass_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::F32);
  EXPECT_TRUE(r.in_place.empty());
}

TEST_F(Filter2DInferTest, KeepsComplexInputInPlace) {
  const std::vector<TDesc> in = {device_desc({2, 2}, DType::CF32)};
  const auto               r  = factory.infer(in, all_pass_settings());

  EXPECT_EQ(r.kind, TaskKind::Sync);
  ASSERT_EQ(r.output_descs.size(), 1);
  EXPECT_EQ(r.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(r.output_descs[0].dtype, DType::CF32);
  ASSERT_EQ(r.in_place.size(), 1);
  EXPECT_EQ(r.in_place[0].in_idx, 0);
  EXPECT_EQ(r.in_place[0].out_idx, 0);
}

class Filter2DExecuteTest : public ::testing::Test {
protected:
  holotask::syncs::Filter2DFactory factory;
};

TEST_F(Filter2DExecuteTest, RealInputUsesRealOutput) {
  const TDesc d    = device_desc({2, 2}, DType::F32);
  const auto  data = as_bytes(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

  const auto run =
      holonp_test::run_sync_factory(factory, std::vector<TDesc>{d},
                                    std::vector<std::vector<std::byte>>{data}, all_pass_settings());

  ASSERT_EQ(run.output_descs.size(), 1);
  EXPECT_EQ(run.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(run.output_descs[0].dtype, DType::F32);

  expect_f32_near(run.output_bytes[0], std::vector<float>{4.0f, 8.0f, 12.0f, 16.0f});
}

TEST(Filter2DNormalized, RealAllPassPreservesAmplitude) {
  holotask::syncs::Filter2DFactory factory;
  auto                             settings = all_pass_settings();
  settings["normalize"]                     = true;
  const auto desc                           = device_desc({2, 2}, DType::F32);
  const auto result                         = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc},
      std::vector<std::vector<std::byte>>{as_bytes(std::vector<float>{1, 2, 3, 4})}, settings);
  expect_f32_near(result.output_bytes[0], {1, 2, 3, 4});
}

TEST(Filter2DNormalized, HalfAllPassPreservesAmplitude) {
  holotask::syncs::Filter2DFactory factory;
  auto                             settings = all_pass_settings();
  settings["normalize"]                     = true;
  const auto          desc                  = device_desc({2, 2}, DType::F16);
  std::vector<__half> values{__float2half(1), __float2half(2), __float2half(3), __float2half(4)};
  const auto          result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(values)},
      settings);
  ASSERT_EQ(result.output_bytes[0].size(), values.size() * sizeof(__half));
  std::vector<__half> actual(values.size());
  std::memcpy(actual.data(), result.output_bytes[0].data(), result.output_bytes[0].size());
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_NEAR(__half2float(actual[i]), __half2float(values[i]), .01f);
}

TEST(Filter2DF16, MatchesNormalizedReferenceAndReplaysMaskUpdates) {
  using namespace holoflow::core;
  holotask::syncs::Filter2DFactory factory;
  for (auto dtype : {DType::F16, DType::F32}) {
    const TDesc desc({3, 320, 512}, dtype, MemLoc::Device);
    if (!holotask::syncs::supports_fused_f16_filter(desc))
      GTEST_SKIP() << "Requires SM86 or SM120";
    auto               settings = holotask::syncs::Filter2DSettings{8, 100, 4, 10, true, true};
    std::vector<float> values(desc.num_elements());
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = float(std::sin(double(i) * .071) * 100 + std::cos(double(i) * .019) * 17);
    std::vector<__half> halves(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      halves[i] = __float2half(values[i]);
      if (dtype == DType::F16)
        values[i] = __half2float(halves[i]);
    }
    holonp_test::TensorTestBuffer input(desc), output(desc);
    input.upload(dtype == DType::F16 ? as_bytes(halves) : as_bytes(values));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    curaii::CudaStream stream(cudaStreamNonBlocking);
    auto            task = holotask::syncs::create_fused_f16_filter(desc, settings, stream.get());
    auto           *original = task.get();
    std::array      inputs{input.view()}, outputs{output.view()};
    SyncCtx         call{inputs, outputs, nullptr, nullptr, nullptr};
    cudaGraph_t     graph{};
    cudaGraphExec_t exec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    CudaGraphCtx capture{inputs, outputs, stream.get(), nullptr};
    task->record_cuda_graph(capture);
    CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    for (int update = 0; update < 7; ++update) {
      SCOPED_TRACE(update);
      SCOPED_TRACE(int(dtype));
      if (update >= 2) {
        settings.r_inner = update == 6 ? 1 : 0;
        settings.r_outer = 1000;
        settings.s_inner = settings.s_outer = 0;
        for (size_t i = 0; i < values.size(); ++i) {
          values[i] = update == 2   ? 0.f
                      : update == 3 ? (dtype == DType::F32 ? 1e30f : 60000.f)
                      : update == 4 ? (i % (320 * 512) == 0 ? 1.f : 0.f)
                      : update == 6 ? 5.f
                                    : float(std::sin(double(i) * .173) * 5);
          halves[i] = __float2half(values[i]);
          if (dtype == DType::F16)
            values[i] = __half2float(halves[i]);
        }
        input.upload(dtype == DType::F16 ? as_bytes(halves) : as_bytes(values));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
      }
      if (update) {
        if (update == 1)
          settings.r_inner = 20;
        ExecutionInvalidation invalidation{[]() noexcept {}};
        task = factory.update(std::move(task), std::array{desc}, settings,
                              {stream.get(), &invalidation});
        ASSERT_EQ(task.get(), original);
        ASSERT_FALSE(invalidation.invalidated);
      }
      CUDA_CHECK(cudaGraphLaunch(exec, stream.get()));
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));
      auto bytes                       = output.download();
      auto reference_settings          = settings;
      reference_settings.use_fused_f16 = false;
      auto reference                   = holonp_test::run_sync_factory(
          factory, std::vector<TDesc>{TDesc(desc.shape, DType::F32, MemLoc::Device)},
          std::vector<std::vector<std::byte>>{as_bytes(values)}, reference_settings);
      std::vector<float> expected(values.size());
      std::memcpy(expected.data(), reference.output_bytes[0].data(),
                  expected.size() * sizeof(float));
      double error = 0, energy = 0;
      for (size_t i = 0; i < values.size(); ++i) {
        float actual;
        if (dtype == DType::F16) {
          __half h;
          std::memcpy(&h, bytes.data() + i * sizeof(h), sizeof(h));
          actual = __half2float(h);
        } else
          std::memcpy(&actual, bytes.data() + i * sizeof(float), sizeof(float));
        ASSERT_TRUE(std::isfinite(actual));
        error += double(actual - expected[i]) * (actual - expected[i]);
        energy += double(expected[i]) * expected[i];
      }
      if (update == 6)
        EXPECT_LT(std::sqrt(error / values.size()) / 5.0, .001);
      else if (energy > 0)
        EXPECT_LT(std::sqrt(error / energy), .01);
      else
        EXPECT_EQ(error, 0);
    }
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = float(std::sin(double(i) * .173) * 5);
      halves[i] = __float2half(values[i]);
      if (dtype == DType::F16)
        values[i] = __half2float(halves[i]);
    }
    input.upload(dtype == DType::F16 ? as_bytes(halves) : as_bytes(values));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    if (dtype == DType::F16) {
      for (int planes : {3, 64}) {
        for (bool native : {false, true}) {
          auto bench_settings          = settings;
          bench_settings.use_fused_f16 = native;
          const TDesc bench_desc({static_cast<size_t>(planes), 320, 512},
                                 native ? DType::F16 : DType::F32, MemLoc::Device);
          holonp_test::TensorTestBuffer bench_input(bench_desc), bench_output(bench_desc);
          std::array bench_inputs{bench_input.view()}, bench_outputs{bench_output.view()};
          if (native) {
            std::vector<__half> samples(bench_desc.num_elements());
            for (size_t i = 0; i < samples.size(); ++i)
              samples[i] = halves[i % halves.size()];
            bench_input.upload(as_bytes(samples));
          } else {
            std::vector<float> samples(bench_desc.num_elements());
            for (size_t i = 0; i < samples.size(); ++i)
              samples[i] = values[i % values.size()];
            bench_input.upload(as_bytes(samples));
          }
          CUDA_CHECK(cudaStreamSynchronize(nullptr));
          auto setup_start = std::chrono::steady_clock::now();
          auto bench_task =
              native ? holotask::syncs::create_fused_f16_filter(bench_desc, bench_settings,
                                                                stream.get())
                     : factory.create(std::array{bench_desc}, bench_settings, {stream.get()});
          CUDA_CHECK(cudaStreamSynchronize(stream.get()));
          double          setup_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - setup_start)
                                         .count();
          cudaGraph_t     bench_graph{};
          cudaGraphExec_t bench_exec{};
          CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
          CudaGraphCtx bench_capture{bench_inputs, bench_outputs, stream.get(), nullptr};
          bench_task->record_cuda_graph(bench_capture);
          CUDA_CHECK(cudaStreamEndCapture(stream.get(), &bench_graph));
          CUDA_CHECK(cudaGraphInstantiateWithFlags(&bench_exec, bench_graph, 0));
          for (int i = 0; i < 10; ++i)
            CUDA_CHECK(cudaGraphLaunch(bench_exec, stream.get()));
          cudaEvent_t start{}, stop{};
          CUDA_CHECK(cudaEventCreate(&start));
          CUDA_CHECK(cudaEventCreate(&stop));
          std::array<float, 40> samples{};
          for (auto &sample : samples) {
            CUDA_CHECK(cudaEventRecord(start, stream.get()));
            for (int i = 0; i < 10; ++i)
              CUDA_CHECK(cudaGraphLaunch(bench_exec, stream.get()));
            CUDA_CHECK(cudaEventRecord(stop, stream.get()));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&sample, start, stop));
            sample *= 100.f; // Convert the ten-replay batch to microseconds per replay.
          }
          std::sort(samples.begin(), samples.end());
          std::fprintf(stderr,
                       "Filter2D %s %d planes: median %.1f us, p95 %.1f us, warm setup %.2f ms\n",
                       native ? "F16" : "FP32", planes, (samples[19] + samples[20]) / 2,
                       samples[37], setup_ms);
          CUDA_CHECK(cudaEventDestroy(start));
          CUDA_CHECK(cudaEventDestroy(stop));
          CUDA_CHECK(cudaGraphExecDestroy(bench_exec));
          CUDA_CHECK(cudaGraphDestroy(bench_graph));
        }
      }
    }
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream.get()));
    for (int i = 0; i < 30; ++i)
      CUDA_CHECK(cudaGraphLaunch(exec, stream.get()));
    CUDA_CHECK(cudaEventRecord(stop, stream.get()));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    std::fprintf(stderr, "F16 Filter2D dtype=%d: %.3f ms per 3-plane graph\n", int(dtype),
                 elapsed / 30);
    const TDesc                   reference_desc(desc.shape, DType::F32, MemLoc::Device);
    holonp_test::TensorTestBuffer reference_input(reference_desc), reference_output(reference_desc);
    reference_input.upload(as_bytes(values));
    auto baseline_settings          = settings;
    baseline_settings.use_fused_f16 = false;
    auto baseline = factory.create(std::array{reference_desc}, baseline_settings, {stream.get()});
    std::array  baseline_inputs{reference_input.view()}, baseline_outputs{reference_output.view()};
    cudaGraph_t baseline_graph{};
    cudaGraphExec_t baseline_exec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    CudaGraphCtx baseline_capture{baseline_inputs, baseline_outputs, stream.get(), nullptr};
    baseline->record_cuda_graph(baseline_capture);
    CUDA_CHECK(cudaStreamEndCapture(stream.get(), &baseline_graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&baseline_exec, baseline_graph, 0));
    CUDA_CHECK(cudaGraphLaunch(baseline_exec, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    CUDA_CHECK(cudaEventRecord(start, stream.get()));
    for (int i = 0; i < 30; ++i)
      CUDA_CHECK(cudaGraphLaunch(baseline_exec, stream.get()));
    CUDA_CHECK(cudaEventRecord(stop, stream.get()));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    std::fprintf(stderr, "FP32 Filter2D: %.3f ms per 3-plane graph\n", elapsed / 30);
    // Check the actual F16 Filter2D -> Fresnel boundary against a normalized FP32 chain.
    holotask::syncs::FresnelDiffractionFactory  fresnel_factory;
    holotask::syncs::FresnelDiffractionSettings fresnel_settings{.lambda           = 851e-9f,
                                                                 .dx               = 20e-6f,
                                                                 .dy               = 20e-6f,
                                                                 .z                = .49f,
                                                                 .output_magnitude = true,
                                                                 .use_fused_f16    = true};
    auto fresnel = fresnel_factory.create(std::array{desc}, fresnel_settings, {stream.get()});
    auto legacy_settings          = fresnel_settings;
    legacy_settings.use_fused_f16 = false;
    auto legacy_fresnel =
        fresnel_factory.create(std::array{reference_desc}, legacy_settings, {stream.get()});
    holonp_test::TensorTestBuffer propagated(reference_desc), reference_propagated(reference_desc);
    std::array                    spatial_inputs{output.view()}, spatial_outputs{propagated.view()};
    std::array                    legacy_spatial_inputs{reference_output.view()},
        legacy_spatial_outputs{reference_propagated.view()};
    cudaGraph_t     chains[2]{};
    cudaGraphExec_t chain_execs[2]{};
    for (int mode = 0; mode < 2; ++mode) {
      CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
      if (mode == 0) {
        task->record_cuda_graph(capture);
        CudaGraphCtx spatial_capture{spatial_inputs, spatial_outputs, stream.get(), nullptr};
        fresnel->record_cuda_graph(spatial_capture);
      } else {
        baseline->record_cuda_graph(baseline_capture);
        CudaGraphCtx spatial_capture{legacy_spatial_inputs, legacy_spatial_outputs, stream.get(),
                                     nullptr};
        legacy_fresnel->record_cuda_graph(spatial_capture);
      }
      CUDA_CHECK(cudaStreamEndCapture(stream.get(), &chains[mode]));
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&chain_execs[mode], chains[mode], 0));
      CUDA_CHECK(cudaGraphLaunch(chain_execs[mode], stream.get()));
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));
      CUDA_CHECK(cudaEventRecord(start, stream.get()));
      for (int i = 0; i < 30; ++i)
        CUDA_CHECK(cudaGraphLaunch(chain_execs[mode], stream.get()));
      CUDA_CHECK(cudaEventRecord(stop, stream.get()));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
      std::fprintf(stderr, "Filter2D + Fresnel mode=%d dtype=%d: %.3f ms per 3-plane graph\n", mode,
                   int(dtype), elapsed / 30);
    }
    auto   actual_image = propagated.download(), expected_image = reference_propagated.download();
    double image_error = 0, image_energy = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      float a, e;
      std::memcpy(&a, actual_image.data() + i * sizeof(float), sizeof(float));
      std::memcpy(&e, expected_image.data() + i * sizeof(float), sizeof(float));
      ASSERT_TRUE(std::isfinite(a));
      image_error += double(a - e) * (a - e);
      image_energy += double(e) * e;
    }
    EXPECT_LT(std::sqrt(image_error / image_energy), .01);
    for (int mode = 0; mode < 2; ++mode) {
      CUDA_CHECK(cudaGraphExecDestroy(chain_execs[mode]));
      CUDA_CHECK(cudaGraphDestroy(chains[mode]));
    }
    CUDA_CHECK(cudaGraphExecDestroy(baseline_exec));
    CUDA_CHECK(cudaGraphDestroy(baseline_graph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }
}
