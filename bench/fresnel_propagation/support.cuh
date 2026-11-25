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

#include "kernels.cuh"

namespace {
bool failed = false;
void run(benchmark::State *state, bool half) {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    require(prop.major * 100 + prop.minor * 10 == FRESNEL_BENCH_SM,
            "GPU does not match compiled benchmark architecture");

    curaii::CudaStream stream_owner;
    auto               stream          = stream_owner.get();
    auto               input           = curaii::make_unique_device_ptr<__half>(N);
    auto               intermediate    = curaii::make_unique_device_ptr<HalfComplex>(N);
    auto               output          = curaii::make_unique_device_ptr<HalfComplex>(N);
    auto               transposed      = curaii::make_unique_device_ptr<HalfComplex>(N);
    auto               transformed     = curaii::make_unique_device_ptr<HalfComplex>(N);
    auto               lens            = curaii::make_unique_device_ptr<float2>(X * Y);
    auto               explicit_output = curaii::make_unique_device_ptr<float2>(N);
    Buffers            buffers{input.get(),      intermediate.get(), output.get(),
                               transposed.get(), transformed.get(),  lens.get()};
    cache_lens<<<(X * Y + 255) / 256, 256, 0, stream>>>(lens.get());
    CUDA_CHECK(cudaGetLastError());
    using namespace holoflow::core;
    Tensor real_input(TDesc({Z, Y, X}, DType::F32, MemLoc::Device));
    Tensor complex_output(TDesc({Z, Y, X}, DType::CF32, MemLoc::Device));
    auto   iv = real_input.view(), ov = complex_output.view();
    holotask::syncs::FresnelDiffractionFactory  factory;
    holotask::syncs::FresnelDiffractionSettings settings{
        .lambda = 852e-9f, .dx = 20e-6f, .dy = 20e-6f, .z = .1f};
    auto task =
        factory.create(std::span<const TDesc>(&iv.desc, 1), nlohmann::json(settings), {stream});
    auto fp32 = [&]() {
      std::atomic<bool> cancelled{false};
      SyncCtx           ctx{.inputs       = {&iv, 1},
                            .outputs      = {&ov, 1},
                            .cancelled    = &cancelled,
                            .event_writer = nullptr,
                            .event_reader = nullptr};
      require(task->execute(ctx) == OpResult::Ok, "Production propagation failed");
    };
    curaii::CufftHandle explicit_plan;
    int                 dims[2]         = {Y, X};
    size_t              workspace_bytes = 0;
    CUFFT_CHECK(cufftMakePlanMany(explicit_plan.get(), 2, dims, nullptr, 1, X * Y, nullptr, 1,
                                  X * Y, CUFFT_C2C, Z, &workspace_bytes));
    CUFFT_CHECK(cufftSetStream(explicit_plan.get(), stream));
    std::vector<Configuration> xs, ys, yts, xf, yf, yshared;
    register_axis<true, false, false>(xs, buffers, stream, prop);
    register_axis<false, false, false>(ys, buffers, stream, prop);
    register_axis<false, true, false>(yts, buffers, stream, prop);
    register_axis<true, false, true>(xf, buffers, stream, prop);
    register_axis<false, true, true>(yf, buffers, stream, prop);
    register_axis<false, false, true>(yshared, buffers, stream, prop);
    auto xconfigs = [&](Layout layout) -> const std::vector<Configuration> & {
      return layout == Layout::FusedTransposed ? xf : xs;
    };
    auto yconfigs = [&](Layout layout) -> const std::vector<Configuration> & {
      switch (layout) {
      case Layout::Strided:
        return ys;
      case Layout::Transposed:
        return yts;
      case Layout::YShared:
        return yshared;
      case Layout::FusedTransposed:
        return yf;
      }
      throw std::runtime_error("Invalid layout");
    };
    auto forward = [&]() {
      transpose<true><<<dim3(X / 32, Y / 32, Z), dim3(32, 8), 0, stream>>>(intermediate.get(),
                                                                           transposed.get());
      CUDA_CHECK(cudaGetLastError());
    };
    auto backward = [&]() {
      transpose<false>
          <<<dim3(Y / 32, X / 32, Z), dim3(32, 8), 0, stream>>>(transformed.get(), output.get());
      CUDA_CHECK(cudaGetLastError());
    };
    auto enqueue = [&](const Pipeline &p) {
      xconfigs(p.layout)[p.x].enqueue();
      if (p.layout == Layout::Transposed)
        forward();
      yconfigs(p.layout)[p.y].enqueue();
      if (p.layout == Layout::Transposed)
        backward();
    };
    auto matched_reference = [&](const Pipeline &p) {
      const auto &x = xconfigs(p.layout)[p.x], &y = yconfigs(p.layout)[p.y];
      return Pipeline{find_configuration(xs, x.elements, x.batch),
                      find_configuration(ys, y.elements, y.batch), Layout::Strided};
    };
    std::vector<__half>      host(N), downloaded_input(N);
    std::vector<float>       widened(N), downloaded_real(N);
    std::vector<HalfComplex> actual(N), replay(N);
    std::vector<float2>      reference(N), explicit_reference(N);
    auto                     upload = [&](int test) {
      std::mt19937 rng(12345);
      for (int i = 0; i < N; ++i) {
        float value = 0;
        if (test == 0)
          value = std::sin(float(i) * .123f);
        if (test == 1)
          value = float(int(rng() % 2049) - 1024) / 1024.f;
        if (test == 3)
          value = .125f;
        if (test == 4) {
          int z = i / (X * Y), local = i % (X * Y);
          value = local == ((z * 71) % Y) * X + (z * 137) % X ? float(z + 1) / Z : 0.f;
        }
        if (test == 5)
          value = 60000.f;
        host[i]    = __float2half_rn(value);
        widened[i] = __half2float(host[i]);
      }
      CUDA_CHECK(cudaMemcpyAsync(input.get(), host.data(), N * sizeof(__half),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(real_input.data(), widened.data(), N * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      fp32();
      CUDA_CHECK(cudaMemcpyAsync(reference.data(), complex_output.data(), N * sizeof(float2),
                                 cudaMemcpyDeviceToHost, stream));
      prepare_reference<<<(N + 255) / 256, 256, 0, stream>>>(input.get(), explicit_output.get());
      CUFFT_CHECK(cufftExecC2C(explicit_plan.get(), explicit_output.get(), explicit_output.get(),
                               CUFFT_FORWARD));
      CUDA_CHECK(cudaMemcpyAsync(explicit_reference.data(), explicit_output.get(),
                                 N * sizeof(float2), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      double error = 0, energy = 0;
      for (int i = 0; i < N; ++i) {
        double r  = reference[i].x - explicit_reference[i].x,
               im = reference[i].y - explicit_reference[i].y;
        error += r * r + im * im;
        energy += double(reference[i].x) * reference[i].x + double(reference[i].y) * reference[i].y;
      }
      require(energy == 0 ? error == 0 : std::sqrt(error / energy) < 1e-5,
              "Production reference mismatch");
    };
    auto validate = [&](const Pipeline &p, bool stress, bool replay_check) {
      enqueue(p);
      CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.get(), N * sizeof(HalfComplex),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      auto e = compare(actual, reference);
      if (!stress)
        require(e.nonfinite == 0 && e.relative < .01, "FP16 accuracy validation failed");
      if (p.layout == Layout::YShared || p.layout == Layout::FusedTransposed) {
        enqueue(matched_reference(p));
        CUDA_CHECK(cudaMemcpyAsync(replay.data(), output.get(), N * sizeof(HalfComplex),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        require(std::memcmp(actual.data(), replay.data(), N * sizeof(HalfComplex)) == 0,
                "Fused output differs from matching unfused descriptors");
      }
      if (replay_check) {
        // measure() replays a captured graph; verify its final result and both inputs.
        if (true) {
          cudaGraph_t     graph = nullptr;
          cudaGraphExec_t exec  = nullptr;
          CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          enqueue(p);
          CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
          CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
          for (int repeat = 0; repeat < 2; ++repeat)
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaGraphExecDestroy(exec));
          CUDA_CHECK(cudaGraphDestroy(graph));
        } else
          (void)measure(stream, [&]() { enqueue(p); });
        CUDA_CHECK(cudaMemcpy(replay.data(), output.get(), N * sizeof(HalfComplex),
                              cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), replay.data(), N * sizeof(HalfComplex)) == 0,
                "Graph replay changed output");
        CUDA_CHECK(cudaMemcpy(downloaded_input.data(), input.get(), N * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(downloaded_real.data(), real_input.data(), N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        require(std::memcmp(host.data(), downloaded_input.data(), N * sizeof(__half)) == 0,
                "FP16 input modified");
        require(std::memcmp(widened.data(), downloaded_real.data(), N * sizeof(float)) == 0,
                "FP32 input modified");
      }
      return e;
    };

    if (!state) {
      for (int test = 0; test < 6; ++test) {
        upload(test);
        for (auto layout :
             {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed}) {
          for (int x = 0; x < int(xconfigs(layout).size()); ++x)
            validate({x, 0, layout}, test == 5, true);
          for (int y = 0; y < int(yconfigs(layout).size()); ++y)
            validate({0, y, layout}, test == 5, true);
        }
      }
      return;
    }
    upload(0);
    Pipeline selected{};
    if (half) {
      static bool     tuned = false;
      static Pipeline cached{};
      if (!tuned) {
        std::fprintf(stderr, "Tuning FP16 Fresnel pipelines...\n");
        std::vector<Pipeline> candidates;
        for (auto layout :
             {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed})
          for (int x = 0; x < int(xconfigs(layout).size()); ++x)
            for (int y = 0; y < int(yconfigs(layout).size()); ++y) {
              Pipeline p{x, y, layout};
              try {
                p.error = validate(p, false, false);
              } catch (const std::runtime_error &) {
                continue;
              }
              p.timing = measure(stream, [&] { enqueue(p); });
              candidates.push_back(p);
            }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &a, const auto &b) { return a.timing.median < b.timing.median; });
        require(!candidates.empty(), "No supported FP16 pipeline");
        bool accepted = false;
        for (const auto &candidate : candidates) {
          try {
            for (int test = 0; test < 5; ++test) {
              upload(test);
              validate(candidate, false, true);
            }
            cached   = candidate;
            accepted = true;
            break;
          } catch (const std::runtime_error &) {
          }
        }
        require(accepted, "No FP16 pipeline passed bounded validation");
        tuned = true;
      }
      selected = cached;
      upload(0);
      auto &x = xconfigs(selected.layout)[selected.x];
      auto &y = yconfigs(selected.layout)[selected.y];
      state->SetLabel(std::string(name(selected.layout)) + " X=" + std::to_string(x.elements) +
                      "/" + std::to_string(x.batch) + " Y=" + std::to_string(y.elements) + "/" +
                      std::to_string(y.batch));
    }
    auto launch = [&] {
      if (half)
        enqueue(selected);
      else
        fp32();
    };
    for (int i = 0; i < 5; ++i)
      launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    GraphResources resources;
    auto          &graph = resources.graph;
    auto          &exec  = resources.exec;
    auto          &begin = resources.begin;
    auto          &end   = resources.end;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (auto _ : *state) {
      CUDA_CHECK(cudaEventRecord(begin, stream));
      for (int i = 0; i < launches; ++i)
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
      CUDA_CHECK(cudaEventRecord(end, stream));
      CUDA_CHECK(cudaEventSynchronize(end));
      float ms{};
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      state->SetIterationTime(ms / (1000.0 * launches));
    }
    state->counters["volumes/s"] =
        benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
  } catch (const std::exception &e) {
    failed = true;
    if (state)
      state->SkipWithError(e.what());
    else
      throw;
  }
}
} // namespace
