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

#include "support.cuh"
#include "syncs/pca/fused_f16.hh"

namespace {
namespace fused = holotask::syncs::pca_f16;

uint64_t hash_value(uint64_t hash, const void *data, size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < size; ++i)
    hash = (hash ^ bytes[i]) * 1099511628211ull;
  return hash;
}

void run_study(size_t batches, bool validate, bool full_pca) {
  constexpr size_t   window         = 32 * samples;
  constexpr size_t   partial_stride = (samples / 512) * 1024;
  curaii::CudaStream owner;
  const auto         stream = owner.get();
  Tensor             input{TDesc({batches, 32, 320, 512}, DType::U8, MemLoc::Device)};
  auto               in       = input.view();
  auto               partial  = curaii::make_unique_device_ptr<float>(batches * partial_stride);
  auto               matrices = curaii::make_unique_device_ptr<float>(batches * 1024);
  std::vector<unsigned char> host(batches * window);
  std::mt19937               rng(42);
  for (auto &value : host)
    value = static_cast<unsigned char>(rng());

  const auto covariance = [&] {
    fused::covariance(in.data(), partial.get(), matrices.get(), batches, stream);
  };
  const auto measure = [&](const char *stage, int selected_count, const auto &launch) {
    Graph graph(stream, launch);
    graph.time(stream);
    for (int i = 0; i < 20; ++i)
      std::cout << "TIME," << stage << ',' << batches << ',' << selected_count << ','
                << graph.time(stream) * 1e6 / batches << '\n';
  };
  for (int pattern = 0; pattern < (validate ? 4 : 1); ++pattern) {
    if (pattern == 1 || pattern == 2)
      std::fill(host.begin(), host.end(), pattern == 1 ? 0 : 255);
    if (pattern == 3) {
      // Disjoint supports with distinct energies and one correlated cross-tile pair.
      std::fill(host.begin(), host.end(), 0);
      for (size_t b = 0; b < batches; ++b) {
        for (size_t feature = 0; feature < 31; ++feature)
          for (size_t i = 0; i < 1024; ++i)
            host[b * window + feature * samples + feature * 1024 + i] =
                static_cast<unsigned char>(8 + feature * 7);
        host[b * window + 16 * samples] = 128;
      }
    }
    CUDA_CHECK(cudaMemcpy(in.data(), host.data(), host.size(), cudaMemcpyHostToDevice));
    if (validate) {
      covariance();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<float> gram(batches * 1024);
      CUDA_CHECK(cudaMemcpy(gram.data(), matrices.get(), gram.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      uint64_t hash = 14695981039346656037ull;
      for (size_t i = 0; i < gram.size(); ++i) {
        if (i % 32 >= (i % 1024) / 32)
          hash = hash_value(hash, &gram[i], sizeof(float));
#ifdef PCA_F16_LOWER_TRIANGLE
        else
          require(gram[i] == 0, "Upper triangle was not cleared");
#endif
      }
      std::cout << "CHECK,gram," << batches << ',' << pattern << ',' << hash << '\n';
    } else if (!full_pca) {
      measure("gram", 0, [&] {
        for (size_t b = 0; b < batches; ++b)
          fused::launch_gram(reinterpret_cast<const unsigned char *>(in.data()) + b * window,
                             partial.get() + b * partial_stride, stream);
      });
      measure("covariance", 0, covariance);
      continue;
    }
    // The all-255 baseline can produce nonfinite projections. Compare its Gram only.
    if (pattern == 2)
      continue;
    for (size_t selected_count : {size_t{1}, size_t{16}}) {
      Tensor output{TDesc({batches, selected_count, 320, 512}, DType::F16, MemLoc::Device)};
      auto   out = output.view();
      const holotask::syncs::PcaSettings settings{0, static_cast<int>(selected_count), true,
                                                  DType::F16};
      auto task = holotask::syncs::PcaFactory{}.create({&in.desc, 1}, settings, {stream});
      std::atomic<bool> cancelled{false};
      SyncCtx           ctx{.inputs       = {&in, 1},
                            .outputs      = {&out, 1},
                            .cancelled    = &cancelled,
                            .event_writer = nullptr,
                            .event_reader = nullptr};
      const auto execute = [&] { require(task->execute(ctx) == OpResult::Ok, "PCA failed"); };
      if (!validate) {
        measure("pca", static_cast<int>(selected_count), execute);
        continue;
      }
      execute();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<__half> projected(batches * selected_count * samples);
      CUDA_CHECK(cudaMemcpy(projected.data(), out.data(), projected.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost));
      uint64_t hash = 14695981039346656037ull;
      float    sign = 1;
      for (size_t i = 0; i < projected.size(); ++i) {
        float value = __half2float(projected[i]);
        require(std::isfinite(value), "Nonfinite PCA projection");
        if (pattern == 1)
          require(value == 0, "Zero input produced a nonzero projection");
        if (i % samples == 0) {
          const auto end   = projected.begin() + i + samples;
          const auto first = std::find_if(projected.begin() + i, end,
                                          [](__half x) { return __half2float(x) != 0; });
          sign             = first != end && __half2float(*first) < 0 ? -1.f : 1.f;
        }
        value = value == 0 ? 0.f : sign * value;
        hash  = hash_value(hash, &value, sizeof(value));
      }
      std::cout << "CHECK,pca," << batches << ',' << pattern << ',' << selected_count << ',' << hash
                << '\n';
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && (std::strcmp(argv[1], "--covariance-f16-study") == 0 ||
                    std::strcmp(argv[1], "--pca-f16-study") == 0 ||
                    std::strcmp(argv[1], "--validate-f16-study") == 0)) {
    const bool validate = std::strcmp(argv[1], "--validate-f16-study") == 0;
    try {
      const bool full_pca = std::strcmp(argv[1], "--pca-f16-study") == 0;
      run_study(1, validate, full_pca);
      run_study(4, validate, full_pca);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RegisterBenchmark("PCA/U8_to_FP32", [](benchmark::State &s) { run(s, 0); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("PCA/FP32", [](benchmark::State &s) { run(s, 1); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("PCA/U8_Fused_FP16_Autotuned",
                               [](benchmark::State &s) { run(s, 2); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RunSpecifiedBenchmarks();
  context.reset();
  benchmark::Shutdown();
  return failed ? 1 : 0;
}
