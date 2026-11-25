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
#include <gtest/gtest.h>
TEST(PcaBenchmark, SupportedConfigurationsPreserveNumericalAndReplayInvariants) {
  Context c;
  c.runtime_solver = std::make_unique<RuntimeSolver>();
  c.rows<128>();
  c.rows<256>();
  c.rows<512>();
  for (int test = 0; test < 7; ++test) {
    std::fprintf(stderr, "PCA validation case %d\n", test);
    for (size_t i = 0; i < c.host.size(); ++i) {
      int feature = int(i / samples), local = int(i % samples);
      if (test == 6)
        c.host[i] = static_cast<unsigned char>((i * 1103515245ULL + 12345ULL) >> 24);
      if (test == 1)
        c.host[i] = 0;
      if (test == 2)
        c.host[i] = 255;
      if (test == 3)
        c.host[i] = static_cast<unsigned char>((feature % 3) * 64);
      if (test == 4)
        c.host[i] = local == feature ? static_cast<unsigned char>(feature * 7 + 1) : 0;
      if (test == 5)
        c.host[i] = static_cast<unsigned char>((local % 256) * (feature % 2));
    }
    CUDA_CHECK(cudaMemcpy(c.bytes.data(), c.host.data(), c.host.size(), cudaMemcpyHostToDevice));
    c.execute(*c.conversion, c.u, c.f);
    CUDA_CHECK(cudaStreamSynchronize(c.stream));
    c.reference();
    for (const auto &candidate : c.candidates) {
      if (test == 6 && &candidate != &c.candidates.front())
        continue;
      candidate.launch();
      CUDA_CHECK(cudaStreamSynchronize(c.stream));
      int status;
      CUDA_CHECK(cudaMemcpy(&status, c.info.get(), sizeof(int), cudaMemcpyDeviceToHost));
      ASSERT_EQ(status, 0) << " case=" << test << " K=" << candidate.k << " M=" << candidate.m;
      ASSERT_TRUE(c.validate(test == 4, candidate.k))
          << "case=" << test << " K=" << candidate.k << " M=" << candidate.m;
      std::vector<__half> before(samples * 15), after(samples * 15);
      CUDA_CHECK(cudaMemcpy(before.data(), c.half.get(), before.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost));
      if (test == 1)
        ASSERT_TRUE(std::all_of(before.begin(), before.end(),
                                [](auto value) { return __half2float(value) == 0.f; }));
      Graph graph(c.stream, candidate.launch);
      CUDA_CHECK(cudaGraphLaunch(graph.exec, c.stream));
      CUDA_CHECK(cudaGraphLaunch(graph.exec, c.stream));
      CUDA_CHECK(cudaStreamSynchronize(c.stream));
      CUDA_CHECK(cudaMemcpy(after.data(), c.half.get(), after.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost));
      ASSERT_EQ(0, std::memcmp(before.data(), after.data(), before.size() * sizeof(__half)));
      std::vector<unsigned char> input(c.host.size());
      CUDA_CHECK(cudaMemcpy(input.data(), c.bytes.data(), input.size(), cudaMemcpyDeviceToHost));
      ASSERT_EQ(input, c.host);
    }
  }
}

TEST(PcaBenchmark, ProductionConversionMatchesPreconvertedInput) {
  Context            c;
  std::vector<float> converted(c.host.size()), baseline(samples * 15), actual(samples * 15);
  CUDA_CHECK(cudaMemcpy(converted.data(), c.floats.data(), converted.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool exact = true;
  for (size_t i = 0; i < converted.size(); ++i)
    exact = exact && (converted[i] == float(c.host[i]));
  ASSERT_TRUE(exact);
  CUDA_CHECK(cudaMemcpy(baseline.data(), c.output.data(), baseline.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < converted.size(); ++i)
    converted[i] = float(c.host[i]);
  CUDA_CHECK(cudaMemcpy(c.floats.data(), converted.data(), converted.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  c.execute(*c.pca, c.f, c.o);
  CUDA_CHECK(cudaStreamSynchronize(c.stream));
  CUDA_CHECK(cudaMemcpy(actual.data(), c.output.data(), actual.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double error = 0, energy = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double d = double(actual[i]) - baseline[i];
    error += d * d;
    energy += double(baseline[i]) * baseline[i];
  }
  EXPECT_LT(std::sqrt(error / std::max(energy, 1e-30)), 1e-4);
}
