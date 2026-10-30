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

#pragma once
#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/pca.hh"
#include "kernels.cuh"
#include "solver.cuh"
#include <algorithm>
#include <atomic>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using namespace pca_bench;
using namespace holoflow::core;
bool failed = false;
void require(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
struct Graph {
  cudaGraph_t     graph{};
  cudaGraphExec_t exec{};
  cudaEvent_t     start{}, end{};
  ~Graph() {
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
    if (start)
      cudaEventDestroy(start);
    if (end)
      cudaEventDestroy(end);
  }
  Graph(cudaStream_t stream, const std::function<void()> &launch) {
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
  }
  double time(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 100; ++i)
      CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms / 100000.0;
  }
};
struct Context {
  curaii::CudaStream               owner;
  cudaStream_t                     stream = owner.get();
  Tensor                           bytes{TDesc({32, 320, 512}, DType::U8, MemLoc::Device)};
  Tensor                           floats{TDesc({32, 320, 512}, DType::F32, MemLoc::Device)};
  Tensor                           output{TDesc({15, 320, 512}, DType::F32, MemLoc::Device)};
  TView                            u = bytes.view(), f = floats.view(), o = output.view();
  std::unique_ptr<ISyncTask>       conversion, pca;
  curaii::unique_device_ptr<float> partial = curaii::make_unique_device_ptr<float>((samples / 128) *
                                                                                   1024),
                                   matrix  = curaii::make_unique_device_ptr<float>(1024),
                                   values  = curaii::make_unique_device_ptr<float>(32);
  curaii::unique_device_ptr<int>   info    = curaii::make_unique_device_ptr<int>(1);
  curaii::unique_device_ptr<__half> half   = curaii::make_unique_device_ptr<__half>(samples * 15);
  std::vector<unsigned char>        host;
  struct Candidate {
    int                   k, m, tg, tp;
    std::function<void()> launch;
  };
  std::vector<Candidate>         candidates;
  std::unique_ptr<RuntimeSolver> runtime_solver;
  int                            selected = -1;
  std::vector<double>            reference_vectors, reference_output, reference_gram;
  void                           reference() {
    curaii::CublasHandle     blas;
    curaii::CusolverDnHandle solver;
    auto                     x   = curaii::make_unique_device_ptr<double>(host.size());
    auto                     a   = curaii::make_unique_device_ptr<double>(1024);
    auto                     eig = curaii::make_unique_device_ptr<double>(32);
    std::vector<double>      h(host.begin(), host.end());
    CUDA_CHECK(cudaMemcpy(x.get(), h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    double one = 1, zero = 0;
    CUBLAS_CHECK(cublasDgemm(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, samples, &one, x.get(),
                             samples, x.get(), samples, &zero, a.get(), 32));
    reference_gram.resize(1024);
    CUDA_CHECK(
        cudaMemcpy(reference_gram.data(), a.get(), 1024 * sizeof(double), cudaMemcpyDeviceToHost));
    int size = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver.get(), CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, 32, a.get(), 32, eig.get(),
                                               &size));
    auto work = curaii::make_unique_device_ptr<double>(size);
    CUSOLVER_CHECK(cusolverDnDsyevd(solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                    32, a.get(), 32, eig.get(), work.get(), size, info.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    int status;
    CUDA_CHECK(cudaMemcpy(&status, info.get(), sizeof(int), cudaMemcpyDeviceToHost));
    require(status == 0, "FP64 reference solver failed");
    reference_vectors.resize(1024);
    CUDA_CHECK(cudaMemcpy(reference_vectors.data(), a.get(), 1024 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    auto y = curaii::make_unique_device_ptr<double>(samples * 15);
    CUBLAS_CHECK(cublasDgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, samples, 15, 32, &one, x.get(),
                             samples, a.get() + 32, 32, &zero, y.get(), samples));
    reference_output.resize(samples * 15);
    CUDA_CHECK(cudaMemcpy(reference_output.data(), y.get(),
                          reference_output.size() * sizeof(double), cudaMemcpyDeviceToHost));
  }
  bool validate(bool compare_reference = true, int gram_k = 0) {

    if (gram_k) {
      std::vector<float> partial_host((samples / gram_k) * 1024);
      CUDA_CHECK(cudaMemcpy(partial_host.data(), partial.get(), partial_host.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      double error = 0, energy = 0;
      for (int i = 0; i < 1024; ++i) {
        float actual = 0;
        for (int part = 0; part < samples / gram_k; ++part)
          actual += partial_host[part * 1024 + i];
        double expected = reference_gram[i];
        error += (actual - expected) * (actual - expected);
        energy += expected * expected;
      }
      if (std::sqrt(error / std::max(energy, 1e-30)) >= .01)
        return false;
    }
    std::vector<float>  v(1024);
    std::vector<__half> y(samples * 15);
    CUDA_CHECK(cudaMemcpy(v.data(), matrix.get(), 1024 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), half.get(), y.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    double error = 0, energy = 0;
    for (int c = 0; c < 15; ++c) {
      double dot = 0;
      for (int feature = 0; feature < 32; ++feature)
        dot += v[(c + 1) * 32 + feature] * reference_vectors[(c + 1) * 32 + feature];
      double sign = dot < 0 ? -1 : 1;
      for (int i = 0; i < samples; ++i) {
        double actual   = sign * __half2float(y[c * samples + i]),
               expected = reference_output[c * samples + i];
        if (!std::isfinite(actual))
          return false;
        error += (actual - expected) * (actual - expected);
        energy += expected * expected;
      }
    }
    if (compare_reference && std::sqrt(error / std::max(energy, 1e-30)) >= .01)
      return false;
    if (compare_reference) {
      double error = 0, energy = 0;
      for (int a = 0; a < 32; ++a)
        for (int b = 0; b < 32; ++b) {
          double actual = 0, expected = 0;
          for (int col = 1; col < 16; ++col) {
            actual += double(v[col * 32 + a]) * v[col * 32 + b];
            expected += reference_vectors[col * 32 + a] * reference_vectors[col * 32 + b];
          }
          error += (actual - expected) * (actual - expected);
          energy += expected * expected;
        }
      if (std::sqrt(error / energy) >= .01)
        return false;
    }
    double ortho = 0;
    for (int a = 0; a < 32; ++a)
      for (int b = 0; b < 32; ++b) {
        double d = 0;
        for (int j = 0; j < 32; ++j)
          d += double(v[a * 32 + j]) * v[b * 32 + j];
        double e = d - (a == b ? 1 : 0);
        ortho += e * e;
      }
    if (std::sqrt(ortho / 32) > .01)
      return false;
    std::vector<float> eigen(32);
    CUDA_CHECK(cudaMemcpy(eigen.data(), values.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost));
    double residual = 0, norm = 0;
    for (int c = 0; c < 32; ++c)
      for (int row = 0; row < 32; ++row) {
        double av = 0;
        for (int j = 0; j < 32; ++j)
          av += reference_gram[j * 32 + row] * v[c * 32 + j];
        double e = av - eigen[c] * v[c * 32 + row];
        residual += e * e;
        norm += reference_gram[c * 32 + row] * reference_gram[c * 32 + row];
      }
    if (std::sqrt(residual / std::max(norm, 1e-30)) > .01)
      return false;
    curaii::CublasHandle blas;
    CUBLAS_CHECK(cublasSetStream(blas.get(), stream));
    float one = 1, zero = 0;
    CUBLAS_CHECK(cublasSgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, samples, 15, 32, &one,
                             static_cast<float *>(floats.data()), samples, matrix.get() + 32, 32,
                             &zero, static_cast<float *>(output.data()), samples));
    std::vector<float> expected(samples * 15);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(expected.data(), output.data(), expected.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    error = energy = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      double a = __half2float(y[i]), e = expected[i];
      if (!std::isfinite(a))
        return false;
      error += (a - e) * (a - e);
      energy += e * e;
    }
    if (!compare_reference)
      for (int i = 0; i < 32; ++i)
        energy = std::max(energy, reference_gram[i * 32 + i]);
    return std::sqrt(error / std::max(energy, 1e-30)) < .01;
  }
  Context() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    require(prop.major * 100 + prop.minor * 10 == PCA_BENCH_SM, "GPU architecture mismatch");
    host.resize(samples * 32);
    std::mt19937 rng(42);
    for (auto &v : host)
      v = static_cast<unsigned char>(rng() % 256);
    CUDA_CHECK(cudaMemcpy(bytes.data(), host.data(), host.size(), cudaMemcpyHostToDevice));
    holotask::syncs::ConversionFactory cf;
    holotask::syncs::PcaFactory        pf;
    conversion = cf.create(
        std::span<const TDesc>(&u.desc, 1),
        holotask::syncs::ConversionSettings{holotask::syncs::ConversionSettings::Target::F32,
                                            holotask::syncs::ConversionSettings::Strategy::Real},
        {stream});
    pca = pf.create(std::span<const TDesc>(&f.desc, 1), holotask::syncs::PcaSettings{1, 16},
                    {stream});
    execute(*conversion, u, f);
    execute(*pca, f, o);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  void execute(ISyncTask &task, TView &in, TView &out) {
    std::atomic<bool> cancel{false};
    SyncCtx           ctx{.inputs       = {&in, 1},
                          .outputs      = {&out, 1},
                          .cancelled    = &cancel,
                          .event_writer = nullptr,
                          .event_reader = nullptr};
    require(task.execute(ctx) == OpResult::Ok, "Task execution failed");
  }
  template <int K, int M, int TG, int TP> void add() {
    constexpr int  sa = 32 * K * 2;
    constexpr int  sb = (M * 32 + 32 * 16) * 2;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (sa > prop.sharedMemPerBlockOptin || sb > prop.sharedMemPerBlockOptin)
      return;
    if (sa > prop.sharedMemPerBlock)
      CUDA_CHECK(
          cudaFuncSetAttribute(gram<K, TG>, cudaFuncAttributeMaxDynamicSharedMemorySize, sa));
    if (sb > prop.sharedMemPerBlock)
      CUDA_CHECK(
          cudaFuncSetAttribute(project<M, TP>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb));
    candidates.push_back(
        {K, M, TG, TP, [this] {
           gram<K, TG><<<samples / K, TG, sa, stream>>>(static_cast<unsigned char *>(bytes.data()),
                                                        partial.get());
           reduce<<<4, 256, 0, stream>>>(partial.get(), matrix.get(), samples / K);
           runtime_solver->launch(matrix.get(), values.get(), info.get(), stream);
           fallback_solve<<<1, 1, 0, stream>>>(matrix.get(), values.get(), info.get());
           project<M, TP><<<samples / M, TP, sb, stream>>>(
               static_cast<unsigned char *>(bytes.data()), matrix.get(), half.get());
           CUDA_CHECK(cudaGetLastError());
         }});
  }
  template <int K, int M> void blocks() {
    add<K, M, 128, 128>();
    add<K, M, 128, 256>();
    add<K, M, 256, 128>();
    add<K, M, 256, 256>();
  }
  template <int K> void rows() {
    blocks<K, 64>();
    blocks<K, 128>();
    blocks<K, 256>();
  }
  void tune() {
    if (selected >= 0)
      return;
    runtime_solver = std::make_unique<RuntimeSolver>();
    reference();
    rows<128>();
    rows<256>();
    rows<512>();
    std::fprintf(stderr, "Tuning direct-U8 PCA (%zu candidates)...\n", candidates.size());
    double best = 1e30;
    for (int i = 0; i < int(candidates.size()); ++i) {
      Graph  graph(stream, candidates[i].launch);
      double ts[5];
      for (auto &t : ts)
        t = graph.time(stream);
      std::sort(ts, ts + 5);
      int status;
      CUDA_CHECK(cudaMemcpy(&status, info.get(), sizeof(int), cudaMemcpyDeviceToHost));
      require(status == 0, "Eigensolver failed");
      if (!validate(true, candidates[i].k))
        continue;
      if (ts[2] < best) {
        best     = ts[2];
        selected = i;
      }
    }
    require(selected >= 0, "No direct-U8 candidate");
  }
};
std::unique_ptr<Context> context;
void                     run(benchmark::State &state, int mode) {
  try {
    if (!context)
      context = std::make_unique<Context>();
    auto &c = *context;
    if (mode == 2)
      c.tune();
    auto launch = [&] {
      if (mode == 2)
        c.candidates[c.selected].launch();
      else {
        if (mode == 0)
          c.execute(*c.conversion, c.u, c.f);
        c.execute(*c.pca, c.f, c.o);
      }
    };
    Graph graph(c.stream, launch);
    if (mode == 2) {
      auto &v = c.candidates[c.selected];
      state.SetLabel("FP16 output K=" + std::to_string(v.k) + " M=" + std::to_string(v.m) +
                     " threads=" + std::to_string(v.tg) + "/" + std::to_string(v.tp));
    }
    for (auto _ : state)
      state.SetIterationTime(graph.time(c.stream));
    state.counters["windows/s"] =
        benchmark::Counter(1, benchmark::Counter::kIsIterationInvariantRate);
  } catch (const std::exception &e) {
    failed = true;
    state.SkipWithError(e.what());
  }
}
} // namespace
