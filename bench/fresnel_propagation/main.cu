// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftdx.hpp>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/fresnel_diffraction.hh"

namespace {
constexpr int X = 512, Y = 320, Z = 32, N = X * Y * Z;
constexpr int repetitions = 5, launches = 100;
struct alignas(4) HalfComplex {
  __half real, imag;
};
static_assert(sizeof(HalfComplex) == 4);

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// -------------------------------------------------------------------------------------------------
// FFT kernels and cached lens
// -------------------------------------------------------------------------------------------------

template <int Size, int Batch>
using BaseFFT =
    decltype(cufftdx::Block() + cufftdx::Size<Size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
             cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<__half>() +
             cufftdx::FFTsPerBlock<Batch>() + cufftdx::SM<860>());
template <int Size, int Elements, int Batch>
using TunedFFT = decltype(BaseFFT<Size, Batch>() + cufftdx::ElementsPerThread<Elements>());

__device__ float2 lens_value(int i) {
  float x = (i % X - X / 2) * 20e-6f, y = (i / X - Y / 2) * 20e-6f;
  float phase = 3.14159265358979323846f / (852e-9f * .1f) * (x * x + y * y);
  return make_float2(cosf(phase), sinf(phase));
}
__global__ void cache_lens(float2 *lens) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < X * Y)
    lens[i] = lens_value(i);
}
__global__ void prepare_reference(const __half *input, float2 *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    auto  lens = lens_value(i % (X * Y));
    float v    = __half2float(input[i]);
    output[i]  = make_float2(v * lens.x, v * lens.y);
  }
}

template <class F, bool Row, bool Transposed, bool Direct, bool Fused = false>
__global__ void fft_pass(const __half *input, const HalfComplex *intermediate, HalfComplex *output,
                         const float2 *lens) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto                                         *tile    = reinterpret_cast<HalfComplex *>(shared);
  constexpr int                                 batch   = F::ffts_per_block;
  int                                           tid     = threadIdx.x + blockDim.x * threadIdx.y;
  int                                           threads = blockDim.x * blockDim.y;
  int                                           group   = blockIdx.x % ((Row ? Y : X) / batch);
  int                                           z       = blockIdx.x / ((Row ? Y : X) / batch);
  if constexpr (Fused && !Row && !Transposed) {
    for (int t = tid; t < Y * batch; t += threads) {
      int y = t / batch, x = t % batch;
      tile[y * (batch + 1) + x] = intermediate[z * X * Y + y * X + group * batch + x];
    }
    __syncthreads();
  }
  typename F::value_type data[F::storage_size];
  constexpr int          pairs = F::ffts_per_block / F::implicit_type_batching;
  int                    pair  = blockIdx.x * pairs + threadIdx.y;
  int                    plane = pair / (Row ? Y / 2 : X / 2);
  int                    base  = 2 * (pair % (Row ? Y / 2 : X / 2));
  for (int j = 0; j < F::elements_per_thread; ++j) {
    int    k    = threadIdx.x + j * F::stride;
    __half r[2] = {}, im[2] = {};
    if (k < (Row ? X : Y)) {
      for (int b = 0; b < 2; ++b) {
        int i = plane * X * Y +
                (Row ? (base + b) * X + k : (Transposed ? (base + b) * Y + k : k * X + base + b));
        if constexpr (Row) {
          auto  l = lens[i % (X * Y)];
          float v = __half2float(input[i]);
          r[b]    = __float2half_rn(v * l.x);
          im[b]   = __float2half_rn(v * l.y);
        } else {
          auto value = (Fused && !Transposed) ? tile[k * (batch + 1) + 2 * threadIdx.y + b]
                                              : intermediate[i];
          r[b]       = value.real;
          im[b]      = value.imag;
        }
      }
    }
    if constexpr (Direct) {
      data[j] = {__halves2half2(r[0], r[1]), __halves2half2(im[0], im[1])};
    } else {
      // Preserve the original packing path as the control.
      data[j] = {__floats2half2_rn(__half2float(r[0]), __half2float(r[1])),
                 __floats2half2_rn(__half2float(im[0]), __half2float(im[1]))};
    }
  }
  if constexpr (Fused)
    __syncthreads();
  F().execute(data, shared);
  if constexpr (Fused)
    __syncthreads();
  for (int j = 0; j < F::elements_per_thread; ++j) {
    int k = threadIdx.x + j * F::stride;
    if (k < (Row ? X : Y)) {
      for (int b = 0; b < 2; ++b) {
        int i = plane * X * Y +
                (Row ? (base + b) * X + k : (Transposed ? (base + b) * Y + k : k * X + base + b));
        HalfComplex value = {b ? __high2half(data[j].x) : __low2half(data[j].x),
                             b ? __high2half(data[j].y) : __low2half(data[j].y)};
        if constexpr (Fused) {
          if constexpr (Row)
            tile[(2 * threadIdx.y + b) * (X + 1) + k] = value;
          else
            tile[k * (batch + 1) + 2 * threadIdx.y + b] = value;
        } else
          output[i] = value;
      }
    }
  }
  if constexpr (Fused) {
    __syncthreads();
    for (int t = tid; t < (Row ? X : Y) * batch; t += threads) {
      int k = t / batch, b = t % batch;
      if constexpr (Row)
        output[z * X * Y + k * Y + group * batch + b] = tile[b * (X + 1) + k];
      else
        output[z * X * Y + k * X + group * batch + b] = tile[k * (batch + 1) + b];
    }
  }
}

template <bool Forward> __global__ void transpose(const HalfComplex *input, HalfComplex *output) {
  __shared__ HalfComplex tile[32][33];
  constexpr int          width = Forward ? X : Y, height = Forward ? Y : X;
  int                    x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    if (x < width && y + j < height)
      tile[threadIdx.y + j][threadIdx.x] = input[blockIdx.z * X * Y + (y + j) * width + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    if (x < height && y + j < width)
      output[blockIdx.z * X * Y + (y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}

// -------------------------------------------------------------------------------------------------
// Measurement and configuration registry
// -------------------------------------------------------------------------------------------------

struct Timing {
  float low, median, high;
};
Timing measure(cudaStream_t stream, const std::function<void()> &enqueue) {
  for (int i = 0; i < 5; ++i)
    enqueue();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t     graph      = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  enqueue();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::array<float, repetitions> times{};
  for (float &time : times) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < launches; ++i)
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
    time /= launches;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  std::sort(times.begin(), times.end());
  return {times.front(), times[repetitions / 2], times.back()};
}
struct Configuration {
  int                   elements, batch, threads, registers, shared, block_x, block_y;
  std::function<void()> enqueue;
};
struct Buffers {
  __half      *input;
  HalfComplex *intermediate, *output, *transposed, *transformed;
  float2      *lens;
};

template <class F, bool Row, bool Transposed, bool Direct, bool Fused = false>
void register_configuration(std::vector<Configuration> &list, Buffers b, cudaStream_t stream,
                            const cudaDeviceProp &prop) {
  // Query support before referencing implementation traits or instantiating the kernel.
  if constexpr (cufftdx::is_supported<F, 860>::value) {
    if constexpr (!F::requires_workspace) {
      constexpr auto block = F::block_dim;
      constexpr int  shared =
          Fused ? std::max<int>(int(F::shared_memory_size),
                                 (Row ? F::ffts_per_block * (X + 1) : Y * (F::ffts_per_block + 1)) *
                                     int(sizeof(HalfComplex)))
                 : F::shared_memory_size;
      if (block.x * block.y * block.z > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
          shared > prop.sharedMemPerBlockOptin)
        return;
      auto kernel = fft_pass<F, Row, Transposed, Direct, Fused>;
      if (shared > prop.sharedMemPerBlock)
        CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
      cudaFuncAttributes attr{};
      CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
      if (block.x * block.y * block.z > static_cast<unsigned>(attr.maxThreadsPerBlock) ||
          attr.numRegs * block.x * block.y * block.z > static_cast<unsigned>(prop.regsPerBlock))
        return;
      list.push_back({F::elements_per_thread, F::ffts_per_block,
                      static_cast<int>(block.x * block.y * block.z), attr.numRegs, shared,
                      static_cast<int>(block.x), static_cast<int>(block.y), [=]() {
                        constexpr int grid = Z * (Row ? Y : X) / F::ffts_per_block;
                        kernel<<<grid, block, shared, stream>>>(
                            b.input, Transposed ? b.transposed : b.intermediate,
                            Row ? (Fused ? b.transposed : b.intermediate)
                                : (Transposed && !Fused ? b.transformed : b.output),
                            b.lens);
                        CUDA_CHECK(cudaGetLastError());
                      }});
    }
  }
}
template <bool Row, bool Transposed, int Elements, bool Fused = false>
void register_batches(std::vector<Configuration> &list, Buffers b, cudaStream_t stream,
                      const cudaDeviceProp &prop) {
  register_configuration<TunedFFT<Row ? X : Y, Elements, 2>, Row, Transposed, true, Fused>(
      list, b, stream, prop);
  register_configuration<TunedFFT<Row ? X : Y, Elements, 4>, Row, Transposed, true, Fused>(
      list, b, stream, prop);
  register_configuration<TunedFFT<Row ? X : Y, Elements, 8>, Row, Transposed, true, Fused>(
      list, b, stream, prop);
  register_configuration<TunedFFT<Row ? X : Y, Elements, 16>, Row, Transposed, true, Fused>(
      list, b, stream, prop);
}

// Reject oversized blocks before completing an FFT descriptor: cuFFTDx's
// support trait checks EPT support but does not reject every FPB combination.
template <bool Row, bool Transposed, int Elements, bool Fused>
void register_batch32(std::vector<Configuration> &list, Buffers b, cudaStream_t stream,
                      const cudaDeviceProp &prop) {
  if constexpr (((Row ? X : Y) / Elements) * (32 / 2) <= 1024)
    register_configuration<TunedFFT<Row ? X : Y, Elements, 32>, Row, Transposed, true, Fused>(
        list, b, stream, prop);
}
template <bool Row, bool Transposed, bool Fused>
void register_axis(std::vector<Configuration> &list, Buffers b, cudaStream_t stream,
                   const cudaDeviceProp &prop) {
  register_configuration<BaseFFT<Row ? X : Y, 2>, Row, Transposed, false, Fused>(list, b, stream,
                                                                                 prop);
  if constexpr (Row) {
    register_batches<Row, Transposed, 4, Fused>(list, b, stream, prop);
    register_batches<Row, Transposed, 8, Fused>(list, b, stream, prop);
    register_batches<Row, Transposed, 16, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 4, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 8, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 16, Fused>(list, b, stream, prop);
  } else {
    register_batches<Row, Transposed, 10, Fused>(list, b, stream, prop);
    register_batches<Row, Transposed, 20, Fused>(list, b, stream, prop);
    register_batches<Row, Transposed, 40, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 10, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 20, Fused>(list, b, stream, prop);
    register_batch32<Row, Transposed, 40, Fused>(list, b, stream, prop);
  }
}
int find_configuration(const std::vector<Configuration> &list, int elements, int batch) {
  for (int i = 0; i < static_cast<int>(list.size()); ++i)
    if (list[i].elements == elements && list[i].batch == batch)
      return i;
  throw std::runtime_error("Requested descriptor unavailable on this device");
}
enum class Layout { Strided, Transposed, YShared, FusedTransposed };
const char *name(Layout layout) {
  switch (layout) {
  case Layout::Strided:
    return "strided";
  case Layout::Transposed:
    return "transposed";
  case Layout::YShared:
    return "y-shared";
  case Layout::FusedTransposed:
    return "fused-transposed";
  }
  throw std::runtime_error("Invalid layout");
}
Layout parse_layout(const std::string &value) {
  for (auto layout :
       {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed})
    if (value == name(layout))
      return layout;
  throw std::runtime_error("Unknown layout");
}

struct Error {
  double relative, maximum;
  int    nonfinite;
};
Error compare(const std::vector<HalfComplex> &actual, const std::vector<float2> &reference) {
  double sum = 0, energy = 0, maximum = 0;
  int    nonfinite = 0;
  for (int i = 0; i < N; ++i) {
    double r = __half2float(actual[i].real), im = __half2float(actual[i].imag);
    require(std::isfinite(reference[i].x) && std::isfinite(reference[i].y),
            "Nonfinite FP32 reference");
    energy += double(reference[i].x) * reference[i].x + double(reference[i].y) * reference[i].y;
    if (!std::isfinite(r) || !std::isfinite(im)) {
      ++nonfinite;
      continue;
    }
    double dr = r - reference[i].x, di = im - reference[i].y;
    sum += dr * dr + di * di;
    maximum = std::max(maximum, std::sqrt(dr * dr + di * di));
  }
  double relative = energy > 0 ? std::sqrt(sum / energy) : (sum == 0 ? 0 : INFINITY);
  if (nonfinite)
    relative = maximum = INFINITY;
  return {relative, maximum, nonfinite};
}
struct Pipeline {
  int    x, y;
  Layout layout = Layout::Strided;
  Timing timing{};
  Error  error{};
};
} // namespace

int main(int argc, char **argv) {
  try {
    const bool study      = argc == 2 && std::string(argv[1]) == "--study-fused";
    const bool sweep      = argc == 1 || (argc == 2 && std::string(argv[1]) == "--sweep") || study;
    const bool selected   = argc == 5 && std::string(argv[1]) == "--config";
    const bool check_only = argc == 5 && std::string(argv[1]) == "--check-config";
    int        variant    = 0;
    if (!sweep && !selected && !check_only) {
      require(argc == 2 && std::strlen(argv[1]) == 1 && argv[1][0] >= '0' && argv[1][0] <= '4',
              "Usage: fresnel_propagation_bench [--sweep|--study-fused|0..4|--config layout X "
              "Y|--check-config layout X Y]");
      variant = argv[1][0] - '0';
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    require(prop.major == 8 && prop.minor == 6, "Benchmark requires SM86");
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
        if (check_only) {
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

    // Keep legacy indices and positional modes stable; fused modes have their
    // own registries and report their actual descriptors.
    std::vector<Pipeline> pipelines;
    if (sweep) {
      if (study) {
        pipelines.push_back(
            {find_configuration(xs, 8, 2), find_configuration(ys, 40, 16), Layout::Strided});
        pipelines.push_back(
            {find_configuration(xs, 8, 4), find_configuration(yts, 20, 2), Layout::Transposed});
      }
      for (auto layout :
           {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed}) {
        if (study && (layout == Layout::Strided || layout == Layout::Transposed))
          continue;
        if (!study && (layout == Layout::YShared || layout == Layout::FusedTransposed))
          continue;
        for (int x = 0; x < static_cast<int>(xconfigs(layout).size()); ++x)
          for (int y = 0; y < static_cast<int>(yconfigs(layout).size()); ++y)
            pipelines.push_back({x, y, layout});
      }
    } else if (selected || check_only) {
      auto index = [](const char *value) {
        size_t count  = 0;
        int    result = std::stoi(value, &count);
        require(count == std::strlen(value), "Invalid configuration index");
        return result;
      };
      auto layout = parse_layout(argv[2]);
      int  x = index(argv[3]), y = index(argv[4]);
      require(x >= 0 && x < static_cast<int>(xconfigs(layout).size()) && y >= 0 &&
                  y < static_cast<int>(yconfigs(layout).size()),
              "Configuration index out of range");
      pipelines.push_back({x, y, layout});
    } else {
      Layout layout =
          variant == 2
              ? Layout::Transposed
              : (variant == 3 ? Layout::YShared
                              : (variant == 4 ? Layout::FusedTransposed : Layout::Strided));
      int xe = (variant == 0 || variant == 3) ? 8 : 8;
      int xb = variant == 0 || variant == 3 ? 2 : (variant == 4 ? 16 : 4);
      int ye = variant == 0 || variant >= 3 ? 40 : 10;
      int yb = variant == 0 ? 2 : (variant >= 3 ? 16 : 4);
      pipelines.push_back({find_configuration(xconfigs(layout), xe, xb),
                           find_configuration(yconfigs(layout), ye, yb), layout});
    }
    auto describe_one = [&](const char *axis, int index, const Configuration &c, bool timing) {
      auto t = timing ? measure(stream, c.enqueue) : Timing{};
      std::printf("STAGE %s[%d] EPT=%d FPB=%d block=(%d,%d,1) registers=%d shared=%d "
                  "min/median/max=%.6f/%.6f/%.6f ms\n",
                  axis, index, c.elements, c.batch, c.block_x, c.block_y, c.registers, c.shared,
                  t.low, t.median, t.high);
    };
    upload(0);
    std::printf("%s; configurations X=%zu Y=%zu YT=%zu XF=%zu YF=%zu YS=%zu\n", prop.name,
                xs.size(), ys.size(), yts.size(), xf.size(), yf.size(), yshared.size());
    Timing baseline = check_only ? Timing{} : measure(stream, fp32);
    std::printf("FP32 min/median/max=%.6f/%.6f/%.6f ms\n", baseline.low, baseline.median,
                baseline.high);
    if (sweep) {
      for (auto layout :
           {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed}) {
        if (!study && (layout == Layout::YShared || layout == Layout::FusedTransposed))
          continue;
        const auto &xc = xconfigs(layout), &yc = yconfigs(layout);
        std::printf("LAYOUT %s\n", name(layout));
        for (int i = 0; i < static_cast<int>(xc.size()); ++i)
          describe_one("X", i, xc[i], true);
        xc[0].enqueue();
        if (layout == Layout::Transposed)
          forward();
        for (int i = 0; i < static_cast<int>(yc.size()); ++i)
          describe_one("Y", i, yc[i], true);
      }
      auto a = measure(stream, forward), b = measure(stream, backward);
      std::printf("STAGE transpose-forward=%.6f transpose-back=%.6f ms\n", a.median, b.median);
    }
    std::array<Pipeline, 4> best{};
    for (auto &p : best)
      p.timing.median = std::numeric_limits<float>::infinity();
    Pipeline control{};
    for (auto &p : pipelines) {
      if (!sweep) {
        describe_one("X", p.x, xconfigs(p.layout)[p.x], false);
        describe_one("Y", p.y, yconfigs(p.layout)[p.y], false);
      }
      p.error      = validate(p, false, false);
      p.timing     = check_only ? Timing{} : measure(stream, [&]() { enqueue(p); });
      auto &winner = best[static_cast<int>(p.layout)];
      if (p.timing.median < winner.timing.median)
        winner = p;
      if (p.layout == Layout::Strided &&
          ((study && p.x == 0 && p.y == 12) || (!study && p.x == 0 && p.y == 0)))
        control = p;
      std::printf(
          "PIPE %s X=%d Y=%d min/median/max=%.6f/%.6f/%.6f ms L2=%.8g max=%.8g nonfinite=%d\n",
          name(p.layout), p.x, p.y, p.timing.low, p.timing.median, p.timing.high, p.error.relative,
          p.error.maximum, p.error.nonfinite);
      std::fflush(stdout);
    }
    std::vector<Pipeline> finalists;
    if (sweep) {
      if (!study)
        finalists.push_back(control);
      for (const auto &p : best)
        if (std::isfinite(p.timing.median))
          finalists.push_back(p);
    } else
      finalists = pipelines;
    const char *cases[] = {"sine",     "random",         "zero",
                           "constant", "batch-impulses", "overflow-stress"};
    for (int test = 0; test < 6; ++test) {
      upload(test);
      for (const auto &p : finalists) {
        auto e = validate(p, test == 5, true);
        std::printf("VALIDATE %s %s X=%d Y=%d L2=%.8g max=%.8g nonfinite=%d\n", cases[test],
                    name(p.layout), p.x, p.y, e.relative, e.maximum, e.nonfinite);
      }
    }
    if (sweep)
      for (int test = 1; test < 5; ++test) {
        upload(test);
        for (auto layout :
             {Layout::Strided, Layout::Transposed, Layout::YShared, Layout::FusedTransposed}) {
          if (!study && (layout == Layout::YShared || layout == Layout::FusedTransposed))
            continue;
          for (int x = 0; x < static_cast<int>(xconfigs(layout).size()); ++x)
            (void)validate({x, 0, layout}, false, false);
          for (int y = 0; y < static_cast<int>(yconfigs(layout).size()); ++y)
            (void)validate({0, y, layout}, false, false);
        }
      }
    if (sweep) {
      Pipeline winner = finalists.front();
      for (const auto &p : finalists) {
        const auto &x = xconfigs(p.layout)[p.x], &y = yconfigs(p.layout)[p.y];
        std::printf("FINAL %s X=%d Y=%d X(EPT=%d,FPB=%d) Y(EPT=%d,FPB=%d) median=%.6f ms "
                    "speedup-v-control=%.4f speedup-v-FP32=%.4f\n",
                    name(p.layout), p.x, p.y, x.elements, x.batch, y.elements, y.batch,
                    p.timing.median, control.timing.median / p.timing.median,
                    baseline.median / p.timing.median);
        if (p.timing.median < winner.timing.median)
          winner = p;
      }
      std::printf("SELECTED %s X=%d Y=%d (validated minimum median full pipeline)\n",
                  name(winner.layout), winner.x, winner.y);
    }
    std::puts("All validation checks passed.");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
