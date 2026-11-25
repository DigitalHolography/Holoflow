// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <benchmark/benchmark.h>
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
struct GraphResources {
  cudaGraph_t     graph{};
  cudaGraphExec_t exec{};
  cudaEvent_t     begin{}, end{};
  ~GraphResources() {
    if (begin)
      cudaEventDestroy(begin);
    if (end)
      cudaEventDestroy(end);
    if (exec)
      cudaGraphExecDestroy(exec);
    if (graph)
      cudaGraphDestroy(graph);
  }
};
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
             cufftdx::FFTsPerBlock<Batch>() + cufftdx::SM<FRESNEL_BENCH_SM>());
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
  if constexpr (cufftdx::is_supported<F, FRESNEL_BENCH_SM>::value) {
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
