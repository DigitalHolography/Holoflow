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
#include <cuda_fp16.h>
#include <cutlass/gemm/warp/default_mma_wmma_tensor_op.h>
namespace pca_bench {
constexpr int features = 32, samples = 320 * 512, components = 15;
using Warp = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
    cutlass::gemm::GemmShape<16, 16, 16>, cutlass::gemm::GemmShape<16, 16, 16>, cutlass::half_t,
    cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor, float,
    cutlass::layout::ColumnMajor>::Type;
__device__ void multiply(const cutlass::half_t *a, const cutlass::half_t *b, float *c, int lda,
                         int ldb, int ldc, int k) {
  int             lane = threadIdx.x % 32;
  Warp::IteratorA ia({const_cast<cutlass::half_t *>(a), cutlass::layout::RowMajor(lda)}, lane);
  Warp::IteratorB ib({const_cast<cutlass::half_t *>(b), cutlass::layout::ColumnMajor(ldb)}, lane);
  Warp::FragmentC accum;
  accum.clear();
  for (int j = 0; j < k / 16; ++j) {
    Warp::FragmentA fa;
    Warp::FragmentB fb;
    ia.load(fa);
    ib.load(fb);
    Warp()(accum, fa, fb, accum);
    ++ia;
    ++ib;
  }
  Warp::IteratorC ic({c, cutlass::layout::ColumnMajor(ldc)}, lane);
  ic.store(accum);
}
template <int K, int Threads> __global__ void gram(const unsigned char *input, float *partial) {
  extern __shared__ __align__(32) cutlass::half_t tile[];
  for (int i = threadIdx.x; i < 32 * K; i += Threads)
    tile[i] = cutlass::half_t(float(input[(i / K) * samples + blockIdx.x * K + i % K]));
  __syncthreads();
  int warp = threadIdx.x / 32;
  if (warp < 4)
    multiply(tile + (warp % 2) * 16 * K, tile + (warp / 2) * 16 * K,
             partial + blockIdx.x * 1024 + (warp % 2) * 16 + (warp / 2) * 16 * 32, K, K, 32, K);
}
__global__ void reduce(const float *partial, float *matrix, int parts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 1024) {
    float v = 0;
    for (int p = 0; p < parts; ++p)
      v += partial[p * 1024 + i];
    matrix[i] = v * (1.f / 17179869184.f);
  }
}
template <int M, int Threads>
__global__ void project(const unsigned char *input, const float *vectors, __half *output) {
  extern __shared__ __align__(32) cutlass::half_t shared[];
  auto                                            a = shared;
  auto                                            b = a + M * 32;
  for (int i = threadIdx.x; i < M * 32; i += Threads)
    a[i] = cutlass::half_t(float(input[(i % 32) * samples + blockIdx.x * M + i / 32]));
  for (int i = threadIdx.x; i < 32 * 16; i += Threads)
    b[i] = cutlass::half_t(i / 32 < 15 ? vectors[(i / 32 + 1) * 32 + i % 32] : 0.f);
  __shared__ __align__(32) float result[M * 16];
  __syncthreads();
  int warp = threadIdx.x / 32;
  for (int row = warp * 16; row < M; row += (Threads / 32) * 16)
    multiply(a + row * 32, b, result + row, 32, 32, M, 32);
  __syncthreads();
  for (int i = threadIdx.x; i < M * 15; i += Threads)
    output[(i / M) * samples + blockIdx.x * M + i % M] = __float2half(result[i]);
}
// The vendor solver leaves the scaled Gram matrix intact on failure.
__global__ void fallback_solve(float *matrix, float *values, int *info) {
  if (*info == 0)
    return;
  float d[1024], q[1024];

#pragma unroll 1
  for (int i = 0; i < 1024; ++i) {
    d[i] = matrix[i];
    q[i] = (i / 32 == i % 32 ? 1.f : 0.f);
  }
  // An exact rank-one Gram matrix has an analytic orthonormal eigenbasis.
  int pivot = 0;

#pragma unroll 1
  for (int i = 1; i < 32; ++i)
    if (d[i * 32 + i] > d[pivot * 32 + pivot])
      pivot = i;
  float trace_value = 0;

#pragma unroll 1
  for (int i = 0; i < 32; ++i)
    trace_value += d[i * 32 + i];
  float rank_error = 0, rank_energy = 0;

#pragma unroll 1
  for (int i = 0; i < 32; ++i)

#pragma unroll 1
    for (int j = 0; j < 32; ++j) {
      float expected = d[pivot * 32 + i] * d[pivot * 32 + j] / d[pivot * 32 + pivot],
            e        = d[j * 32 + i] - expected;
      rank_error += e * e;
      rank_energy += d[j * 32 + i] * d[j * 32 + i];
    }
  if (rank_error <= 1e-12f * rank_energy) {
    float u[32], norm = 0;

#pragma unroll 1
    for (int i = 0; i < 32; ++i) {
      u[i] = d[pivot * 32 + i];
      norm += u[i] * u[i];
    }
    norm = sqrtf(norm);

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
      u[i] = (i == 31 ? 1.f : 0.f) - u[i] / norm;
    norm = 0;

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
      norm += u[i] * u[i];

#pragma unroll 1
    for (int i = 0; i < 32; ++i)

#pragma unroll 1
      for (int j = 0; j < 32; ++j)
        matrix[j * 32 + i] =
            (i == j ? 1.f : 0.f) - (norm > 1e-20f ? 2.f * u[i] * u[j] / norm : 0.f);

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
      values[i] = i == 31 ? trace_value * 17179869184.f : 0.f;
    *info = 0;
    return;
  }
  bool converged = false;

#pragma unroll 1
  for (int sweep = 0; sweep < 64; ++sweep) {
    float maximum = 0, diag = 0;

#pragma unroll 1
    for (int i = 0; i < 32; ++i)
      diag = fmaxf(diag, fabsf(d[i * 32 + i]));

#pragma unroll 1
    for (int p = 0; p < 31; ++p)

#pragma unroll 1
      for (int r = p + 1; r < 32; ++r) {
        float off = d[r * 32 + p];
        maximum   = fmaxf(maximum, fabsf(off));
        if (fabsf(off) <= 1e-7f * fmaxf(diag, 1e-30f))
          continue;
        float angle = .5f * atan2f(2.f * off, d[r * 32 + r] - d[p * 32 + p]), c = cosf(angle),
              s = sinf(angle), app = d[p * 32 + p], arr = d[r * 32 + r];

#pragma unroll 1
        for (int k = 0; k < 32; ++k) {
          if (k != p && k != r) {
            float u = d[p * 32 + k], v = d[r * 32 + k];
            d[p * 32 + k] = d[k * 32 + p] = c * u - s * v;
            d[r * 32 + k] = d[k * 32 + r] = s * u + c * v;
          }
          float u = q[p * 32 + k], v = q[r * 32 + k];
          q[p * 32 + k] = c * u - s * v;
          q[r * 32 + k] = s * u + c * v;
        }
        d[p * 32 + p] = c * c * app - 2 * s * c * off + s * s * arr;
        d[r * 32 + r] = s * s * app + 2 * s * c * off + c * c * arr;
        d[r * 32 + p] = d[p * 32 + r] = 0;
      }
    if (maximum <= 1e-7f * fmaxf(diag, 1e-30f)) {
      converged = true;
      break;
    }
  }

#pragma unroll 1
  for (int p = 0; p < 31; ++p) {
    int best = p;

#pragma unroll 1
    for (int r = p + 1; r < 32; ++r)
      if (d[r * 32 + r] < d[best * 32 + best])
        best = r;
    if (best != p) {
      float tmp           = d[p * 32 + p];
      d[p * 32 + p]       = d[best * 32 + best];
      d[best * 32 + best] = tmp;

#pragma unroll 1
      for (int k = 0; k < 32; ++k) {
        tmp              = q[p * 32 + k];
        q[p * 32 + k]    = q[best * 32 + k];
        q[best * 32 + k] = tmp;
      }
    }
  }

#pragma unroll 1
  for (int i = 0; i < 1024; ++i)
    matrix[i] = q[i];

#pragma unroll 1
  for (int i = 0; i < 32; ++i)
    values[i] = d[i * 32 + i] * 17179869184.f;
  *info = converged ? 0 : 1;
}
} // namespace pca_bench
