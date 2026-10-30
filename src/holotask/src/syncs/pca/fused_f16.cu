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

#include <cuda_fp16.h>
#include <cutlass/gemm/warp/default_mma_wmma_tensor_op.h>
namespace holotask::syncs::pca_f16 {
constexpr int samples = 320 * 512;
using Warp            = typename cutlass::gemm::warp::DefaultMmaTensorOpWmma<
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
    matrix[i] = v;
  }
}
template <int M, int Threads>
__global__ void project(const unsigned char *input, const float *vectors, __half *output, int begin,
                        int components) {
  extern __shared__ __align__(32) cutlass::half_t shared[];
  auto                                            a = shared;
  auto                                            b = a + M * 32;
  for (int i = threadIdx.x; i < M * 32; i += Threads)
    a[i] = cutlass::half_t(float(input[(i % 32) * samples + blockIdx.x * M + i / 32]));
  for (int i = threadIdx.x; i < 32 * 16; i += Threads)
    b[i] = cutlass::half_t(i / 32 < components ? vectors[(i / 32 + begin) * 32 + i % 32] : 0.f);
  __shared__ __align__(32) float result[M * 16];
  __syncthreads();
  int warp = threadIdx.x / 32;
  for (int row = warp * 16; row < M; row += (Threads / 32) * 16)
    multiply(a + row * 32, b, result + row, 32, 32, M, 32);
  __syncthreads();
  for (int i = threadIdx.x; i < M * components; i += Threads)
    output[(i / M) * samples + blockIdx.x * M + i % M] = __float2half(result[i]);
}

} // namespace holotask::syncs::pca_f16
#include "curaii/cuda.hh"
#include "fused_f16.hh"
namespace holotask::syncs::pca_f16 {
void covariance(const void *input, float *partial, float *matrices, size_t batches,
                cudaStream_t stream) {
  for (size_t b = 0; b < batches; ++b) {
    gram<512, 256><<<samples / 512, 256, 32 * 512 * 2, stream>>>(
        static_cast<const unsigned char *>(input) + b * 32 * samples,
        partial + b * (samples / 512) * 1024);
    reduce<<<4, 256, 0, stream>>>(partial + b * (samples / 512) * 1024, matrices + b * 1024,
                                  samples / 512);
  }
  CUDA_CHECK(cudaGetLastError());
}
void projection(const void *input, const float *vectors, void *output, size_t batches, int begin,
                int components, cudaStream_t stream) {
  for (size_t b = 0; b < batches; ++b)
    project<256, 256><<<samples / 256, 256, (256 * 32 + 32 * 16) * 2, stream>>>(
        static_cast<const unsigned char *>(input) + b * 32 * samples, vectors + b * 1024,
        static_cast<__half *>(output) + b * components * samples, begin, components);
  CUDA_CHECK(cudaGetLastError());
}
} // namespace holotask::syncs::pca_f16
