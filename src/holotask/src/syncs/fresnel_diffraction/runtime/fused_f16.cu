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
#include <cufftdx.hpp>
#include <math_constants.h>
struct alignas(4) Complex16 {
  __half real, imag;
};
using XF =
    decltype(cufftdx::Block() + cufftdx::Size<512>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
             cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<__half>() +
             cufftdx::FFTsPerBlock<8>() + cufftdx::ElementsPerThread<8>() + cufftdx::SM<FFT_SM>());
using YF =
    decltype(cufftdx::Block() + cufftdx::Size<320>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
             cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<__half>() +
             cufftdx::FFTsPerBlock<4>() + cufftdx::ElementsPerThread<20>() + cufftdx::SM<FFT_SM>());
constexpr int    W = 512, H = 320, WH = W * H, Chunk = REDUCE_MEAN ? 4 : 1,
                 G = (COMPONENTS + Chunk - 1) / Chunk;
__device__ float read_input(const void *input, int i) {
#if INPUT_HALF
  return __half2float(static_cast<const __half *>(input)[i]);
#else
  return static_cast<const float *>(input)[i];
#endif
}
extern "C" __global__ void fused_f16_lens(float2 *lens, float wavelength, float pitch,
                                          float distance) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < WH) {
    float x = (i % W - W / 2) * pitch, y = (i / W - H / 2) * pitch;
    float phase = CUDART_PI_F / (wavelength * distance) * (x * x + y * y);
    lens[i]     = make_float2(cosf(phase), sinf(phase));
  }
}
// Split the plane reduction so the full image scan is not restricted to one block per plane.
constexpr int              ScaleParts = 64;
extern "C" __global__ void fused_f16_scale_partial(const void *input, const float2 *lens,
                                                   float *partial) {
  __shared__ float maximum[256];
  int              z = blockIdx.x / ScaleParts, part = blockIdx.x % ScaleParts;
  float            peak = 0;
  for (int i = part * 256 + threadIdx.x; i < WH; i += ScaleParts * 256) {
    float v = read_input(input, z * WH + i);
    peak    = fmaxf(peak, fabsf(v * lens[i].x) + fabsf(v * lens[i].y));
  }
  maximum[threadIdx.x] = peak;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x], maximum[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[z * ScaleParts + part] = maximum[0];
}
extern "C" __global__ void fused_f16_scale(const float *partial, int *scales) {
  __shared__ float maximum[256];
  int              z   = blockIdx.x;
  maximum[threadIdx.x] = threadIdx.x < ScaleParts ? partial[z * ScaleParts + threadIdx.x] : 0.f;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x], maximum[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float log_peak    = maximum[0] > 0 ? log2f(maximum[0]) : -1000.f;
    int   sx          = max(0, int(ceilf(log_peak + 9.f - 14.f)));
    int   sy          = max(0, int(ceilf(log_peak + 9.f + log2f(320.f) - sx - 14.f)));
    scales[2 * z]     = sx;
    scales[2 * z + 1] = sy;
  }
}
extern "C" __global__ void fused_f16_fft_x(const void *input, const float2 *lens, const int *scales,
                                           Complex16 *intermediate) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto                                          tile = reinterpret_cast<Complex16 *>(shared);
  int            group = blockIdx.x % (H / 8), z = blockIdx.x / (H / 8),
                 tid = threadIdx.x + blockDim.x * threadIdx.y;
  XF::value_type data[XF::storage_size];
  for (int j = 0; j < XF::elements_per_thread; ++j) {
    int    x = threadIdx.x + j * XF::stride;
    __half real[2], imag[2];
    for (int b = 0; b < 2; ++b) {
      int    y = group * 8 + 2 * threadIdx.y + b;
      float  v = read_input(input, z * WH + y * W + x);
      float2 l = lens[y * W + x];
      real[b]  = __float2half_rn(ldexpf(v * l.x, -scales[2 * z]));
      imag[b]  = __float2half_rn(ldexpf(v * l.y, -scales[2 * z]));
    }
    data[j] = {__halves2half2(real[0], real[1]), __halves2half2(imag[0], imag[1])};
  }
  XF().execute(data, shared);
  __syncthreads();
  for (int j = 0; j < XF::elements_per_thread; ++j) {
    int x                                     = threadIdx.x + j * XF::stride;
    tile[(2 * threadIdx.y) * (W + 1) + x]     = {__low2half(data[j].x), __low2half(data[j].y)};
    tile[(2 * threadIdx.y + 1) * (W + 1) + x] = {__high2half(data[j].x), __high2half(data[j].y)};
  }
  __syncthreads();
  for (int t = tid; t < W * 8; t += blockDim.x * blockDim.y) {
    int x = t / 8, y = t % 8;
    intermediate[z * WH + x * H + group * 8 + y] = tile[y * (W + 1) + x];
  }
}
extern "C" __global__ void fused_f16_fft_y_mean(const Complex16 *intermediate, const int *scales,
                                                float *partial) {
  extern __shared__ __align__(16) unsigned char shared[];
  int   group = blockIdx.x / (W / 4), xbase = (blockIdx.x % (W / 4)) * 4, outer = group / G,
        cg                                = group % G;
  float accum[YF::elements_per_thread][2] = {};
  for (int component = cg * Chunk; component < min((cg + 1) * Chunk, COMPONENTS); ++component) {
    int            z = outer * COMPONENTS + component;
    YF::value_type data[YF::storage_size];
    for (int j = 0; j < YF::elements_per_thread; ++j) {
      int    y = threadIdx.x + j * YF::stride;
      __half real[2], imag[2];
      for (int b = 0; b < 2; ++b) {
        auto v  = intermediate[z * WH + (xbase + 2 * threadIdx.y + b) * H + y];
        real[b] = __float2half_rn(ldexpf(__half2float(v.real), -scales[2 * z + 1]));
        imag[b] = __float2half_rn(ldexpf(__half2float(v.imag), -scales[2 * z + 1]));
      }
      data[j] = {__halves2half2(real[0], real[1]), __halves2half2(imag[0], imag[1])};
    }
    YF().execute(data, shared);
    __syncthreads();
    for (int j = 0; j < YF::elements_per_thread; ++j) {
      accum[j][0] +=
          ldexpf(hypotf(__half2float(__low2half(data[j].x)), __half2float(__low2half(data[j].y))),
                 scales[2 * z] + scales[2 * z + 1]) /
          (REDUCE_MEAN ? COMPONENTS : 1);
      accum[j][1] +=
          ldexpf(hypotf(__half2float(__high2half(data[j].x)), __half2float(__high2half(data[j].y))),
                 scales[2 * z] + scales[2 * z + 1]) /
          (REDUCE_MEAN ? COMPONENTS : 1);
    }
    __syncthreads();
  }
  auto tile = reinterpret_cast<float *>(shared);
  for (int j = 0; j < YF::elements_per_thread; ++j) {
    int y                             = threadIdx.x + j * YF::stride;
    tile[y * 5 + 2 * threadIdx.y]     = accum[j][0];
    tile[y * 5 + 2 * threadIdx.y + 1] = accum[j][1];
  }
  __syncthreads();
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  for (int t = tid; t < H * 4; t += blockDim.x * blockDim.y) {
    int y = t / 4, x = t % 4;
    partial[group * WH + y * W + xbase + x] = tile[y * 5 + x];
  }
}
extern "C" __global__ void fused_f16_finish(const float *partial, float *output, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
#if REDUCE_MEAN
    int   outer = i / WH, local = i % WH;
    float sum = 0;
    for (int g = 0; g < G; ++g)
      sum += partial[(outer * G + g) * WH + local];
    output[i] = sum;
#else
    output[i] = partial[i];
#endif
  }
}
extern "C" __constant__ unsigned int fused_x_shared =
    XF::shared_memory_size > 16416 ? XF::shared_memory_size : 16416;
extern "C" __constant__ unsigned int fused_y_shared =
    YF::shared_memory_size > 6400 ? YF::shared_memory_size : 6400;
