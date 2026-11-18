// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <math_constants.h>
struct Complex16 {
  __half real, imag;
};
constexpr int RowsPerBlock = 8;
constexpr int W = 512, H = 320, N = W * H, Spectrum = (W / 2 + 1) * H, Parts = 64;
template <int Size, cufftdx::fft_direction Direction, int Ept, int Batch>
using FFT = decltype(cufftdx::Block() + cufftdx::Size<Size>() +
                     cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Direction>() +
                     cufftdx::Precision<__half>() + cufftdx::FFTsPerBlock<Batch>() +
                     cufftdx::ElementsPerThread<Ept>() + cufftdx::SM<FFT_SM>());
using XF  = FFT<W, cufftdx::fft_direction::forward, 8, 8>;
using YF  = FFT<H, cufftdx::fft_direction::forward, 20, 2>;
using XI  = FFT<W, cufftdx::fft_direction::inverse, 8, 8>;
using YI  = FFT<H, cufftdx::fft_direction::inverse, 20, 2>;
__device__ float real_input(const void *in, int i) {
#if INPUT_HALF
  return __half2float(static_cast<const __half *>(in)[i]);
#else
  return static_cast<const float *>(in)[i];
#endif
}
extern "C" __global__ void mask_kernel(float *mask, int ri, int ro, int si, int so) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  int   x = i % W, y = i / W;
  float d = hypotf(float(x <= W / 2 ? x : x - W), float(y <= H / 2 ? y : y - H));
  float a =
      d < ro ? 1.f
             : (so > 0 && d < float(ro) + float(so) ? cosf((d - ro) / so * CUDART_PI_F / 2) : 0.f);
  float b =
      d < ri ? 1.f
             : (si > 0 && d < float(ri) + float(si) ? cosf((d - ri) / si * CUDART_PI_F / 2) : 0.f);
  mask[i] = a * (1 - b);
}
extern "C" __global__ void peak_kernel(const void *in, float *partial, int stage) {
  __shared__ float v[256];
  int              plane = blockIdx.x / Parts, part = blockIdx.x % Parts;
  float            peak = 0;
  for (int i = part * 256 + threadIdx.x; i < (stage == 0 ? N : Spectrum); i += Parts * 256) {
    float value;
    if (stage == 0)
      value = fabsf(real_input(in, plane * N + i));
    else {
      auto c = static_cast<const Complex16 *>(in)[plane * Spectrum + i];
      value  = fabsf(__half2float(c.real)) + fabsf(__half2float(c.imag));
    }
    peak = fmaxf(peak, value);
  }
  v[threadIdx.x] = peak;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      v[threadIdx.x] = fmaxf(v[threadIdx.x], v[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[plane * Parts + part] = v[0];
}
extern "C" __global__ void scale_kernel(const float *partial, int *scales, int stage) {
  __shared__ float v[64];
  int              p = blockIdx.x;
  v[threadIdx.x]     = partial[p * Parts + threadIdx.x];
  __syncthreads();
  for (int s = 32; s; s /= 2) {
    if (threadIdx.x < s)
      v[threadIdx.x] = fmaxf(v[threadIdx.x], v[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // Scale both upward and downward to retain small inputs without overflowing any FFT.
    float log_length      = stage == 0 ? 9.f : log2f(float(H));
    scales[2 * p + stage] = v[0] > 0 ? int(ceilf(log2f(v[0]) + log_length)) - 12 : 0;
  }
}

// Transposed intermediate keeps adjacent Y threads on adjacent samples.
extern "C" __global__ void forward_x(const void *in, Complex16 *out, const int *s) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto                                         *tile  = reinterpret_cast<Complex16 *>(shared);
  int                                           plane = blockIdx.x / (H / RowsPerBlock);
  int            row_base = blockIdx.x % (H / RowsPerBlock) * RowsPerBlock;
  int            lane = 2 * threadIdx.y, row = row_base + lane;
  int            thread = threadIdx.x + threadIdx.y * blockDim.x;
  XF::value_type data[XF::storage_size];
  for (int j = 0; j < XF::elements_per_thread; ++j) {
    int  x  = threadIdx.x + j * XF::stride;
    auto lo = __float2half_rn(ldexpf(real_input(in, plane * N + row * W + x), -s[2 * plane]));
    auto hi = __float2half_rn(ldexpf(real_input(in, plane * N + (row + 1) * W + x), -s[2 * plane]));
    data[j] = {__halves2half2(lo, hi), __float2half2_rn(0)};
  }
  XF().execute(data, shared);
  __syncthreads();
  for (int j = 0; j < XF::elements_per_thread; ++j) {
    int x                          = threadIdx.x + j * XF::stride;
    tile[lane * (W + 1) + x]       = {__low2half(data[j].x), __low2half(data[j].y)};
    tile[(lane + 1) * (W + 1) + x] = {__high2half(data[j].x), __high2half(data[j].y)};
  }
  __syncthreads();
  for (int i = thread; i < RowsPerBlock * (W / 2 + 1); i += blockDim.x * blockDim.y)
    out[plane * Spectrum + (i / RowsPerBlock) * H + row_base + i % RowsPerBlock] =
        tile[(i % RowsPerBlock) * (W + 1) + i / RowsPerBlock];
}

extern "C" __global__ void filter_y(const Complex16 *in, Complex16 *out, const int *s,
                                    const float *mask) {
  extern __shared__ __align__(16) unsigned char shared[];
  constexpr int                                 pairs = (W / 2 + 2) / 2;
  int            plane = blockIdx.x / pairs, xbase = 2 * (blockIdx.x % pairs);
  const float    scale = ldexpf(1.f, -s[2 * plane + 1]);
  YF::value_type data[YF::storage_size];
  for (int j = 0; j < YF::elements_per_thread; ++j) {
    int    y = threadIdx.x + j * YF::stride;
    __half r[2], im[2];
    for (int lane = 0; lane < 2; ++lane) {
      int  x   = xbase + lane;
      auto c   = x <= W / 2 ? in[plane * Spectrum + x * H + y] : Complex16{};
      r[lane]  = __float2half_rn(__half2float(c.real) * scale);
      im[lane] = __float2half_rn(__half2float(c.imag) * scale);
    }
    data[j] = {__halves2half2(r[0], r[1]), __halves2half2(im[0], im[1])};
  }
  YF().execute(data, shared);
  __syncthreads();
  for (int j = 0; j < YF::elements_per_thread; ++j) {
    int y = threadIdx.x + j * YF::stride;
    // 512 exceeds the inverse transform length (320), preserving the forward bound.
    auto gain = __floats2half2_rn(mask[y * W + xbase] / 512.f, mask[y * W + xbase + 1] / 512.f);
    data[j].x = __hmul2(data[j].x, gain);
    data[j].y = __hmul2(data[j].y, gain);
  }
  YI().execute(data, shared);
  for (int j = 0; j < YI::elements_per_thread; ++j) {
    int y                                 = threadIdx.x + j * YI::stride;
    out[plane * Spectrum + xbase * H + y] = {__low2half(data[j].x), __low2half(data[j].y)};
    if (xbase < W / 2)
      out[plane * Spectrum + (xbase + 1) * H + y] = {__high2half(data[j].x),
                                                     __high2half(data[j].y)};
  }
}

extern "C" __global__ void inverse_x(const Complex16 *in, void *out, const int *s,
                                     bool normalized) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto                                         *tile  = reinterpret_cast<Complex16 *>(shared);
  int                                           plane = blockIdx.x / (H / RowsPerBlock);
  int row_base = blockIdx.x % (H / RowsPerBlock) * RowsPerBlock;
  int lane = 2 * threadIdx.y, row = row_base + lane;
  int thread = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thread; i < RowsPerBlock * (W / 2 + 1); i += blockDim.x * blockDim.y)
    tile[(i % RowsPerBlock) * (W + 1) + i / RowsPerBlock] =
        in[plane * Spectrum + (i / RowsPerBlock) * H + row_base + i % RowsPerBlock];
  __syncthreads();
  XI::value_type data[XI::storage_size];
  for (int j = 0; j < XI::elements_per_thread; ++j) {
    int   x = threadIdx.x + j * XI::stride, bin = x <= W / 2 ? x : W - x;
    auto  lo = tile[lane * (W + 1) + bin], hi = tile[(lane + 1) * (W + 1) + bin];
    float sign = x > W / 2 ? -1.f : 1.f;
    // Divide by 512 before the inverse row transform, as for the column inverse.
    data[j] = {__floats2half2_rn(__half2float(lo.real) / 512.f, __half2float(hi.real) / 512.f),
               __floats2half2_rn(sign * __half2float(lo.imag) / 512.f,
                                 sign * __half2float(hi.imag) / 512.f)};
  }
  __syncthreads();
  XI().execute(data, shared);
  for (int j = 0; j < XI::elements_per_thread; ++j) {
    int   x        = threadIdx.x + j * XI::stride;
    int   exponent = s[2 * plane] + s[2 * plane + 1] + 18;
    float divisor  = normalized ? float(N) : 1.f;
    float lo       = ldexpf(__half2float(__low2half(data[j].x)) / divisor, exponent);
    float hi       = ldexpf(__half2float(__high2half(data[j].x)) / divisor, exponent);
#if INPUT_HALF
    static_cast<__half *>(out)[plane * N + row * W + x]       = __float2half_rn(lo);
    static_cast<__half *>(out)[plane * N + (row + 1) * W + x] = __float2half_rn(hi);
#else
    static_cast<float *>(out)[plane * N + row * W + x]       = lo;
    static_cast<float *>(out)[plane * N + (row + 1) * W + x] = hi;
#endif
  }
}
constexpr unsigned int               x_fft_shared  = XF::shared_memory_size > XI::shared_memory_size
                                                         ? XF::shared_memory_size
                                                         : XI::shared_memory_size;
constexpr unsigned int               x_tile_shared = RowsPerBlock * (W + 1) * sizeof(Complex16);
extern "C" __constant__ unsigned int x_shared =
    x_fft_shared > x_tile_shared ? x_fft_shared : x_tile_shared;
extern "C" __constant__ unsigned int y_shared = YF::shared_memory_size > YI::shared_memory_size
                                                    ? YF::shared_memory_size
                                                    : YI::shared_memory_size;
