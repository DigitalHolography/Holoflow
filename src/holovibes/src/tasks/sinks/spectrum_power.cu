// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "spectrum_power.hh"

#include "curaii/cuda.hh"

namespace holovibes::tasks::sinks {
namespace {

constexpr unsigned int kBlockSize = 256;

__global__ void partial_power(const cuFloatComplex *input, float *scratch, size_t batches,
                              size_t bins, size_t pixels) {
  __shared__ float sums[kBlockSize];
  const size_t     bin     = blockIdx.x;
  const size_t     partial = blockIdx.y;
  float            sum     = 0.0f;
  for (size_t sample = partial * blockDim.x + threadIdx.x; sample < batches * pixels;
       sample += kSpectrumPowerPartials * blockDim.x) {
    const auto value = input[((sample / pixels) * bins + bin) * pixels + sample % pixels];
    sum += value.x * value.x + value.y * value.y;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kBlockSize / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scratch[bin * kSpectrumPowerPartials + partial] = sums[0];
  }
}

__global__ void finish_power(const float *scratch, float *output, size_t samples) {
  __shared__ float   sums[kSpectrumPowerPartials];
  const unsigned int tid = threadIdx.x;
  sums[tid]              = scratch[blockIdx.x * kSpectrumPowerPartials + tid];
  __syncthreads();
  for (unsigned int stride = static_cast<unsigned int>(kSpectrumPowerPartials) / 2; stride > 0;
       stride /= 2) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sums[0] / static_cast<float>(samples);
  }
}

} // namespace

void compute_spectrum_power(const cuFloatComplex *input, float *scratch, float *output,
                            size_t batches, size_t bins, size_t pixels, cudaStream_t stream) {
  const auto count = static_cast<unsigned int>(bins);
  partial_power<<<dim3(count, static_cast<unsigned int>(kSpectrumPowerPartials)), kBlockSize, 0,
                  stream>>>(input, scratch, batches, bins, pixels);
  CUDA_CHECK(cudaGetLastError());
  finish_power<<<count, static_cast<unsigned int>(kSpectrumPowerPartials), 0, stream>>>(
      scratch, output, batches * pixels);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace holovibes::tasks::sinks
