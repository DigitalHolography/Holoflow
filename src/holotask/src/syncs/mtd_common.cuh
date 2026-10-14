// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"

namespace holotask::syncs::mtd_detail {

using namespace holoflow::core;
template <typename T> using DevicePtr = curaii::unique_device_ptr<T>;
constexpr unsigned int block_size     = 256;
constexpr double       pi             = 3.14159265358979323846;

inline void require(bool valid, const std::string &message) {
  if (!valid)
    throw std::invalid_argument("MTD: " + message);
}

inline bool contiguous(const TDesc &desc) {
  if (desc.shape.size() != desc.strides.size())
    return false;
  size_t stride = size_of(desc.dtype);
  for (size_t i = desc.rank(); i-- > 0;) {
    if (desc.strides[i] != stride || desc.shape[i] == 0)
      return false;
    if (desc.shape[i] > std::numeric_limits<size_t>::max() / stride)
      return false;
    stride *= desc.shape[i];
  }
  return true;
}

inline unsigned int blocks(size_t size) {
  require(size <= static_cast<size_t>(std::numeric_limits<unsigned int>::max()) * block_size,
          "tensor exceeds supported CUDA grid size");
  return static_cast<unsigned int>((size + block_size - 1) / block_size);
}

inline bool cancelled(const SyncCtx &ctx) {
  return ctx.cancelled != nullptr && ctx.cancelled->load(std::memory_order_relaxed);
}

inline std::vector<float> window(int size, float alpha) {
  std::vector<float> result(static_cast<size_t>(size) * size);
  auto               tukey = [size, alpha](int index) {
    if (alpha == 0.0f)
      return 1.0;
    const double x = static_cast<double>(index) / (size - 1);
    if (x < alpha / 2.0)
      return 0.5 * (1 + std::cos(pi * (2 * x / alpha - 1)));
    if (x > 1 - alpha / 2.0)
      return 0.5 * (1 + std::cos(pi * (2 * x / alpha - 2 / alpha + 1)));
    return 1.0;
  };
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x)
      result[static_cast<size_t>(y) * size + x] = static_cast<float>(tukey(x) * tukey(y));
  return result;
}

inline void plan2(curaii::CufftHandle &plan, int size, cudaStream_t stream) {
  size_t work = 0;
  CUFFT_CHECK(cufftMakePlan2d(plan.get(), size, size, CUFFT_C2C, &work));
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));
}

inline void plan3(curaii::CufftHandle &plan, int size, cudaStream_t stream) {
  size_t work = 0;
  CUFFT_CHECK(cufftMakePlan3d(plan.get(), size, size, size, CUFFT_C2C, &work));
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));
}

// An even-sized 3D FFT shift pairs each lower-Z voxel with exactly one upper-Z voxel.
// Swapping these disjoint pairs avoids allocating another full volume.
static __global__ void shift3(cuFloatComplex *data, int dimension, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count / 2)
    return;
  const size_t dim = static_cast<size_t>(dimension);
  const size_t x = i % dim, y = (i / dim) % dim, z = i / (dim * dim);
  const size_t other = ((z + dim / 2) * dim + (y + dim / 2) % dim) * dim + (x + dim / 2) % dim;
  const auto   saved = data[i];
  data[i]            = data[other];
  data[other]        = saved;
}

static __global__ void scale(cuFloatComplex *data, size_t count, float factor) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i].x *= factor;
    data[i].y *= factor;
  }
}

inline void centered_transform(curaii::CufftHandle &plan, cuFloatComplex *data, int dimension,
                               size_t count, int direction, cudaStream_t stream) {
  shift3<<<blocks(count / 2), block_size, 0, stream>>>(data, dimension, count);
  CUFFT_CHECK(cufftExecC2C(plan.get(), data, data, direction));
  shift3<<<blocks(count / 2), block_size, 0, stream>>>(data, dimension, count);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace holotask::syncs::mtd_detail
