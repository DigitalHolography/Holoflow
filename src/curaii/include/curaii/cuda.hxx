#pragma once

#ifndef CURAII_CUDA_HXX_INCLUDED
#include "curaii/cuda.hh"
#endif

namespace curaii {

template <typename T> [[nodiscard]] unique_host_ptr<T> make_unique_host_ptr(size_t count) {
  T *p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, count * sizeof(T)));
  return unique_host_ptr<T>(p);
}

template <typename T>
[[nodiscard]] unique_device_ptr<T> make_unique_device_ptr(size_t count, cudaStream_t stream) {
  T *p = nullptr;
  // CUDA_CHECK(cudaMallocAsync(&p, count * sizeof(T), stream));
  (void)stream; // Currently unused, but will be needed for async allocations
  CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
  return unique_device_ptr<T>(p);
}

} // namespace curaii