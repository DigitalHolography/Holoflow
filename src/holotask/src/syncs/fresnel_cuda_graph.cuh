// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <array>
#include <utility>

#include "curaii/cuda.hh"

namespace holotask::syncs::detail {

class FresnelCudaGraph {
public:
  using Addresses = std::array<const void *, 2>;

  FresnelCudaGraph() = default;
  ~FresnelCudaGraph() noexcept {
    if (executable_ != nullptr)
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
  }

  FresnelCudaGraph(const FresnelCudaGraph &)            = delete;
  FresnelCudaGraph &operator=(const FresnelCudaGraph &) = delete;

  [[nodiscard]] bool matches(const Addresses &addresses) const noexcept {
    return addresses_ == addresses;
  }

  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

  template <typename Enqueue>
  FresnelCudaGraph(cudaStream_t stream, const Addresses &addresses, Enqueue &&enqueue)
      : addresses_(addresses) {
    cudaGraph_t graph     = nullptr;
    bool        capturing = false;
    try {
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      std::forward<Enqueue>(enqueue)();
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      capturing = false;
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable_, graph, 0));
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
    } catch (...) {
      if (capturing) {
        cudaGraph_t discarded = nullptr;
        (void)cudaStreamEndCapture(stream, &discarded);
        if (discarded != nullptr)
          CUDA_CHECK_NT(cudaGraphDestroy(discarded));
      } else if (graph != nullptr) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
      }
      if (executable_ != nullptr) {
        CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
        executable_ = nullptr;
      }
      throw;
    }
  }

private:
  cudaGraphExec_t executable_ = nullptr;
  Addresses       addresses_  = {};
};

} // namespace holotask::syncs::detail
