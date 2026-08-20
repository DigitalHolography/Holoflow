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

#include <array>
#include <utility>

#include <cuda_runtime.h>

#include "curaii/cuda.hh"

namespace holotask::syncs::detail {

class ShackHartmannCudaGraph {
public:
  using Addresses = std::array<const void *, 3>;

  ShackHartmannCudaGraph() = default;
  ~ShackHartmannCudaGraph() noexcept { reset(); }

  ShackHartmannCudaGraph(const ShackHartmannCudaGraph &)            = delete;
  ShackHartmannCudaGraph &operator=(const ShackHartmannCudaGraph &) = delete;

  [[nodiscard]] bool matches(const Addresses &addresses) const noexcept {
    return executable_ != nullptr && addresses_ == addresses;
  }

  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

  void reset() noexcept {
    if (executable_ != nullptr) {
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
      executable_ = nullptr;
    }
    addresses_ = {};
  }

  template <typename Enqueue>
  bool capture(cudaStream_t stream, const Addresses &addresses, Enqueue &&enqueue) {
    cudaGraph_t graph     = nullptr;
    bool        capturing = false;
    try {
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      capturing          = true;
      const bool complete = std::forward<Enqueue>(enqueue)();
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      capturing = false;

      if (!complete) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
        graph = nullptr;
        return false;
      }

      cudaGraphExec_t executable = nullptr;
      try {
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
      } catch (...) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
        graph = nullptr;
        throw;
      }
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
      graph = nullptr;

      reset();
      executable_ = executable;
      addresses_  = addresses;
      return true;
    } catch (...) {
      if (capturing) {
        cudaGraph_t discarded = nullptr;
        (void)cudaStreamEndCapture(stream, &discarded);
        if (discarded != nullptr) {
          CUDA_CHECK_NT(cudaGraphDestroy(discarded));
        }
      } else if (graph != nullptr) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
      }
      throw;
    }
  }

private:
  cudaGraphExec_t executable_ = nullptr;
  Addresses       addresses_  = {};
};

} // namespace holotask::syncs::detail
