// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0.
#pragma once
#include "curaii/cuda.hh"
#include <utility>

namespace holotask::detail {
template <typename Enqueue> cudaGraph_t capture_cuda_graph(cudaStream_t stream, Enqueue &&enqueue) {
  cudaGraph_t graph     = nullptr;
  bool        capturing = false;

  try {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    capturing = true;

    std::forward<Enqueue>(enqueue)();

    const auto end_result = cudaStreamEndCapture(stream, &graph);
    capturing             = false;
    CUDA_CHECK(end_result);
    return graph;
  } catch (...) {
    if (capturing) {
      cudaGraph_t discarded = nullptr;
      (void)cudaStreamEndCapture(stream, &discarded);
      if (discarded != nullptr) {
        (void)cudaGraphDestroy(discarded);
      }
    } else if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    (void)cudaGetLastError();
    throw;
  }
}

} // namespace holotask::detail
