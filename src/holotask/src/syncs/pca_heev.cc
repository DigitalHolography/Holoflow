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

#include "pca_heev.hh"

#include <cuda.h>

#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"

#include "logger.hh"
#include "pca_cusolverdx.hh"

namespace holotask::syncs::detail {

namespace {

constexpr int cusolver_fallback_min_features = 256;

CUcontext stream_context(cudaStream_t stream) {
  CUcontext  context = nullptr;
  const auto result  = stream != nullptr
                           ? cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context)
                           : cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context == nullptr) {
    throw std::runtime_error("[Pca] Failed to determine the CUDA stream context");
  }
  return context;
}

} // namespace

struct PcaHeevSolver::Impl {
  Impl(int n_features, size_t n_batch, float *covariance, float *eigenvalues, cudaStream_t stream)
      : n_features(n_features), n_batch(n_batch), context(stream_context(stream)) {
    if (n_features < cusolver_fallback_min_features) {
      try {
        cusolverdx = std::make_unique<PcaHeevKernel>(n_features, n_batch, stream);
        return;
      } catch (const std::exception &e) {
        logger()->warn("[Pca] cuSolverDx initialization failed for depth {}: {}\n"
                       "[Pca] Falling back to the conventional cuSOLVER eigensolver",
                       n_features, e.what());
      }
    }

    try {
      initialize_cusolver(covariance, eigenvalues, stream);
    } catch (const std::exception &e) {
      logger()->error("[Pca] Failed to initialize any GPU eigensolver for depth {}: {}", n_features,
                      e.what());
      throw std::runtime_error(
          std::format("PCA could not initialize a GPU eigensolver for depth {}. "
                      "See the terminal log for details.",
                      n_features));
    }
  }

  void initialize_cusolver(float *covariance, float *eigenvalues, cudaStream_t stream) {
    if (n_batch > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument("[Pca] Batch count exceeds the cuSOLVER API limit");
    }

    cusolver_handle = std::make_unique<curaii::CusolverDnHandle>();
    cusolver_params = std::make_unique<curaii::CusolverDnParams>();

    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
        cusolver_handle->get(), cusolver_params->get(), CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, n_features, CUDA_R_32F, covariance, n_features, CUDA_R_32F,
        eigenvalues, CUDA_R_32F, &device_workspace_size, &host_workspace_size,
        static_cast<int64_t>(n_batch)));

    if (device_workspace_size != 0) {
      device_workspace = curaii::make_unique_device_ptr<std::byte>(device_workspace_size);
    }
    host_workspace.resize(host_workspace_size);
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const {
    return stream_context(stream) == context;
  }

  void launch(float *covariance, float *eigenvalues, int *info, size_t batches,
              cudaStream_t stream) {
    if (batches != n_batch) {
      throw std::invalid_argument("[Pca] Eigensolver batch count changed after task creation");
    }
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument("[Pca] Eigensolver and PCA stream use different contexts");
    }

    if (cusolverdx != nullptr) {
      cusolverdx->launch(covariance, eigenvalues, info, batches, stream);
      return;
    }

    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        cusolver_handle->get(), cusolver_params->get(), CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, n_features, CUDA_R_32F, covariance, n_features, CUDA_R_32F,
        eigenvalues, CUDA_R_32F, device_workspace.get(), device_workspace_size,
        host_workspace.data(), host_workspace_size, info, static_cast<int64_t>(batches)));
  }

  int                                       n_features;
  size_t                                    n_batch;
  CUcontext                                 context{nullptr};
  std::unique_ptr<PcaHeevKernel>            cusolverdx;
  std::unique_ptr<curaii::CusolverDnHandle> cusolver_handle;
  std::unique_ptr<curaii::CusolverDnParams> cusolver_params;
  curaii::unique_device_ptr<std::byte>      device_workspace;
  size_t                                    device_workspace_size{0};
  std::vector<std::byte>                    host_workspace;
  size_t                                    host_workspace_size{0};
};

PcaHeevSolver::PcaHeevSolver(int n_features, size_t n_batch, float *covariance, float *eigenvalues,
                             cudaStream_t stream)
    : impl_(std::make_unique<Impl>(n_features, n_batch, covariance, eigenvalues, stream)) {}

PcaHeevSolver::~PcaHeevSolver() noexcept = default;

PcaHeevSolver::PcaHeevSolver(PcaHeevSolver &&) noexcept = default;

PcaHeevSolver &PcaHeevSolver::operator=(PcaHeevSolver &&) noexcept = default;

bool PcaHeevSolver::is_compatible_stream(cudaStream_t stream) const {
  return impl_->is_compatible_stream(stream);
}

void PcaHeevSolver::launch(float *covariance, float *eigenvalues, int *info, size_t n_batch,
                           cudaStream_t stream) {
  impl_->launch(covariance, eigenvalues, info, n_batch, stream);
}

} // namespace holotask::syncs::detail
