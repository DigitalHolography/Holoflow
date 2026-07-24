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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace holotask::syncs::detail {

class PcaHeevKernel {
public:
  PcaHeevKernel(int n_features, int n_samples, size_t n_batch, cudaStream_t stream);
  ~PcaHeevKernel() noexcept;

  PcaHeevKernel(const PcaHeevKernel &)            = delete;
  PcaHeevKernel &operator=(const PcaHeevKernel &) = delete;

  PcaHeevKernel(PcaHeevKernel &&) noexcept;
  PcaHeevKernel &operator=(PcaHeevKernel &&) noexcept;

  [[nodiscard]] bool   is_compatible_stream(cudaStream_t stream) const;
  [[nodiscard]] size_t covariance_partial_count() const;

  void launch_covariance(const std::uint8_t *input, float *partials, size_t n_batch,
                         cudaStream_t stream) const;
  void launch_covariance_reduction(const float *partials, float *covariance, size_t n_batch,
                                   cudaStream_t stream) const;

  void launch(float *covariance, float *eigenvalues, int *info, size_t n_batch,
              cudaStream_t stream) const;
  void launch_projection(const std::uint8_t *input, const float *eigenvectors, float *output,
                         int begin, int components, size_t n_batch, cudaStream_t stream) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holotask::syncs::detail
