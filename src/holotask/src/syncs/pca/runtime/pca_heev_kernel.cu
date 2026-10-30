// Copyright 2025 Digital Holography Foundation
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

#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

using namespace cusolverdx;

using Base =
    decltype(Size<PCA_FEATURES>() + Precision<float>() + Type<type::real>() + Function<heev>() +
             FillMode<fill_mode::lower>() + Arrangement<arrangement::col_major>() +
             Job<job::overwrite_vectors>() + SM<PCA_SOLVER_SM>() + Block());
using BatchedSolver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());
using TailSolver    = decltype(Base() + BatchesPerBlock<1>());

extern "C" __constant__ unsigned int pca_batched_block_x       = BatchedSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_batched_shared_memory = BatchedSolver::shared_memory_size;
extern "C" __constant__ unsigned int pca_batches_per_block     = BatchedSolver::batches_per_block;
extern "C" __constant__ unsigned int pca_tail_block_x          = TailSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_tail_shared_memory    = TailSolver::shared_memory_size;

template <class Solver>
__device__ void solve(float *covariance, float *eigenvalues, int *info,
                      const unsigned int batches) {
  constexpr unsigned int m                 = Solver::m_size;
  constexpr unsigned int batches_per_block = Solver::batches_per_block;
  constexpr unsigned int lda_shared        = Solver::lda;
  constexpr unsigned int matrix_elements   = m * m;

  const unsigned int batch = blockIdx.x * batches_per_block;
  if (batch >= batches) {
    return;
  }

  extern __shared__ __align__(16) cusolverdx::byte shared_memory[];
  auto [matrix_shared, eigenvalues_shared, workspace_shared] =
      cusolverdx::shared_memory::slice<float, float, float>(
          shared_memory, alignof(float), lda_shared * m * batches_per_block, alignof(float),
          m * batches_per_block, alignof(float));

  float *matrix_global      = covariance + matrix_elements * batch;
  float *eigenvalues_global = eigenvalues + m * batch;

  cusolverdx::copy_2d<Solver, m, m, arrangement::col_major, batches_per_block>(
      matrix_global, m, matrix_shared, lda_shared);
  __syncthreads();

  Solver().execute(matrix_shared, lda_shared, eigenvalues_shared, workspace_shared, info + batch);

  cusolverdx::copy_2d<Solver, m, 1, arrangement::col_major, batches_per_block>(
      eigenvalues_shared, m, eigenvalues_global, m);
  __syncthreads();
  cusolverdx::copy_2d<Solver, m, m, arrangement::col_major, batches_per_block>(
      matrix_shared, lda_shared, matrix_global, m);
}

extern "C" __global__ void pca_heev_batched(float *covariance, float *eigenvalues, int *info,
                                            const unsigned int batches) {
  solve<BatchedSolver>(covariance, eigenvalues, info, batches);
}

extern "C" __global__ void pca_heev_tail(float *covariance, float *eigenvalues, int *info,
                                         const unsigned int batches) {
  solve<TailSolver>(covariance, eigenvalues, info, batches);
}
