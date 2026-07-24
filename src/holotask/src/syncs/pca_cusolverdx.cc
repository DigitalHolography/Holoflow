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

#include "pca_cusolverdx.hh"

#include <cuda.h>
#include <nvJitLink.h>

#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/nvrtc.hh"

#include "logger.hh"

namespace holotask::syncs::detail {

namespace {

constexpr unsigned int kProjectionTileComponents = 8;

constexpr const char *pca_heev_source = R"mathdx(
#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

constexpr unsigned int covariance_tile_samples = 64;
constexpr unsigned int covariance_split_samples = 4096;
constexpr unsigned int projection_tile_samples = 128;
constexpr unsigned int projection_tile_components = 8;
constexpr unsigned int mathdx_block_size = 256;

using CovarianceBlas =
    decltype(cublasdx::Size<PCA_FEATURES, PCA_FEATURES, covariance_tile_samples>() +
             cublasdx::Precision<__half, __half, float>() +
             cublasdx::Type<cublasdx::type::real>() +
             cublasdx::Function<cublasdx::function::MM>() +
             cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major,
                                   cublasdx::col_major>() +
             cublasdx::Alignment<2, 2, 4>() + cublasdx::Block() +
             cublasdx::BlockDim<mathdx_block_size>() + cublasdx::SM<PCA_BLAS_SM>());

template <unsigned int Components>
using ProjectionBlas =
    decltype(cublasdx::Size<projection_tile_samples, Components, PCA_FEATURES>() +
             cublasdx::Precision<__half, __half, float>() +
             cublasdx::Type<cublasdx::type::real>() +
             cublasdx::Function<cublasdx::function::MM>() +
             cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major,
                                   cublasdx::col_major>() +
             cublasdx::Alignment<1, 4, 4>() + cublasdx::Block() +
             cublasdx::BlockDim<mathdx_block_size>() + cublasdx::SM<PCA_BLAS_SM>());

using ProjectionBlas8 = ProjectionBlas<projection_tile_components>;
using ProjectionBlas1 = ProjectionBlas<1>;

using Base = decltype(cusolverdx::Size<PCA_FEATURES>() + cusolverdx::Precision<float>() +
                      cusolverdx::Type<cusolverdx::type::real>() +
                      cusolverdx::Function<cusolverdx::heev>() +
                      cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
                      cusolverdx::Arrangement<cusolverdx::arrangement::col_major>() +
                      cusolverdx::Job<cusolverdx::job::overwrite_vectors>() +
                      cusolverdx::SM<PCA_SOLVER_SM>() + cusolverdx::Block());
using BatchedSolver =
    decltype(Base() + cusolverdx::BatchesPerBlock<Base::suggested_batches_per_block>());
using TailSolver = decltype(Base() + cusolverdx::BatchesPerBlock<1>());

extern "C" __constant__ unsigned int pca_batched_block_x = BatchedSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_batched_shared_memory =
    BatchedSolver::shared_memory_size;
extern "C" __constant__ unsigned int pca_batches_per_block =
    BatchedSolver::batches_per_block;
extern "C" __constant__ unsigned int pca_tail_block_x = TailSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_tail_shared_memory = TailSolver::shared_memory_size;
extern "C" __constant__ unsigned int pca_covariance_block_x = CovarianceBlas::block_dim.x;
extern "C" __constant__ unsigned int pca_covariance_shared_memory =
    cublasdx::get_shared_storage_size_ab<CovarianceBlas, __half, __half>(
        CovarianceBlas::suggest_layout_smem_a(), CovarianceBlas::suggest_layout_smem_b());
extern "C" __constant__ unsigned int pca_covariance_split_samples =
    covariance_split_samples;
extern "C" __constant__ unsigned int pca_projection8_block_x = ProjectionBlas8::block_dim.x;
extern "C" __constant__ unsigned int pca_projection8_shared_memory =
    cublasdx::get_shared_storage_size_ab<ProjectionBlas8, unsigned char, float>();
extern "C" __constant__ unsigned int pca_projection1_block_x = ProjectionBlas1::block_dim.x;
extern "C" __constant__ unsigned int pca_projection1_shared_memory =
    cublasdx::get_shared_storage_size_ab<ProjectionBlas1, unsigned char, float>();
extern "C" __constant__ unsigned int pca_projection_tile_samples =
    projection_tile_samples;

extern "C" __global__ void pca_covariance_u8(const unsigned char *input, float *partials,
                                               const unsigned int samples,
                                               const unsigned int partial_count,
                                               const unsigned int batches) {
  CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(CovarianceBlas);
  const unsigned int partial = blockIdx.x;
  const unsigned int batch = blockIdx.y;
  if (partial >= partial_count || batch >= batches) {
    return;
  }

  extern __shared__ __align__(16) cublasdx::byte covariance_shared_memory[];
  auto [a_ptr, b_ptr] =
      cublasdx::slice_shared_memory_ab<CovarianceBlas, __half, __half>(
          covariance_shared_memory, CovarianceBlas::suggest_layout_smem_a(),
          CovarianceBlas::suggest_layout_smem_b());
  auto a_shared = cublasdx::make_tensor(a_ptr, CovarianceBlas::suggest_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(b_ptr, CovarianceBlas::suggest_layout_smem_b());
  auto accumulator = CovarianceBlas().suggest_accumulator();
  accumulator.clear();

  const unsigned char *batch_input =
      input + static_cast<unsigned long long>(batch) * PCA_FEATURES * samples;
  const unsigned int split_begin = partial * covariance_split_samples;
  const unsigned int split_end =
      min(split_begin + covariance_split_samples, samples);

  for (unsigned int stage_begin = split_begin; stage_begin < split_end;
       stage_begin += covariance_tile_samples) {
    for (unsigned int linear = threadIdx.x;
         linear < PCA_FEATURES * covariance_tile_samples; linear += blockDim.x) {
      const unsigned int feature = linear / covariance_tile_samples;
      const unsigned int sample_in_tile = linear % covariance_tile_samples;
      const unsigned int sample = stage_begin + sample_in_tile;
      const unsigned char value =
          sample < split_end ? batch_input[feature * samples + sample]
                             : static_cast<unsigned char>(0);
      const __half converted = __float2half_rn(static_cast<float>(value));
      a_shared(feature, sample_in_tile) = converted;
      b_shared(sample_in_tile, feature) = converted;
    }
    __syncthreads();
    CovarianceBlas().execute(a_shared, b_shared, accumulator);
    __syncthreads();
  }

  float *partial_output =
      partials +
      (static_cast<unsigned long long>(batch) * partial_count + partial) *
          PCA_FEATURES * PCA_FEATURES;
  auto output_tensor =
      cublasdx::make_tensor(partial_output, CovarianceBlas::get_layout_gmem_c());
  accumulator.partition_and_store(output_tensor);
}

extern "C" __global__ void pca_reduce_covariance(const float *partials, float *covariance,
                                                   const unsigned int partial_count,
                                                   const unsigned int batches) {
  const unsigned int batch = blockIdx.x;
  if (batch >= batches) {
    return;
  }

  constexpr unsigned int matrix_elements = PCA_FEATURES * PCA_FEATURES;
  for (unsigned int element = threadIdx.x; element < matrix_elements; element += blockDim.x) {
    float sum = 0.0f;
    const float *batch_partials =
        partials + static_cast<unsigned long long>(batch) * partial_count * matrix_elements;
    for (unsigned int partial = 0; partial < partial_count; ++partial) {
      sum += batch_partials[partial * matrix_elements + element];
    }
    covariance[static_cast<unsigned long long>(batch) * matrix_elements + element] = sum;
  }
}

template <class Blas, unsigned int TileComponents>
__device__ void project_u8(const unsigned char *input, const float *eigenvectors, float *output,
                           const unsigned int samples, const unsigned int begin,
                           const unsigned int components, const unsigned int component_base,
                           const unsigned int batches) {
  CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(Blas);
  const unsigned int sample_tile = blockIdx.x;
  const unsigned int component_tile = blockIdx.y;
  const unsigned int batch = blockIdx.z;
  if (batch >= batches) {
    return;
  }

  const unsigned int first_sample = sample_tile * projection_tile_samples;
  const unsigned int first_component =
      component_base + component_tile * TileComponents;

  extern __shared__ __align__(16) cublasdx::byte projection_shared_memory[];
  auto [a_ptr, b_ptr] =
      cublasdx::slice_shared_memory_ab<Blas, unsigned char, float>(projection_shared_memory);
  auto a_shared = cublasdx::make_tensor(a_ptr, Blas::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(b_ptr, Blas::get_layout_smem_b());

  const unsigned char *batch_input =
      input + static_cast<unsigned long long>(batch) * PCA_FEATURES * samples;
  const float *batch_eigenvectors =
      eigenvectors + static_cast<unsigned long long>(batch) * PCA_FEATURES * PCA_FEATURES;
  float *batch_output =
      output + static_cast<unsigned long long>(batch) * components * samples;

  for (unsigned int linear = threadIdx.x;
       linear < projection_tile_samples * PCA_FEATURES; linear += blockDim.x) {
    const unsigned int sample_in_tile = linear % projection_tile_samples;
    const unsigned int feature = linear / projection_tile_samples;
    a_shared(sample_in_tile, feature) =
        batch_input[feature * samples + first_sample + sample_in_tile];
  }
  for (unsigned int linear = threadIdx.x;
       linear < PCA_FEATURES * TileComponents; linear += blockDim.x) {
    const unsigned int feature = linear % PCA_FEATURES;
    const unsigned int component_in_tile = linear / PCA_FEATURES;
    const unsigned int component = first_component + component_in_tile;
    b_shared(feature, component_in_tile) =
        batch_eigenvectors[(begin + component) * PCA_FEATURES + feature];
  }
  __syncthreads();

  auto accumulator = Blas().execute(a_shared, b_shared);
  auto global_output =
      cublasdx::make_gmem_tensor<cublasdx::col_major>(batch_output, samples, components, samples);
  auto output_tile =
      cublasdx::get_tile(global_output, Blas::c_shape, sample_tile,
                         first_component / TileComponents);
  accumulator.partition_and_store(output_tile);
}

extern "C" __global__ void pca_projection_u8x8(const unsigned char *input,
                                                const float *eigenvectors, float *output,
                                                const unsigned int samples,
                                                const unsigned int begin,
                                                const unsigned int components,
                                                const unsigned int batches) {
  project_u8<ProjectionBlas8, projection_tile_components>(
      input, eigenvectors, output, samples, begin, components, 0, batches);
}

extern "C" __global__ void pca_projection_u8x1(const unsigned char *input,
                                                const float *eigenvectors, float *output,
                                                const unsigned int samples,
                                                const unsigned int begin,
                                                const unsigned int components,
                                                const unsigned int component_base,
                                                const unsigned int batches) {
  project_u8<ProjectionBlas1, 1>(input, eigenvectors, output, samples, begin, components,
                                component_base, batches);
}

extern "C" __global__ void pca_projection_u8_tail(const unsigned char *input,
                                                   const float *eigenvectors, float *output,
                                                   const unsigned int samples,
                                                   const unsigned int full_samples,
                                                   const unsigned int begin,
                                                   const unsigned int components,
                                                   const unsigned int batches) {
  const unsigned int tail_samples = samples - full_samples;
  const unsigned long long count =
      static_cast<unsigned long long>(batches) * components * tail_samples;
  const unsigned long long linear =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) {
    return;
  }

  const unsigned int sample_in_tail = linear % tail_samples;
  const unsigned long long component_linear = linear / tail_samples;
  const unsigned int component = component_linear % components;
  const unsigned int batch = component_linear / components;
  const unsigned int sample = full_samples + sample_in_tail;
  const unsigned char *batch_input =
      input + static_cast<unsigned long long>(batch) * PCA_FEATURES * samples;
  const float *batch_eigenvectors =
      eigenvectors + static_cast<unsigned long long>(batch) * PCA_FEATURES * PCA_FEATURES;

  float sum = 0.0f;
  for (unsigned int feature = 0; feature < PCA_FEATURES; ++feature) {
    sum += static_cast<float>(batch_input[feature * samples + sample]) *
           batch_eigenvectors[(begin + component) * PCA_FEATURES + feature];
  }
  output[(static_cast<unsigned long long>(batch) * components + component) * samples + sample] =
      sum;
}

template <class Solver>
__device__ void solve(float *covariance, float *eigenvalues, int *info,
                      const unsigned int batches) {
  constexpr unsigned int m = Solver::m_size;
  constexpr unsigned int batches_per_block = Solver::batches_per_block;
  constexpr unsigned int lda_shared = Solver::lda;
  constexpr unsigned int matrix_elements = m * m;

  const unsigned int batch = blockIdx.x * batches_per_block;
  if (batch >= batches) {
    return;
  }

  extern __shared__ __align__(16) cusolverdx::byte solver_shared_memory[];
  auto [matrix_shared, eigenvalues_shared, workspace_shared] =
      cusolverdx::shared_memory::slice<float, float, float>(
          solver_shared_memory, alignof(float), lda_shared * m * batches_per_block, alignof(float),
          m * batches_per_block, alignof(float));

  float *matrix_global = covariance + matrix_elements * batch;
  float *eigenvalues_global = eigenvalues + m * batch;

  cusolverdx::copy_2d<Solver, m, m, cusolverdx::arrangement::col_major, batches_per_block>(
      matrix_global, m, matrix_shared, lda_shared);
  __syncthreads();

  Solver().execute(matrix_shared, lda_shared, eigenvalues_shared, workspace_shared, info + batch);

  cusolverdx::copy_2d<Solver, m, 1, cusolverdx::arrangement::col_major, batches_per_block>(
      eigenvalues_shared, m, eigenvalues_global, m);
  __syncthreads();
  cusolverdx::copy_2d<Solver, m, m, cusolverdx::arrangement::col_major, batches_per_block>(
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
)mathdx";

std::string driver_error_message(CUresult result, const char *expression, const char *file,
                                 int line) {
  const char *name    = nullptr;
  const char *message = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &message);
  return std::format("CUDA Driver error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     message != nullptr ? message : "unknown",
                     name != nullptr ? name : std::to_string(static_cast<int>(result)), expression,
                     file, line);
}

void driver_check(CUresult result, const char *expression, const char *file, int line) {
  if (result != CUDA_SUCCESS) {
    const auto message = driver_error_message(result, expression, file, line);
    logger()->error("{}", message);
    throw std::runtime_error(message);
  }
}

#define PCA_DRIVER_CHECK(expr) driver_check((expr), #expr, __FILE__, __LINE__)

std::string jitlink_error_log(nvJitLinkHandle linker) {
  size_t size = 0;
  if (linker == nullptr || nvJitLinkGetErrorLogSize(linker, &size) != NVJITLINK_SUCCESS ||
      size == 0) {
    return {};
  }

  std::vector<char> log(size);
  if (nvJitLinkGetErrorLog(linker, log.data()) != NVJITLINK_SUCCESS) {
    return {};
  }
  return std::string(log.data());
}

void jitlink_check(nvJitLinkHandle linker, nvJitLinkResult result, const char *expression,
                   const char *file, int line) {
  if (result != NVJITLINK_SUCCESS) {
    const auto log     = jitlink_error_log(linker);
    const auto message = std::format("nvJitLink error: {} at {}:{}\n  expression : {}\n{}",
                                     static_cast<int>(result), file, line, expression, log);
    logger()->error("{}", message);
    throw std::runtime_error(message);
  }
}

#define PCA_NVJITLINK_CHECK(linker, expr) jitlink_check((linker), (expr), #expr, __FILE__, __LINE__)

class ContextGuard {
public:
  explicit ContextGuard(CUcontext context) {
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&previous_));
    if (previous_ != context) {
      PCA_DRIVER_CHECK(cuCtxPushCurrent(context));
      pushed_ = true;
    }
  }

  ~ContextGuard() noexcept {
    if (pushed_) {
      CUcontext  popped = nullptr;
      const auto result = cuCtxPopCurrent(&popped);
      if (result != CUDA_SUCCESS) {
        logger()->critical("{}",
                           driver_error_message(result, "cuCtxPopCurrent", __FILE__, __LINE__));
        std::abort();
      }
    }
  }

  ContextGuard(const ContextGuard &)            = delete;
  ContextGuard &operator=(const ContextGuard &) = delete;

private:
  CUcontext previous_{nullptr};
  bool      pushed_{false};
};

template <typename T> T module_constant(CUmodule module, const char *name) {
  CUdeviceptr address = 0;
  size_t      size    = 0;
  PCA_DRIVER_CHECK(cuModuleGetGlobal(&address, &size, module, name));
  if (size != sizeof(T)) {
    throw std::runtime_error(
        std::format("Unexpected size for cuSolverDx module constant {}", name));
  }

  T value{};
  PCA_DRIVER_CHECK(cuMemcpyDtoH(&value, address, sizeof(T)));
  return value;
}

CUcontext stream_context(cudaStream_t stream) {
  CUcontext context = nullptr;
  if (stream != nullptr) {
    PCA_DRIVER_CHECK(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context));
  } else {
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&context));
  }
  if (context == nullptr) {
    throw std::runtime_error("PCA cuSolverDx compilation requires an active CUDA context");
  }
  return context;
}

std::vector<char> compile_lto_ir(int n_features, int solver_sm, int architecture) {
  curaii::NvrtcProgram program(pca_heev_source, "pca_heev_kernel.cu");

  std::vector<std::string> options = {
      "--std=c++17",
      "--device-as-default-execution-space",
      "-dlto",
      "--relocatable-device-code=true",
      std::format("--gpu-architecture=sm_{}", architecture),
      std::format("-DPCA_FEATURES={}", n_features),
      std::format("-DPCA_BLAS_SM={}", solver_sm),
      std::format("-DPCA_SOLVER_SM={}", solver_sm),
      std::format("--include-path={}", HOLOFLOW_CUDA_INCLUDE_DIR),
      std::format("--include-path={}/cccl", HOLOFLOW_CUDA_INCLUDE_DIR),
      std::format("--include-path={}", HOLOFLOW_CUBLASDX_INCLUDE_DIR),
      std::format("--include-path={}", HOLOFLOW_CUBLASDX_CUTLASS_INCLUDE_DIR),
      std::format("--include-path={}", HOLOFLOW_CUSOLVERDX_INCLUDE_DIR),
      std::format("--include-path={}", HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR),
  };

  std::vector<const char *> option_ptrs;
  option_ptrs.reserve(options.size());
  for (const auto &option : options) {
    option_ptrs.push_back(option.c_str());
  }

  const auto compile_result =
      nvrtcCompileProgram(program.get(), static_cast<int>(option_ptrs.size()), option_ptrs.data());
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program.get(), &log_size));
    std::vector<char> log(log_size);
    NVRTC_CHECK(nvrtcGetProgramLog(program.get(), log.data()));
    logger()->error("[Pca] NVRTC compilation log:\n{}", log.data());
    NVRTC_CHECK(compile_result);
  }

  size_t lto_size = 0;
  NVRTC_CHECK(nvrtcGetLTOIRSize(program.get(), &lto_size));
  std::vector<char> lto_ir(lto_size);
  NVRTC_CHECK(nvrtcGetLTOIR(program.get(), lto_ir.data()));
  return lto_ir;
}

std::vector<char> link_cubin(const std::vector<char> &lto_ir, int architecture) {
  nvJitLinkHandle linker      = nullptr;
  const auto      arch_option = std::format("-arch=sm_{}", architecture);
  const char     *options[]   = {"-lto", arch_option.c_str()};

  PCA_NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker, 2, options));
  try {
    PCA_NVJITLINK_CHECK(linker,
                        nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, HOLOFLOW_CUBLASDX_FATBIN));
    PCA_NVJITLINK_CHECK(
        linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, HOLOFLOW_CUSOLVERDX_FATBIN));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR,
                                                 const_cast<char *>(lto_ir.data()), lto_ir.size(),
                                                 "pca_heev_lto_ir"));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));

    size_t cubin_size = 0;
    PCA_NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    PCA_NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkDestroy(&linker));
    return cubin;
  } catch (...) {
    if (linker != nullptr) {
      (void)nvJitLinkDestroy(&linker);
    }
    throw;
  }
}

} // namespace

struct PcaHeevKernel::Impl {
  Impl(int n_features, int n_samples, size_t n_batch, cudaStream_t stream)
      : n_features(n_features), n_samples(n_samples), n_batch(n_batch) {
    CUDA_CHECK(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    architecture        = major * 10 + minor;
    const int solver_sm = architecture * 10;

    PCA_DRIVER_CHECK(cuInit(0));
    context = stream_context(stream);
    ContextGuard guard(context);

    const auto lto_ir = compile_lto_ir(n_features, solver_sm, architecture);
    const auto cubin  = link_cubin(lto_ir, architecture);

    PCA_DRIVER_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr));
    try {
      PCA_DRIVER_CHECK(cuModuleGetFunction(&batched_function, module, "pca_heev_batched"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&tail_function, module, "pca_heev_tail"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&covariance_function, module, "pca_covariance_u8"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&reduction_function, module, "pca_reduce_covariance"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&projection8_function, module, "pca_projection_u8x8"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&projection1_function, module, "pca_projection_u8x1"));
      PCA_DRIVER_CHECK(
          cuModuleGetFunction(&projection_tail_function, module, "pca_projection_u8_tail"));

      batched_block_x       = module_constant<unsigned int>(module, "pca_batched_block_x");
      batched_shared_memory = module_constant<unsigned int>(module, "pca_batched_shared_memory");
      batches_per_block     = module_constant<unsigned int>(module, "pca_batches_per_block");
      tail_block_x          = module_constant<unsigned int>(module, "pca_tail_block_x");
      tail_shared_memory    = module_constant<unsigned int>(module, "pca_tail_shared_memory");
      covariance_block_x    = module_constant<unsigned int>(module, "pca_covariance_block_x");
      covariance_shared_memory =
          module_constant<unsigned int>(module, "pca_covariance_shared_memory");
      covariance_split_samples =
          module_constant<unsigned int>(module, "pca_covariance_split_samples");
      projection8_block_x = module_constant<unsigned int>(module, "pca_projection8_block_x");
      projection8_shared_memory =
          module_constant<unsigned int>(module, "pca_projection8_shared_memory");
      projection1_block_x = module_constant<unsigned int>(module, "pca_projection1_block_x");
      projection1_shared_memory =
          module_constant<unsigned int>(module, "pca_projection1_shared_memory");
      projection_tile_samples =
          module_constant<unsigned int>(module, "pca_projection_tile_samples");
      partial_count = (static_cast<size_t>(n_samples) + covariance_split_samples - 1) /
                      covariance_split_samples;

      int max_shared_memory = 0;
      CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                        device));

      const bool uses_batched = n_batch >= batches_per_block;
      const bool uses_tail    = n_batch < batches_per_block || (n_batch % batches_per_block) != 0;
      if ((uses_batched && batched_shared_memory > static_cast<unsigned int>(max_shared_memory)) ||
          (uses_tail && tail_shared_memory > static_cast<unsigned int>(max_shared_memory))) {
        const auto required = uses_batched ? batched_shared_memory : tail_shared_memory;
        throw std::runtime_error(
            std::format("[Pca] cuSolverDx HEEV for {} features requires {} bytes of dynamic "
                        "shared memory, but device sm_{} provides {} bytes",
                        n_features, required, architecture, max_shared_memory));
      }

      if (uses_batched) {
        PCA_DRIVER_CHECK(cuFuncSetAttribute(batched_function,
                                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                            static_cast<int>(batched_shared_memory)));
      }
      if (uses_tail) {
        PCA_DRIVER_CHECK(cuFuncSetAttribute(tail_function,
                                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                            static_cast<int>(tail_shared_memory)));
      }
      const auto set_shared_memory = [&](CUfunction function, unsigned int bytes,
                                         const char *kernel_name) {
        if (bytes > static_cast<unsigned int>(max_shared_memory)) {
          throw std::runtime_error(std::format(
              "[Pca] {} requires {} bytes of dynamic shared memory, but device sm_{} provides {} "
              "bytes",
              kernel_name, bytes, architecture, max_shared_memory));
        }
        PCA_DRIVER_CHECK(cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(bytes)));
      };
      set_shared_memory(covariance_function, covariance_shared_memory, "cuBLASDx covariance");
      set_shared_memory(projection8_function, projection8_shared_memory, "cuBLASDx projection x8");
      set_shared_memory(projection1_function, projection1_shared_memory, "cuBLASDx projection x1");
    } catch (...) {
      (void)cuModuleUnload(module);
      module = nullptr;
      throw;
    }
  }

  ~Impl() noexcept {
    if (module != nullptr) {
      try {
        ContextGuard guard(context);
        const auto   result = cuModuleUnload(module);
        if (result != CUDA_SUCCESS) {
          logger()->critical("{}",
                             driver_error_message(result, "cuModuleUnload", __FILE__, __LINE__));
          std::abort();
        }
      } catch (...) {
        std::abort();
      }
    }
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const {
    return stream_context(stream) == context;
  }

  void validate_launch(size_t batches, cudaStream_t stream) const {
    if (batches != n_batch) {
      throw std::invalid_argument("[Pca] MathDx batch count changed after task creation");
    }
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument("[Pca] MathDx kernels and PCA stream use different contexts");
    }
    if (batches > std::numeric_limits<unsigned int>::max()) {
      throw std::invalid_argument("[Pca] MathDx batch count exceeds CUDA grid limits");
    }
  }

  void launch_covariance(const std::uint8_t *input, float *partials, size_t batches,
                         cudaStream_t stream) const {
    validate_launch(batches, stream);
    CUdeviceptr input_arg          = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr partials_arg       = reinterpret_cast<CUdeviceptr>(partials);
    auto        samples_arg        = static_cast<unsigned int>(n_samples);
    auto        partials_count_arg = static_cast<unsigned int>(partial_count);
    auto        batches_arg        = static_cast<unsigned int>(batches);
    void       *arguments[]        = {&input_arg, &partials_arg, &samples_arg, &partials_count_arg,
                                      &batches_arg};
    PCA_DRIVER_CHECK(cuLaunchKernel(covariance_function, partials_count_arg, batches_arg, 1,
                                    covariance_block_x, 1, 1, covariance_shared_memory,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  void launch_covariance_reduction(const float *partials, float *covariance, size_t batches,
                                   cudaStream_t stream) const {
    validate_launch(batches, stream);
    CUdeviceptr partials_arg       = reinterpret_cast<CUdeviceptr>(partials);
    CUdeviceptr covariance_arg     = reinterpret_cast<CUdeviceptr>(covariance);
    auto        partials_count_arg = static_cast<unsigned int>(partial_count);
    auto        batches_arg        = static_cast<unsigned int>(batches);
    void       *arguments[] = {&partials_arg, &covariance_arg, &partials_count_arg, &batches_arg};
    PCA_DRIVER_CHECK(cuLaunchKernel(reduction_function, batches_arg, 1, 1, 256, 1, 1, 0,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  void launch(float *covariance, float *eigenvalues, int *info, size_t batches,
              cudaStream_t stream) const {
    validate_launch(batches, stream);

    const size_t full_batches = (batches / batches_per_block) * batches_per_block;
    if (full_batches != 0) {
      CUdeviceptr covariance_arg  = reinterpret_cast<CUdeviceptr>(covariance);
      CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues);
      CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info);
      auto        batch_arg       = static_cast<unsigned int>(full_batches);
      void       *arguments[]     = {&covariance_arg, &eigenvalues_arg, &info_arg, &batch_arg};

      PCA_DRIVER_CHECK(cuLaunchKernel(batched_function,
                                      static_cast<unsigned int>(full_batches / batches_per_block),
                                      1, 1, batched_block_x, 1, 1, batched_shared_memory,
                                      reinterpret_cast<CUstream>(stream), arguments, nullptr));
    }

    const size_t tail_batches = batches - full_batches;
    if (tail_batches != 0) {
      const size_t matrix_offset = full_batches * static_cast<size_t>(n_features) * n_features;
      const size_t value_offset  = full_batches * static_cast<size_t>(n_features);

      CUdeviceptr covariance_arg  = reinterpret_cast<CUdeviceptr>(covariance + matrix_offset);
      CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues + value_offset);
      CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info + full_batches);
      auto        batch_arg       = static_cast<unsigned int>(tail_batches);
      void       *arguments[]     = {&covariance_arg, &eigenvalues_arg, &info_arg, &batch_arg};

      PCA_DRIVER_CHECK(cuLaunchKernel(tail_function, static_cast<unsigned int>(tail_batches), 1, 1,
                                      tail_block_x, 1, 1, tail_shared_memory,
                                      reinterpret_cast<CUstream>(stream), arguments, nullptr));
    }
  }

  void launch_projection(const std::uint8_t *input, const float *eigenvectors, float *output,
                         int begin, int components, size_t batches, cudaStream_t stream) const {
    validate_launch(batches, stream);
    if (begin < 0 || components <= 0 || begin + components > n_features) {
      throw std::invalid_argument("[Pca] invalid projection component range");
    }

    CUdeviceptr input_arg         = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr eigenvectors_arg  = reinterpret_cast<CUdeviceptr>(eigenvectors);
    CUdeviceptr output_arg        = reinterpret_cast<CUdeviceptr>(output);
    auto        samples_arg       = static_cast<unsigned int>(n_samples);
    auto        begin_arg         = static_cast<unsigned int>(begin);
    auto        components_arg    = static_cast<unsigned int>(components);
    auto        batches_arg       = static_cast<unsigned int>(batches);
    const auto  full_sample_tiles = static_cast<unsigned int>(n_samples) / projection_tile_samples;
    const auto  full_samples      = full_sample_tiles * projection_tile_samples;
    const auto  full_component_groups =
        static_cast<unsigned int>(components) / kProjectionTileComponents;
    const auto full_components = full_component_groups * kProjectionTileComponents;

    if (full_sample_tiles != 0 && full_component_groups != 0) {
      void *arguments[] = {&input_arg, &eigenvectors_arg, &output_arg, &samples_arg,
                           &begin_arg, &components_arg,   &batches_arg};
      PCA_DRIVER_CHECK(cuLaunchKernel(projection8_function, full_sample_tiles,
                                      full_component_groups, batches_arg, projection8_block_x, 1, 1,
                                      projection8_shared_memory, reinterpret_cast<CUstream>(stream),
                                      arguments, nullptr));
    }

    if (full_sample_tiles != 0 && full_components < static_cast<unsigned int>(components)) {
      auto  component_base_arg = full_components;
      auto  remainder          = static_cast<unsigned int>(components) - full_components;
      void *arguments[]        = {&input_arg, &eigenvectors_arg, &output_arg,         &samples_arg,
                                  &begin_arg, &components_arg,   &component_base_arg, &batches_arg};
      PCA_DRIVER_CHECK(cuLaunchKernel(
          projection1_function, full_sample_tiles, remainder, batches_arg, projection1_block_x, 1,
          1, projection1_shared_memory, reinterpret_cast<CUstream>(stream), arguments, nullptr));
    }

    if (full_samples < static_cast<unsigned int>(n_samples)) {
      auto       full_samples_arg = full_samples;
      const auto tail_samples =
          static_cast<unsigned long long>(n_samples - static_cast<int>(full_samples));
      const auto count = static_cast<unsigned long long>(batches) *
                         static_cast<unsigned int>(components) * tail_samples;
      const auto blocks      = static_cast<unsigned int>((count + 255) / 256);
      void      *arguments[] = {&input_arg,        &eigenvectors_arg, &output_arg,     &samples_arg,
                                &full_samples_arg, &begin_arg,        &components_arg, &batches_arg};
      PCA_DRIVER_CHECK(cuLaunchKernel(projection_tail_function, blocks, 1, 1, 256, 1, 1, 0,
                                      reinterpret_cast<CUstream>(stream), arguments, nullptr));
    }
  }

  int          n_features;
  int          n_samples;
  size_t       n_batch;
  size_t       partial_count{0};
  int          device{0};
  int          architecture{0};
  CUcontext    context{nullptr};
  CUmodule     module{nullptr};
  CUfunction   batched_function{nullptr};
  CUfunction   tail_function{nullptr};
  CUfunction   covariance_function{nullptr};
  CUfunction   reduction_function{nullptr};
  CUfunction   projection8_function{nullptr};
  CUfunction   projection1_function{nullptr};
  CUfunction   projection_tail_function{nullptr};
  unsigned int batched_block_x{0};
  unsigned int batched_shared_memory{0};
  unsigned int batches_per_block{1};
  unsigned int tail_block_x{0};
  unsigned int tail_shared_memory{0};
  unsigned int covariance_block_x{0};
  unsigned int covariance_shared_memory{0};
  unsigned int covariance_split_samples{0};
  unsigned int projection8_block_x{0};
  unsigned int projection8_shared_memory{0};
  unsigned int projection1_block_x{0};
  unsigned int projection1_shared_memory{0};
  unsigned int projection_tile_samples{0};
};

PcaHeevKernel::PcaHeevKernel(int n_features, int n_samples, size_t n_batch, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(n_features, n_samples, n_batch, stream)) {}

PcaHeevKernel::~PcaHeevKernel() noexcept = default;

PcaHeevKernel::PcaHeevKernel(PcaHeevKernel &&) noexcept = default;

PcaHeevKernel &PcaHeevKernel::operator=(PcaHeevKernel &&) noexcept = default;

bool PcaHeevKernel::is_compatible_stream(cudaStream_t stream) const {
  return impl_->is_compatible_stream(stream);
}

size_t PcaHeevKernel::covariance_partial_count() const { return impl_->partial_count; }

void PcaHeevKernel::launch_covariance(const std::uint8_t *input, float *partials, size_t n_batch,
                                      cudaStream_t stream) const {
  impl_->launch_covariance(input, partials, n_batch, stream);
}

void PcaHeevKernel::launch_covariance_reduction(const float *partials, float *covariance,
                                                size_t n_batch, cudaStream_t stream) const {
  impl_->launch_covariance_reduction(partials, covariance, n_batch, stream);
}

void PcaHeevKernel::launch(float *covariance, float *eigenvalues, int *info, size_t n_batch,
                           cudaStream_t stream) const {
  impl_->launch(covariance, eigenvalues, info, n_batch, stream);
}

void PcaHeevKernel::launch_projection(const std::uint8_t *input, const float *eigenvectors,
                                      float *output, int begin, int components, size_t n_batch,
                                      cudaStream_t stream) const {
  impl_->launch_projection(input, eigenvectors, output, begin, components, n_batch, stream);
}

} // namespace holotask::syncs::detail
