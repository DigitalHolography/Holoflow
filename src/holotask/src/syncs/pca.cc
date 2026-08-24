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

#include "holotask/syncs/pca.hh"

#include <cuda.h>
#include <nvJitLink.h>
#include <nvtx3/nvtx3.hpp>
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "curaii/nvrtc.hh"

#include "logger.hh"

namespace holotask::syncs {

namespace {

// -------------------------------------------------------------------------------------------------
// cuSolverDx device program
// -------------------------------------------------------------------------------------------------


constexpr const char *pca_heev_source = R"cusolverdx(
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

using namespace cusolverdx;

using Base = decltype(Size<PCA_FEATURES>() + Precision<float>() + Type<type::real>() +
                      Function<heev>() + FillMode<fill_mode::lower>() +
                      Arrangement<arrangement::col_major>() +
                      Job<job::overwrite_vectors>() + SM<PCA_SOLVER_SM>() + Block());
using BatchedSolver =
    decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());
using TailSolver = decltype(Base() + BatchesPerBlock<1>());

extern "C" __constant__ unsigned int pca_batched_block_x = BatchedSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_batched_shared_memory =
    BatchedSolver::shared_memory_size;
extern "C" __constant__ unsigned int pca_batches_per_block =
    BatchedSolver::batches_per_block;
extern "C" __constant__ unsigned int pca_tail_block_x = TailSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_tail_shared_memory = TailSolver::shared_memory_size;

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

  extern __shared__ __align__(16) cusolverdx::byte shared_memory[];
  auto [matrix_shared, eigenvalues_shared, workspace_shared] =
      cusolverdx::shared_memory::slice<float, float, float>(
          shared_memory, alignof(float), lda_shared * m * batches_per_block, alignof(float),
          m * batches_per_block, alignof(float));

  float *matrix_global = covariance + matrix_elements * batch;
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
)cusolverdx";

// -------------------------------------------------------------------------------------------------
// CUDA driver support
// -------------------------------------------------------------------------------------------------

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

struct NvrtcAssets {
  std::filesystem::path cuda_include;
  std::filesystem::path cusolverdx_include;
  std::filesystem::path cutlass_include;
  std::filesystem::path cusolverdx_fatbin;
};

// -------------------------------------------------------------------------------------------------
// Runtime compilation
// -------------------------------------------------------------------------------------------------

std::filesystem::path executable_directory() {
  std::vector<wchar_t> buffer(MAX_PATH);
  while (true) {
    const auto length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      throw std::runtime_error(
          std::format("[Pca] Failed to locate the executable (Windows error {})", GetLastError()));
    }
    if (length < buffer.size() - 1) {
      return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    }
    buffer.resize(buffer.size() * 2);
  }
}

bool assets_exist(const NvrtcAssets &assets) {
  return std::filesystem::is_directory(assets.cuda_include) &&
         std::filesystem::is_regular_file(assets.cuda_include / "cuda_runtime.h") &&
         std::filesystem::is_directory(assets.cuda_include / "cccl") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx.hpp") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx_io.hpp") &&
         std::filesystem::is_directory(assets.cutlass_include) &&
         std::filesystem::is_regular_file(assets.cusolverdx_fatbin);
}

NvrtcAssets nvrtc_assets() {
  const auto installed_root =
      (executable_directory() / HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR).lexically_normal();
  NvrtcAssets installed{
      installed_root / "cuda/include",
      installed_root / "mathdx/include",
      installed_root / "mathdx/cutlass/include",
      installed_root / "mathdx/lib/libcusolverdx.fatbin",
  };
  if (assets_exist(installed)) {
    return installed;
  }

  NvrtcAssets build{
      HOLOFLOW_CUDA_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_FATBIN,
  };
  if (assets_exist(build)) {
    return build;
  }

  throw std::runtime_error(std::format(
      "[Pca] cuSolverDx runtime-compilation assets are missing. Expected installed assets under "
      "'{}' (CUDA headers, cuSolverDx/CommonDx headers, CUTLASS headers, and "
      "libcusolverdx.fatbin); build-tree fallback is also unavailable",
      installed_root.string()));
}

std::vector<char> compile_lto_ir(int n_features, int solver_sm, int architecture,
                                 const NvrtcAssets &assets) {
  curaii::NvrtcProgram program(pca_heev_source, "pca_heev_kernel.cu");

  std::vector<std::string> options = {
      "--std=c++17",
      "--device-as-default-execution-space",
      "-dlto",
      "--relocatable-device-code=true",
      std::format("--gpu-architecture=sm_{}", architecture),
      std::format("-DPCA_FEATURES={}", n_features),
      std::format("-DPCA_SOLVER_SM={}", solver_sm),
      std::format("--include-path={}", assets.cuda_include.string()),
      std::format("--include-path={}", (assets.cuda_include / "cccl").string()),
      std::format("--include-path={}", assets.cusolverdx_include.string()),
      std::format("--include-path={}", assets.cutlass_include.string()),
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

std::vector<char> link_cubin(const std::vector<char> &lto_ir, int architecture,
                             const NvrtcAssets &assets) {
  nvJitLinkHandle linker      = nullptr;
  const auto      arch_option = std::format("-arch=sm_{}", architecture);
  const char     *options[]   = {"-lto", arch_option.c_str()};

  PCA_NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker, 2, options));
  try {
    PCA_NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN,
                                                 assets.cusolverdx_fatbin.string().c_str()));
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


// -------------------------------------------------------------------------------------------------
// cuSolverDx eigensolver
// -------------------------------------------------------------------------------------------------

class CusolverDxEigensolver {
public:
  CusolverDxEigensolver(int n_features, size_t n_batch, cudaStream_t stream)
      : n_features(n_features), n_batch(n_batch) {
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

    const auto assets = nvrtc_assets();
    const auto lto_ir = compile_lto_ir(n_features, solver_sm, architecture, assets);
    const auto cubin  = link_cubin(lto_ir, architecture, assets);

    PCA_DRIVER_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr));
    try {
      PCA_DRIVER_CHECK(cuModuleGetFunction(&batched_function, module, "pca_heev_batched"));
      PCA_DRIVER_CHECK(cuModuleGetFunction(&tail_function, module, "pca_heev_tail"));

      batched_block_x       = module_constant<unsigned int>(module, "pca_batched_block_x");
      batched_shared_memory = module_constant<unsigned int>(module, "pca_batched_shared_memory");
      batches_per_block     = module_constant<unsigned int>(module, "pca_batches_per_block");
      tail_block_x          = module_constant<unsigned int>(module, "pca_tail_block_x");
      tail_shared_memory    = module_constant<unsigned int>(module, "pca_tail_shared_memory");

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
    } catch (...) {
      (void)cuModuleUnload(module);
      module = nullptr;
      throw;
    }
  }

  ~CusolverDxEigensolver() noexcept {
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

  void launch(float *covariance, float *eigenvalues, int *info, size_t batches,
              cudaStream_t stream) const {
    if (batches != n_batch) {
      throw std::invalid_argument("[Pca] cuSolverDx batch count changed after task creation");
    }
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument("[Pca] cuSolverDx kernel and PCA stream use different contexts");
    }

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

private:
  int          n_features;
  size_t       n_batch;
  int          device{0};
  int          architecture{0};
  CUcontext    context{nullptr};
  CUmodule     module{nullptr};
  CUfunction   batched_function{nullptr};
  CUfunction   tail_function{nullptr};
  unsigned int batched_block_x{0};
  unsigned int batched_shared_memory{0};
  unsigned int batches_per_block{1};
  unsigned int tail_block_x{0};
  unsigned int tail_shared_memory{0};
};


// -------------------------------------------------------------------------------------------------
// Eigensolver selection and cuSOLVER fallback
// -------------------------------------------------------------------------------------------------

constexpr int cusolver_fallback_min_features = 256;

class PcaEigensolver {
public:
  PcaEigensolver(int n_features, size_t n_batch, float *covariance, float *eigenvalues,
                 cudaStream_t stream)
      : n_features(n_features), n_batch(n_batch), context(stream_context(stream)) {
    if (n_features < cusolver_fallback_min_features) {
      try {
        cusolverdx = std::make_unique<CusolverDxEigensolver>(n_features, n_batch, stream);
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
    if (n_batch > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
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

private:
  int                                       n_features;
  size_t                                    n_batch;
  CUcontext                                 context{nullptr};
  std::unique_ptr<CusolverDxEigensolver>    cusolverdx;
  std::unique_ptr<curaii::CusolverDnHandle> cusolver_handle;
  std::unique_ptr<curaii::CusolverDnParams> cusolver_params;
  curaii::unique_device_ptr<std::byte>      device_workspace;
  size_t                                    device_workspace_size{0};
  std::vector<std::byte>                    host_workspace;
  size_t                                    host_workspace_size{0};
};


// -------------------------------------------------------------------------------------------------
// PCA task implementation
// -------------------------------------------------------------------------------------------------

class PcaTask : public holoflow::core::ISyncTask {
public:
  // n_features: size of the feature dimension (shape[-3])
  // n_batch:    product of all leading dimensions (1 for rank-3 input)
  PcaTask(const PcaSettings &settings, const holoflow::core::TDesc &idesc,
          const holoflow::core::SyncCreateCtx &ctx, int n_features, size_t n_batch)
      : settings_(settings), idesc_(idesc), stream_(ctx.stream), n_features_(n_features),
        n_batch_(n_batch) {

    const auto cov_elems = n_batch_ * static_cast<size_t>(n_features_) * n_features_;
    d_cov_               = curaii::make_unique_device_ptr<float>(cov_elems);
    d_eigvals_           = curaii::make_unique_device_ptr<float>(n_batch_ * n_features_);
    d_info_              = curaii::make_unique_device_ptr<int>(n_batch_);

    CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    eigensolver_ = std::make_unique<PcaEigensolver>(n_features_, n_batch_, d_cov_.get(),
                                                    d_eigvals_.get(), stream_);
  }

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const PcaSettings           &get_settings() const { return settings_; }
  bool                         is_compatible_stream(cudaStream_t stream) const {
    return eigensolver_->is_compatible_stream(stream);
  }

  void update_settings(const PcaSettings &settings, cudaStream_t stream) {
    settings_ = settings;
    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range r("PCA Sync Task");
    auto               &iview = ctx.inputs[0];
    auto               &oview = ctx.outputs[0];
    auto               *idata = reinterpret_cast<float *>(iview.data());
    auto               *odata = reinterpret_cast<float *>(oview.data());

    const auto     &idesc      = iview.desc;
    const int       rank       = static_cast<int>(idesc.shape.size());
    const int       height     = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 2)));
    const int       width      = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 1)));
    const int       n_samples  = height * width;
    const int       components = settings_.components();
    constexpr float alpha      = 1.0f;
    constexpr float beta       = 0.0f;

    // Strides (in elements) between consecutive batch slices
    const long long stride_I = static_cast<long long>(n_features_) * n_samples;
    const long long stride_C = static_cast<long long>(n_features_) * n_features_;
    const long long stride_O = static_cast<long long>(n_samples) * components;

    {
      nvtx3::scoped_range covariance_range("PCA covariance");
      // Independent GEMMs preserve cuBLAS Split-K selection for the large sample dimension.
      for (size_t b = 0; b < n_batch_; ++b) {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_T, CUBLAS_OP_N, n_features_,
                                  n_features_, n_samples, &alpha, idata + b * stride_I, CUDA_R_32F,
                                  n_samples, idata + b * stride_I, CUDA_R_32F, n_samples, &beta,
                                  d_cov_.get() + b * stride_C, CUDA_R_32F, n_features_,
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
      }
    }

    {
      nvtx3::scoped_range eigendecomposition_range("PCA eigendecomposition");
      eigensolver_->launch(d_cov_.get(), d_eigvals_.get(), d_info_.get(), n_batch_, stream_);
    }

    {
      nvtx3::scoped_range projection_range("PCA projection");
      const auto eigenvector_offset = static_cast<long long>(settings_.begin) * n_features_;
      for (size_t b = 0; b < n_batch_; ++b) {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples,
                                  components, n_features_, &alpha, idata + b * stride_I, CUDA_R_32F,
                                  n_samples, d_cov_.get() + b * stride_C + eigenvector_offset,
                                  CUDA_R_32F, n_features_, &beta, odata + b * stride_O, CUDA_R_32F,
                                  n_samples, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
      }
    }

    return holoflow::core::OpResult::Ok;
  }

private:
  PcaSettings           settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  int                   n_features_;
  size_t                n_batch_;

  curaii::CublasHandle               cublas_handle_;
  std::unique_ptr<PcaEigensolver>     eigensolver_;
  curaii::unique_device_ptr<float>    d_cov_;
  curaii::unique_device_ptr<float>    d_eigvals_;
  curaii::unique_device_ptr<int>      d_info_;
};


} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const PcaSettings &settings) {
  j = nlohmann::json{
      {"begin", settings.begin},
      {"end", settings.end},
  };
}

void from_json(const nlohmann::json &j, PcaSettings &settings) {
  j.at("begin").get_to(settings.begin);
  j.at("end").get_to(settings.end);
}

// -------------------------------------------------------------------------------------------------
// Factory methods
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult PcaFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[PcaFactory::infer] error: {}", msg);
      throw std::invalid_argument("PcaFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<PcaSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() >= 3, "expected input rank >= 3");
  check(idesc.dtype == holoflow::core::DType::F32,
        "cuSolverDx PCA currently supports F32 input only");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");

  const int feat_dim = static_cast<int>(idesc.rank()) - 3;
  check(settings.end <= static_cast<int>(idesc.shape.at(static_cast<size_t>(feat_dim))),
        "expected end <= n_features");

  auto oshape                              = idesc.shape;
  oshape.at(static_cast<size_t>(feat_dim)) = static_cast<size_t>(settings.components());
  holoflow::core::TDesc odesc(oshape, idesc.dtype, idesc.mem_loc);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {

  this->infer(input_descs, jsettings); // Validate bounds & types

  auto        settings = jsettings.get<PcaSettings>();
  const auto &idesc    = input_descs[0];
  const int   feat_dim = static_cast<int>(idesc.rank()) - 3;
  const int   n_feats  = static_cast<int>(idesc.shape.at(static_cast<size_t>(feat_dim)));

  // Flatten all leading dimensions into a single batch count
  size_t n_batch = 1;
  for (int i = 0; i < feat_dim; ++i) {
    n_batch *= idesc.shape[static_cast<size_t>(i)];
  }

  return std::make_unique<PcaTask>(settings, idesc, ctx, n_feats, n_batch);
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {

  this->infer(input_descs, jsettings); // Validate settings before updating the existing task

  if (input_descs.size() == 1) {
    const auto  new_settings = jsettings.get<PcaSettings>();
    const auto &new_idesc    = input_descs[0];

    if (new_idesc.dtype == holoflow::core::DType::F32) {
      auto *old_pca = dynamic_cast<PcaTask *>(old_task.get());
      if (old_pca != nullptr) {
        const auto &old_idesc = old_pca->get_idesc();
        if ((new_idesc.shape == old_idesc.shape) && (new_idesc.strides == old_idesc.strides) &&
            (new_idesc.dtype == old_idesc.dtype) && (new_idesc.mem_loc == old_idesc.mem_loc) &&
            old_pca->is_compatible_stream(ctx.stream)) {

          old_pca->update_settings(new_settings, ctx.stream);
          return old_task;
        }
      }
    }
  }

  // Fallback: Recreate the task entirely.
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
