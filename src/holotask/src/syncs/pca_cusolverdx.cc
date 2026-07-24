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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/nvrtc.hh"

#include "logger.hh"

namespace holotask::syncs::detail {

namespace {

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
      std::format("-DPCA_SOLVER_SM={}", solver_sm),
      std::format("--include-path={}", HOLOFLOW_CUDA_INCLUDE_DIR),
      std::format("--include-path={}/cccl", HOLOFLOW_CUDA_INCLUDE_DIR),
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
  Impl(int n_features, size_t n_batch, cudaStream_t stream)
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

    const auto lto_ir = compile_lto_ir(n_features, solver_sm, architecture);
    const auto cubin  = link_cubin(lto_ir, architecture);

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

PcaHeevKernel::PcaHeevKernel(int n_features, size_t n_batch, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(n_features, n_batch, stream)) {}

PcaHeevKernel::~PcaHeevKernel() noexcept = default;

PcaHeevKernel::PcaHeevKernel(PcaHeevKernel &&) noexcept = default;

PcaHeevKernel &PcaHeevKernel::operator=(PcaHeevKernel &&) noexcept = default;

bool PcaHeevKernel::is_compatible_stream(cudaStream_t stream) const {
  return impl_->is_compatible_stream(stream);
}

void PcaHeevKernel::launch(float *covariance, float *eigenvalues, int *info, size_t n_batch,
                           cudaStream_t stream) const {
  impl_->launch(covariance, eigenvalues, info, n_batch, stream);
}

} // namespace holotask::syncs::detail
