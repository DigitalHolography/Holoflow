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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <list>
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
// PCA data model
// -------------------------------------------------------------------------------------------------

// PCA operates on tensors shaped [..., features, height, width]. All leading dimensions are
// flattened into independent batches.
struct PcaLayout {
  size_t feature_axis;
  int    features;
  int    height;
  int    width;
  int    samples;
  size_t batches;

  // Strides in elements between consecutive flattened batches.
  long long input_batch_stride;
  long long matrix_batch_stride;

  explicit PcaLayout(const holoflow::core::TDesc &desc) {
    const auto rank = desc.rank();

    feature_axis = rank - 3;
    features     = static_cast<int>(desc.shape.at(feature_axis));
    height       = static_cast<int>(desc.shape.at(rank - 2));
    width        = static_cast<int>(desc.shape.at(rank - 1));
    samples      = height * width;

    batches = 1;
    for (size_t axis = 0; axis < feature_axis; ++axis) {
      batches *= desc.shape.at(axis);
    }

    input_batch_stride  = static_cast<long long>(features) * samples;
    matrix_batch_stride = static_cast<long long>(features) * features;
  }

  [[nodiscard]] long long output_batch_stride(int components) const {
    return static_cast<long long>(samples) * components;
  }
};

struct PcaWorkspace {
  // The matrix buffer contains covariance matrices before HEEV and eigenvectors afterwards.
  curaii::unique_device_ptr<float> matrices;
  curaii::unique_device_ptr<float> eigenvalues;
  curaii::unique_device_ptr<int>   solver_info;

  explicit PcaWorkspace(const PcaLayout &layout)
      : matrices(curaii::make_unique_device_ptr<float>(
            layout.batches * static_cast<size_t>(layout.features) * layout.features)),
        eigenvalues(curaii::make_unique_device_ptr<float>(layout.batches * layout.features)),
        solver_info(curaii::make_unique_device_ptr<int>(layout.batches)) {}
};

// -------------------------------------------------------------------------------------------------
// CUDA Graph replay
// -------------------------------------------------------------------------------------------------

using PcaGraphKey = std::array<const void *, 2>;

template <typename Enqueue> cudaGraph_t capture_cuda_graph(cudaStream_t stream, Enqueue &&enqueue) {
  cudaGraph_t graph     = nullptr;
  bool        capturing = false;

  try {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    capturing = true;

    std::forward<Enqueue>(enqueue)();

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    capturing = false;
    return graph;
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

[[nodiscard]] bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

class PcaCudaGraph {
public:
  explicit PcaCudaGraph(PcaGraphKey key) : key_(key) {}

  ~PcaCudaGraph() noexcept {
    if (executable_ != nullptr) {
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
    }
  }

  PcaCudaGraph(const PcaCudaGraph &)            = delete;
  PcaCudaGraph &operator=(const PcaCudaGraph &) = delete;

  [[nodiscard]] bool matches(const PcaGraphKey &key) const noexcept {
    return executable_ != nullptr && key_ == key;
  }

  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

  template <typename Enqueue> void capture(cudaStream_t stream, Enqueue &&enqueue) {
    cudaGraph_t graph = capture_cuda_graph(stream, std::forward<Enqueue>(enqueue));

    try {
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable_, graph, 0));
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
    } catch (...) {
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
      throw;
    }
  }

private:
  PcaGraphKey     key_{};
  cudaGraphExec_t executable_{nullptr};
};

class PcaGraphCache {
public:
  void invalidate() {
    graphs_.clear();
    capture_enabled_ = true;
  }

  void enable_capture() noexcept { capture_enabled_ = true; }

  [[nodiscard]] bool try_launch(const PcaGraphKey &key, cudaStream_t stream) {
    const auto graph = std::find_if(graphs_.begin(), graphs_.end(),
                                    [&](const PcaCudaGraph &entry) { return entry.matches(key); });

    if (graph == graphs_.end()) {
      return false;
    }

    graph->launch(stream);
    graphs_.splice(graphs_.begin(), graphs_, graph);
    return true;
  }

  template <typename Enqueue>
  void try_capture(const PcaGraphKey &key, cudaStream_t stream, Enqueue &&enqueue) {
    if (!capture_enabled_) {
      return;
    }

    bool entry_inserted = false;
    try {
      graphs_.emplace_front(key);
      entry_inserted = true;

      graphs_.front().capture(stream, std::forward<Enqueue>(enqueue));

      if (graphs_.size() > capacity) {
        graphs_.pop_back();
      }
    } catch (const std::exception &error) {
      if (entry_inserted) {
        graphs_.pop_front();
      }
      capture_enabled_ = false;
      logger()->warn("[Pca] CUDA Graph capture disabled: {}", error.what());
    }
  }

private:
  // cuBLAS and cuSOLVER graph nodes embed their buffer addresses. Cache the finite set of rotating
  // pipeline buffers instead of attempting to update opaque library nodes.
  static constexpr size_t capacity = 128;

  bool                    capture_enabled_{true};
  std::list<PcaCudaGraph> graphs_;
};

// -------------------------------------------------------------------------------------------------
// Eigensolver abstraction
// -------------------------------------------------------------------------------------------------

class Eigensolver {
public:
  virtual ~Eigensolver() = default;

  Eigensolver(const Eigensolver &)            = delete;
  Eigensolver &operator=(const Eigensolver &) = delete;

  [[nodiscard]] virtual bool is_compatible_stream(cudaStream_t stream) const = 0;

  virtual void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) = 0;

protected:
  Eigensolver() = default;
};

std::unique_ptr<Eigensolver> make_eigensolver(const PcaLayout    &layout,
                                              const PcaWorkspace &workspace, cudaStream_t stream);

// -------------------------------------------------------------------------------------------------
// PCA task
// -------------------------------------------------------------------------------------------------

class PcaTask final : public holoflow::core::ISyncTask {
public:
  PcaTask(const PcaSettings &settings, const holoflow::core::TDesc &input_desc,
          const holoflow::core::SyncCreateCtx &ctx)
      : settings_(settings), input_desc_(input_desc), layout_(input_desc), stream_(ctx.stream),
        workspace_(layout_) {
    CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));
    eigensolver_ = make_eigensolver(layout_, workspace_, stream_);
  }

  [[nodiscard]] bool can_reuse(const holoflow::core::TDesc &input_desc, cudaStream_t stream) const {
    return input_desc.shape == input_desc_.shape && input_desc.strides == input_desc_.strides &&
           input_desc.dtype == input_desc_.dtype && input_desc.mem_loc == input_desc_.mem_loc &&
           eigensolver_->is_compatible_stream(stream);
  }

  void reconfigure(const PcaSettings &settings, cudaStream_t stream) {
    if (settings_ != settings) {
      settings_ = settings;
      graph_cache_.invalidate();
    }

    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));

      // Graph executables are not tied to the stream on which they were captured. Reusing them is
      // valid here because can_reuse() already guarantees that the CUDA context is unchanged.
      graph_cache_.enable_capture();
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range range("PCA Sync Task");

    if (stream_ == nullptr || stream_is_capturing(stream_)) {
      return enqueue(ctx);
    }

    const PcaGraphKey key{ctx.inputs[0].data(), ctx.outputs[0].data()};
    if (graph_cache_.try_launch(key, stream_)) {
      return holoflow::core::OpResult::Ok;
    }

    const auto result = enqueue(ctx);
    if (result == holoflow::core::OpResult::Ok) {
      graph_cache_.try_capture(key, stream_, [&]() { (void)enqueue(ctx); });
    }
    return result;
  }

private:
  holoflow::core::OpResult enqueue(holoflow::core::SyncCtx &ctx) {
    auto *input  = reinterpret_cast<float *>(ctx.inputs[0].data());
    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());

    enqueue_covariance(input);
    enqueue_eigendecomposition();
    enqueue_projection(input, output);

    return holoflow::core::OpResult::Ok;
  }

  void enqueue_covariance(const float *input) {
    nvtx3::scoped_range range("PCA covariance");

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    // Independent GEMMs preserve cuBLAS Split-K selection for the large sample dimension.
    for (size_t batch = 0; batch < layout_.batches; ++batch) {
      const auto input_offset  = static_cast<long long>(batch) * layout_.input_batch_stride;
      const auto matrix_offset = static_cast<long long>(batch) * layout_.matrix_batch_stride;

      CUBLAS_CHECK(cublasGemmEx(cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, layout_.features,
                                layout_.features, layout_.samples, &alpha, input + input_offset,
                                CUDA_R_32F, layout_.samples, input + input_offset, CUDA_R_32F,
                                layout_.samples, &beta, workspace_.matrices.get() + matrix_offset,
                                CUDA_R_32F, layout_.features, CUBLAS_COMPUTE_32F_FAST_16F,
                                CUBLAS_GEMM_DEFAULT));
    }
  }

  void enqueue_eigendecomposition() {
    nvtx3::scoped_range range("PCA eigendecomposition");

    eigensolver_->solve(workspace_.matrices.get(), workspace_.eigenvalues.get(),
                        workspace_.solver_info.get(), stream_);
  }

  void enqueue_projection(const float *input, float *output) {
    nvtx3::scoped_range range("PCA projection");

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    const int components = settings_.components();

    const auto eigenvector_offset = static_cast<long long>(settings_.begin) * layout_.features;
    const auto output_stride      = layout_.output_batch_stride(components);

    for (size_t batch = 0; batch < layout_.batches; ++batch) {
      const auto input_offset  = static_cast<long long>(batch) * layout_.input_batch_stride;
      const auto matrix_offset = static_cast<long long>(batch) * layout_.matrix_batch_stride;
      const auto output_offset = static_cast<long long>(batch) * output_stride;

      CUBLAS_CHECK(
          cublasGemmEx(cublas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, layout_.samples, components,
                       layout_.features, &alpha, input + input_offset, CUDA_R_32F, layout_.samples,
                       workspace_.matrices.get() + matrix_offset + eigenvector_offset, CUDA_R_32F,
                       layout_.features, &beta, output + output_offset, CUDA_R_32F, layout_.samples,
                       CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
    }
  }

  PcaSettings           settings_;
  holoflow::core::TDesc input_desc_;
  PcaLayout             layout_;
  cudaStream_t          stream_;

  curaii::CublasHandle         cublas_;
  PcaWorkspace                 workspace_;
  std::unique_ptr<Eigensolver> eigensolver_;
  PcaGraphCache                graph_cache_;
};

// -------------------------------------------------------------------------------------------------
// CUDA Driver API support
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
  if (result == CUDA_SUCCESS) {
    return;
  }

  const auto message = driver_error_message(result, expression, file, line);
  logger()->error("{}", message);
  throw std::runtime_error(message);
}

#define PCA_DRIVER_CHECK(expr) driver_check((expr), #expr, __FILE__, __LINE__)

class ScopedCudaContext {
public:
  explicit ScopedCudaContext(CUcontext context) {
    CUcontext current = nullptr;
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&current));

    if (current != context) {
      PCA_DRIVER_CHECK(cuCtxPushCurrent(context));
      pushed_ = true;
    }
  }

  ~ScopedCudaContext() noexcept {
    if (!pushed_) {
      return;
    }

    CUcontext  popped = nullptr;
    const auto result = cuCtxPopCurrent(&popped);
    if (result != CUDA_SUCCESS) {
      logger()->critical("{}", driver_error_message(result, "cuCtxPopCurrent", __FILE__, __LINE__));
      std::abort();
    }
  }

  ScopedCudaContext(const ScopedCudaContext &)            = delete;
  ScopedCudaContext &operator=(const ScopedCudaContext &) = delete;

private:
  bool pushed_{false};
};

CUcontext cuda_context_for_stream(cudaStream_t stream) {
  CUcontext context = nullptr;

  if (stream != nullptr) {
    PCA_DRIVER_CHECK(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context));
  } else {
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&context));
  }

  if (context == nullptr) {
    throw std::runtime_error("PCA eigensolver requires an active CUDA context");
  }

  return context;
}

template <typename T> T read_module_constant(CUmodule module, const char *name) {
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

CUfunction module_function(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  PCA_DRIVER_CHECK(cuModuleGetFunction(&function, module, name));
  return function;
}

// -------------------------------------------------------------------------------------------------
// Conventional cuSOLVER eigensolver
// -------------------------------------------------------------------------------------------------

class CusolverEigensolver final : public Eigensolver {
public:
  CusolverEigensolver(int features, size_t batches, float *matrices, float *eigenvalues,
                      cudaStream_t stream)
      : features_(features), batches_(batches), context_(cuda_context_for_stream(stream)),
        handle_(std::make_unique<curaii::CusolverDnHandle>()),
        params_(std::make_unique<curaii::CusolverDnParams>()) {
    if (batches_ > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
      throw std::invalid_argument("[Pca] Batch count exceeds the cuSOLVER API limit");
    }

    CUSOLVER_CHECK(cusolverDnSetStream(handle_->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
        handle_->get(), params_->get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, features_,
        CUDA_R_32F, matrices, features_, CUDA_R_32F, eigenvalues, CUDA_R_32F,
        &device_workspace_size_, &host_workspace_size_, static_cast<int64_t>(batches_)));

    if (device_workspace_size_ != 0) {
      device_workspace_ = curaii::make_unique_device_ptr<std::byte>(device_workspace_size_);
    }
    host_workspace_.resize(host_workspace_size_);
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const override {
    return cuda_context_for_stream(stream) == context_;
  }

  void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) override {
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument(
          "[Pca] cuSOLVER eigensolver and PCA stream use different contexts");
    }

    CUSOLVER_CHECK(cusolverDnSetStream(handle_->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        handle_->get(), params_->get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, features_,
        CUDA_R_32F, matrices, features_, CUDA_R_32F, eigenvalues, CUDA_R_32F,
        device_workspace_.get(), device_workspace_size_, host_workspace_.data(),
        host_workspace_size_, info, static_cast<int64_t>(batches_)));
  }

private:
  int       features_;
  size_t    batches_;
  CUcontext context_{nullptr};

  std::unique_ptr<curaii::CusolverDnHandle> handle_;
  std::unique_ptr<curaii::CusolverDnParams> params_;

  curaii::unique_device_ptr<std::byte> device_workspace_;
  size_t                               device_workspace_size_{0};
  std::vector<std::byte>               host_workspace_;
  size_t                               host_workspace_size_{0};
};

// -------------------------------------------------------------------------------------------------
// cuSolverDx runtime compilation
// -------------------------------------------------------------------------------------------------

const char *cusolverdx_source();

namespace cusolverdx_runtime {

struct Assets {
  std::filesystem::path cuda_include;
  std::filesystem::path cusolverdx_include;
  std::filesystem::path cutlass_include;
  std::filesystem::path cusolverdx_fatbin;
};

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
  if (result == NVJITLINK_SUCCESS) {
    return;
  }

  const auto log     = jitlink_error_log(linker);
  const auto message = std::format("nvJitLink error: {} at {}:{}\n  expression : {}\n{}",
                                   static_cast<int>(result), file, line, expression, log);
  logger()->error("{}", message);
  throw std::runtime_error(message);
}

#define PCA_NVJITLINK_CHECK(linker, expr) jitlink_check((linker), (expr), #expr, __FILE__, __LINE__)

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

bool assets_exist(const Assets &assets) {
  return std::filesystem::is_directory(assets.cuda_include) &&
         std::filesystem::is_regular_file(assets.cuda_include / "cuda_runtime.h") &&
         std::filesystem::is_directory(assets.cuda_include / "cccl") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx.hpp") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx_io.hpp") &&
         std::filesystem::is_directory(assets.cutlass_include) &&
         std::filesystem::is_regular_file(assets.cusolverdx_fatbin);
}

Assets locate_assets() {
  const auto installed_root =
      (executable_directory() / HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR).lexically_normal();

  const Assets installed{
      installed_root / "cuda/include",
      installed_root / "mathdx/include",
      installed_root / "mathdx/cutlass/include",
      installed_root / "mathdx/lib/libcusolverdx.fatbin",
  };

  if (assets_exist(installed)) {
    return installed;
  }

  const Assets build{
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

std::vector<char> compile_lto_ir(int features, int solver_sm, int architecture,
                                 const Assets &assets) {
  curaii::NvrtcProgram program(cusolverdx_source(), "pca_heev_kernel.cu");

  std::vector<std::string> options = {
      "--std=c++17",
      "--device-as-default-execution-space",
      "-dlto",
      "--relocatable-device-code=true",
      std::format("--gpu-architecture=sm_{}", architecture),
      std::format("-DPCA_FEATURES={}", features),
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
                             const Assets &assets) {
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

std::vector<char> compile_cubin(int features, int solver_sm, int architecture) {
  const auto assets = locate_assets();
  const auto lto_ir = compile_lto_ir(features, solver_sm, architecture, assets);
  return link_cubin(lto_ir, architecture, assets);
}

} // namespace cusolverdx_runtime

// -------------------------------------------------------------------------------------------------
// cuSolverDx eigensolver
// -------------------------------------------------------------------------------------------------

struct DxKernel {
  CUfunction   function{nullptr};
  unsigned int block_x{0};
  unsigned int shared_memory{0};
};

class CusolverDxEigensolver final : public Eigensolver {
public:
  CusolverDxEigensolver(int features, size_t batches, cudaStream_t stream)
      : features_(features), batches_(batches) {
    CUDA_CHECK(cudaGetDevice(&device_));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_));

    architecture_       = major * 10 + minor;
    const int solver_sm = architecture_ * 10;

    PCA_DRIVER_CHECK(cuInit(0));
    context_ = cuda_context_for_stream(stream);

    ScopedCudaContext context_guard(context_);

    const auto cubin = cusolverdx_runtime::compile_cubin(features_, solver_sm, architecture_);
    PCA_DRIVER_CHECK(cuModuleLoadDataEx(&module_, cubin.data(), 0, nullptr, nullptr));

    try {
      load_kernel_configuration();
      configure_shared_memory();
    } catch (...) {
      (void)cuModuleUnload(module_);
      module_ = nullptr;
      throw;
    }
  }

  ~CusolverDxEigensolver() noexcept override {
    if (module_ == nullptr) {
      return;
    }

    try {
      ScopedCudaContext context_guard(context_);
      const auto        result = cuModuleUnload(module_);

      if (result != CUDA_SUCCESS) {
        logger()->critical("{}",
                           driver_error_message(result, "cuModuleUnload", __FILE__, __LINE__));
        std::abort();
      }
    } catch (...) {
      std::abort();
    }
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const override {
    return cuda_context_for_stream(stream) == context_;
  }

  void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) override {
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument(
          "[Pca] cuSolverDx eigensolver and PCA stream use different contexts");
    }

    const size_t full_batches = (batches_ / batches_per_block_) * batches_per_block_;
    if (full_batches != 0) {
      launch_batched(matrices, eigenvalues, info, full_batches, stream);
    }

    const size_t tail_batches = batches_ - full_batches;
    if (tail_batches != 0) {
      launch_tail(matrices, eigenvalues, info, full_batches, tail_batches, stream);
    }
  }

private:
  void load_kernel_configuration() {
    batched_kernel_ = {
        .function      = module_function(module_, "pca_heev_batched"),
        .block_x       = read_module_constant<unsigned int>(module_, "pca_batched_block_x"),
        .shared_memory = read_module_constant<unsigned int>(module_, "pca_batched_shared_memory"),
    };

    tail_kernel_ = {
        .function      = module_function(module_, "pca_heev_tail"),
        .block_x       = read_module_constant<unsigned int>(module_, "pca_tail_block_x"),
        .shared_memory = read_module_constant<unsigned int>(module_, "pca_tail_shared_memory"),
    };

    batches_per_block_ = read_module_constant<unsigned int>(module_, "pca_batches_per_block");
  }

  void configure_shared_memory() {
    int max_shared_memory = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device_));

    const bool uses_batched = batches_ >= batches_per_block_;
    const bool uses_tail    = batches_ < batches_per_block_ || (batches_ % batches_per_block_) != 0;

    if (uses_batched &&
        batched_kernel_.shared_memory > static_cast<unsigned int>(max_shared_memory)) {
      throw_shared_memory_error(batched_kernel_.shared_memory, max_shared_memory);
    }

    if (uses_tail && tail_kernel_.shared_memory > static_cast<unsigned int>(max_shared_memory)) {
      throw_shared_memory_error(tail_kernel_.shared_memory, max_shared_memory);
    }

    if (uses_batched) {
      PCA_DRIVER_CHECK(cuFuncSetAttribute(batched_kernel_.function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          static_cast<int>(batched_kernel_.shared_memory)));
    }

    if (uses_tail) {
      PCA_DRIVER_CHECK(cuFuncSetAttribute(tail_kernel_.function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          static_cast<int>(tail_kernel_.shared_memory)));
    }
  }

  [[noreturn]] void throw_shared_memory_error(unsigned int required, int max_shared_memory) const {
    throw std::runtime_error(
        std::format("[Pca] cuSolverDx HEEV for {} features requires {} bytes of dynamic "
                    "shared memory, but device sm_{} provides {} bytes",
                    features_, required, architecture_, max_shared_memory));
  }

  void launch_batched(float *matrices, float *eigenvalues, int *info, size_t full_batches,
                      cudaStream_t stream) const {
    CUdeviceptr matrix_arg      = reinterpret_cast<CUdeviceptr>(matrices);
    CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues);
    CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info);
    auto        batch_arg       = static_cast<unsigned int>(full_batches);
    void       *arguments[]     = {&matrix_arg, &eigenvalues_arg, &info_arg, &batch_arg};

    PCA_DRIVER_CHECK(cuLaunchKernel(batched_kernel_.function,
                                    static_cast<unsigned int>(full_batches / batches_per_block_), 1,
                                    1, batched_kernel_.block_x, 1, 1, batched_kernel_.shared_memory,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  void launch_tail(float *matrices, float *eigenvalues, int *info, size_t full_batches,
                   size_t tail_batches, cudaStream_t stream) const {
    const size_t matrix_offset = full_batches * static_cast<size_t>(features_) * features_;
    const size_t value_offset  = full_batches * static_cast<size_t>(features_);

    CUdeviceptr matrix_arg      = reinterpret_cast<CUdeviceptr>(matrices + matrix_offset);
    CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues + value_offset);
    CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info + full_batches);
    auto        batch_arg       = static_cast<unsigned int>(tail_batches);
    void       *arguments[]     = {&matrix_arg, &eigenvalues_arg, &info_arg, &batch_arg};

    PCA_DRIVER_CHECK(cuLaunchKernel(tail_kernel_.function, static_cast<unsigned int>(tail_batches),
                                    1, 1, tail_kernel_.block_x, 1, 1, tail_kernel_.shared_memory,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  int       features_;
  size_t    batches_;
  int       device_{0};
  int       architecture_{0};
  CUcontext context_{nullptr};
  CUmodule  module_{nullptr};

  DxKernel     batched_kernel_;
  DxKernel     tail_kernel_;
  unsigned int batches_per_block_{1};
};

// -------------------------------------------------------------------------------------------------
// Eigensolver selection
// -------------------------------------------------------------------------------------------------

constexpr int cusolverdx_max_features_exclusive = 256;

std::unique_ptr<Eigensolver> make_eigensolver(const PcaLayout    &layout,
                                              const PcaWorkspace &workspace, cudaStream_t stream) {
  if (layout.features < cusolverdx_max_features_exclusive) {
    try {
      return std::make_unique<CusolverDxEigensolver>(layout.features, layout.batches, stream);
    } catch (const std::exception &error) {
      logger()->warn("[Pca] cuSolverDx initialization failed for depth {}: {}\n"
                     "[Pca] Falling back to the conventional cuSOLVER eigensolver",
                     layout.features, error.what());
    }
  }

  try {
    return std::make_unique<CusolverEigensolver>(layout.features, layout.batches,
                                                 workspace.matrices.get(),
                                                 workspace.eigenvalues.get(), stream);
  } catch (const std::exception &error) {
    logger()->error("[Pca] Failed to initialize any GPU eigensolver for depth {}: {}",
                    layout.features, error.what());

    throw std::runtime_error(std::format("PCA could not initialize a GPU eigensolver for depth {}. "
                                         "See the terminal log for details.",
                                         layout.features));
  }
}

// -------------------------------------------------------------------------------------------------
// Embedded cuSolverDx device program
// -------------------------------------------------------------------------------------------------

const char *cusolverdx_source() {
  static constexpr char source[] = R"cusolverdx(
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

  return source;
}

#undef PCA_NVJITLINK_CHECK
#undef PCA_DRIVER_CHECK

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
  const auto check = [&](bool condition, const std::string &message) {
    if (!condition) {
      logger()->error("[PcaFactory::infer] error: {}", message);
      throw std::invalid_argument("PcaFactory inference error: " + message);
    }
  };

  const auto settings = jsettings.get<PcaSettings>();

  check(input_descs.size() == 1, "expected exactly one input");

  const auto &input_desc = input_descs.front();
  check(input_desc.rank() >= 3, "expected input rank >= 3");
  check(input_desc.dtype == holoflow::core::DType::F32,
        "cuSolverDx PCA currently supports F32 input only");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");

  const PcaLayout layout(input_desc);
  check(settings.end <= layout.features, "expected end <= n_features");

  auto output_shape                    = input_desc.shape;
  output_shape.at(layout.feature_axis) = static_cast<size_t>(settings.components());

  holoflow::core::TDesc output_desc(output_shape, input_desc.dtype, input_desc.mem_loc);

  return holoflow::core::InferResult{
      .input_descs   = {input_desc},
      .output_descs  = {output_desc},
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
  this->infer(input_descs, jsettings);

  return std::make_unique<PcaTask>(jsettings.get<PcaSettings>(), input_descs.front(), ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  this->infer(input_descs, jsettings);

  const auto  settings   = jsettings.get<PcaSettings>();
  const auto &input_desc = input_descs.front();

  auto *pca = dynamic_cast<PcaTask *>(old_task.get());
  if (pca != nullptr && pca->can_reuse(input_desc, ctx.stream)) {
    pca->reconfigure(settings, ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
