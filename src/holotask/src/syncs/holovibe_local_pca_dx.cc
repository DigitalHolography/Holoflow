// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include <cuda.h>
#include <nvJitLink.h>
#include <windows.h>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>
#include "curaii/cuda.hh"
#include "curaii/nvrtc.hh"
#include "logger.hh"

namespace holotask::syncs {
namespace {
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
                                 const Assets &assets, const char *source) {
  curaii::NvrtcProgram program(source, "pca_heev_kernel.cu");

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

std::vector<char> compile_cubin(int features, int solver_sm, int architecture,
                                const char *source) {
  const auto assets = locate_assets();
  const auto lto_ir = compile_lto_ir(features, solver_sm, architecture, assets, source);
  return link_cubin(lto_ir, architecture, assets);
}

} // namespace cusolverdx_runtime
} // namespace
std::vector<char> compile_local_pca_heev_cubin(int frames, int architecture) {
  static constexpr char source[] = R"cusolverdx(
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>
using namespace cusolverdx;
using Solver = decltype(Size<PCA_FEATURES>() + Precision<float>() + Type<type::complex>() +
                        Function<heev>() + FillMode<fill_mode::lower>() +
                        Arrangement<arrangement::col_major>() +
                        Job<job::overwrite_vectors>() + SM<PCA_SOLVER_SM>() + Block() +
                        BatchesPerBlock<1>());
extern "C" __constant__ unsigned int local_pca_block_x = Solver::block_dim.x;
extern "C" __constant__ unsigned int local_pca_shared_memory = Solver::shared_memory_size;
extern "C" __global__ void local_pca_heev(cuFloatComplex *matrix, float *eigenvalues,
                                          int *info) {
  extern __shared__ __align__(16) cusolverdx::byte shared[];
  auto [a, values, workspace] =
      cusolverdx::shared_memory::slice<Solver::a_data_type, float, Solver::a_data_type>(
          shared, alignof(Solver::a_data_type), Solver::lda * PCA_FEATURES,
          alignof(float), PCA_FEATURES, alignof(Solver::a_data_type));
  auto *global = reinterpret_cast<Solver::a_data_type *>(matrix);
  cusolverdx::copy_2d<Solver, PCA_FEATURES, PCA_FEATURES, arrangement::col_major, 1>(
      global, PCA_FEATURES, a, Solver::lda);
  __syncthreads();
  Solver().execute(a, Solver::lda, values, workspace, info);
  cusolverdx::copy_2d<Solver, PCA_FEATURES, 1, arrangement::col_major, 1>(
      values, PCA_FEATURES, eigenvalues, PCA_FEATURES);
  __syncthreads();
  cusolverdx::copy_2d<Solver, PCA_FEATURES, PCA_FEATURES, arrangement::col_major, 1>(
      a, Solver::lda, global, PCA_FEATURES);
}
)cusolverdx";
  return cusolverdx_runtime::compile_cubin(frames, architecture * 10, architecture, source);
}


} // namespace holotask::syncs
