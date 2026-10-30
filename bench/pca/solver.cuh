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
#include "assets.hh"
#include "curaii/nvrtc.hh"
#include <cuda.h>
#include <nvJitLink.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace pca_bench {
inline void driver(CUresult code) {
  if (code != CUDA_SUCCESS)
    throw std::runtime_error("PCA CUDA driver error " + std::to_string(code));
}
inline void linkcheck(nvJitLinkResult code) {
  if (code != NVJITLINK_SUCCESS)
    throw std::runtime_error("PCA JIT linker error " + std::to_string(code));
}
struct RuntimeSolver {
  CUmodule     module{};
  CUfunction   function{};
  unsigned int threads{}, shared{};
  ~RuntimeSolver() {
    if (module)
      cuModuleUnload(module);
  }
  RuntimeSolver() {
    const char          *source = R"solver(
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>
using Solver=decltype(cusolverdx::Size<32>()+cusolverdx::Precision<float>()+cusolverdx::Type<cusolverdx::type::real>()+cusolverdx::Function<cusolverdx::heev>()+cusolverdx::FillMode<cusolverdx::fill_mode::lower>()+cusolverdx::Arrangement<cusolverdx::arrangement::col_major>()+cusolverdx::Job<cusolverdx::job::overwrite_vectors>()+cusolverdx::SM<PCA_BENCH_SM>()+cusolverdx::Block()+cusolverdx::BatchesPerBlock<1>());
extern "C" __global__ void solve(float* matrix,float* values,int* info) {
 extern __shared__ __align__(16) cusolverdx::byte smem[];
 auto [a,l,w]=cusolverdx::shared_memory::slice<float,float,float>(smem,alignof(float),Solver::lda*32,alignof(float),32,alignof(float));
 cusolverdx::copy_2d<Solver,32,32,cusolverdx::arrangement::col_major,1>(matrix,32,a,Solver::lda);
 __syncthreads();
 __shared__ float block_trace;
 if(threadIdx.x==0){block_trace=0;for(int i=0;i<32;++i)block_trace+=matrix[i*32+i];}
 __syncthreads();
 float trace=block_trace;
 if(trace==0){for(int i=threadIdx.x;i<1024;i+=blockDim.x)matrix[i]=(i/32==i%32?1.f:0.f);for(int i=threadIdx.x;i<32;i+=blockDim.x)values[i]=0;if(threadIdx.x==0)*info=0;return;}
 Solver().execute(a,Solver::lda,l,w,info);
 __syncthreads();
 __shared__ int block_status;
 if(threadIdx.x==0)block_status=*info;
 __syncthreads();
 if(block_status!=0)return;

 cusolverdx::copy_2d<Solver,32,32,cusolverdx::arrangement::col_major,1>(a,Solver::lda,matrix,32);
 for(int i=threadIdx.x;i<32;i+=blockDim.x)l[i]*=17179869184.f;
 __syncthreads();
 cusolverdx::copy_2d<Solver,32,1,cusolverdx::arrangement::col_major,1>(l,32,values,32);
}

extern "C" __constant__ unsigned int solver_threads=Solver::block_dim.x;
extern "C" __constant__ unsigned int solver_shared=Solver::shared_memory_size;

)solver";
    curaii::NvrtcProgram program(source, "pca_benchmark_solver.cu");
    std::string          cuda_include = pca_cuda_include;
    cuda_include                      = cuda_include.substr(0, cuda_include.find(';'));
    std::vector<std::string>  opts = {"--std=c++17",
                                      "--device-as-default-execution-space",
                                      "-dlto",
                                      "--relocatable-device-code=true",
                                      "--gpu-architecture=sm_" + std::to_string(PCA_BENCH_SM / 10),
                                      "-DPCA_BENCH_SM=" + std::to_string(PCA_BENCH_SM),
                                      std::string("--include-path=") + cuda_include,
                                      std::string("--include-path=") + cuda_include + "/cccl",
                                      std::string("--include-path=") + pca_dx_include,
                                      std::string("--include-path=") + pca_cutlass_include};
    std::vector<const char *> ptrs;
    for (auto &o : opts)
      ptrs.push_back(o.c_str());
    auto result = nvrtcCompileProgram(program.get(), int(ptrs.size()), ptrs.data());
    if (result != NVRTC_SUCCESS) {
      size_t n;
      nvrtcGetProgramLogSize(program.get(), &n);
      std::vector<char> log(n);
      nvrtcGetProgramLog(program.get(), log.data());
      throw std::runtime_error(log.data());
    }
    size_t size;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program.get(), &size));
    std::vector<char> ir(size);
    NVRTC_CHECK(nvrtcGetLTOIR(program.get(), ir.data()));
    nvJitLinkHandle linker{};
    std::string     arch      = "-arch=sm_" + std::to_string(PCA_BENCH_SM / 10);
    const char     *options[] = {"-lto", arch.c_str()};
    linkcheck(nvJitLinkCreate(&linker, 2, options));
    try {
      linkcheck(nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, pca_solver_fatbin));
      linkcheck(nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, ir.data(), ir.size(), "solver"));
      linkcheck(nvJitLinkComplete(linker));
      linkcheck(nvJitLinkGetLinkedCubinSize(linker, &size));
      std::vector<char> cubin(size);
      linkcheck(nvJitLinkGetLinkedCubin(linker, cubin.data()));
      driver(cuModuleLoadData(&module, cubin.data()));
      linkcheck(nvJitLinkDestroy(&linker));
    } catch (...) {
      if (linker)
        nvJitLinkDestroy(&linker);
      throw;
    }
    driver(cuModuleGetFunction(&function, module, "solve"));
    CUdeviceptr address;
    driver(cuModuleGetGlobal(&address, &size, module, "solver_threads"));
    driver(cuMemcpyDtoH(&threads, address, sizeof(threads)));
    driver(cuModuleGetGlobal(&address, &size, module, "solver_shared"));
    driver(cuMemcpyDtoH(&shared, address, sizeof(shared)));
  }
  void launch(float *matrix, float *values, int *info, cudaStream_t stream) {
    void *args[] = {&matrix, &values, &info};
    driver(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, shared, stream, args, nullptr));
  }
};
} // namespace pca_bench
