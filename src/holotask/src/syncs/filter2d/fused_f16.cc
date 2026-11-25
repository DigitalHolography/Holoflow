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

#include "fused_f16.hh"
#include "curaii/cuda.hh"
#include "curaii/nvrtc.hh"
#include "filter_fused_f16_source.hh"
#include "logger.hh"
#include "runtime_assets.hh"
#include <algorithm>
#include <atomic>
#include <cuda.h>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace holotask::syncs {
namespace {
void driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(text ? text : "F16 Filter2D driver failure");
  }
}
std::vector<char> compile(int sm, bool half) {
  static std::mutex                               mutex;
  static std::map<std::string, std::vector<char>> cache;
  const auto      key = std::to_string(sm) + "/" + std::to_string(half);
  std::lock_guard lock(mutex);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  const auto cuda   = holotask::detail::cuda_include_directory();
  const auto module = holotask::detail::containing_module_directory();
  auto       dx     = module / "_assets/nvrtc/mathdx/include";
  if (!std::filesystem::exists(dx / "cufftdx.hpp"))
    dx = module / HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR / "mathdx/include";
  if (!std::filesystem::exists(dx / "cufftdx.hpp"))
    dx = HOLOFLOW_CUSOLVERDX_INCLUDE_DIR;
  curaii::NvrtcProgram      program(holotask::detail::filter_fused_f16_source, "fused_f16.cu");
  std::vector<std::string>  options = {"--std=c++17",
                                       "--device-as-default-execution-space",
                                       "--gpu-architecture=compute_" + std::to_string(sm / 10),
                                       "-DFFT_SM=" + std::to_string(sm),
                                       "-DINPUT_HALF=" + std::to_string(half),
                                       "--include-path=" + cuda.string(),
                                       "--include-path=" + (cuda / "cccl").string(),
                                       "--include-path=" + dx.string()};
  std::vector<const char *> ptrs;
  for (auto &option : options)
    ptrs.push_back(option.c_str());
  auto result = nvrtcCompileProgram(program.get(), int(ptrs.size()), ptrs.data());
  if (result != NVRTC_SUCCESS) {
    size_t size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program.get(), &size));
    std::vector<char> log(size);
    NVRTC_CHECK(nvrtcGetProgramLog(program.get(), log.data()));
    throw std::runtime_error(log.data());
  }
  size_t size;
  NVRTC_CHECK(nvrtcGetPTXSize(program.get(), &size));
  std::vector<char> ptx(size);
  NVRTC_CHECK(nvrtcGetPTX(program.get(), ptx.data()));
  cache.emplace(key, ptx);
  return ptx;
}
struct Module {
  CUmodule value{};
  ~Module() {
    if (value)
      cuModuleUnload(value);
  }
};

class FusedFilter final : public holoflow::core::ISyncTask {
  Module                                  module_;
  holoflow::core::TDesc                   input_;
  Filter2DSettings                        settings_;
  CUcontext                               context_{};
  cudaStream_t                            stream_;
  unsigned int                            planes_, xs_, ys_;
  curaii::unique_device_ptr<unsigned int> a_, b_;
  curaii::unique_device_ptr<float>        mask_, peaks_;
  curaii::unique_device_ptr<int>          scales_;
  CUfunction                              peak_, scale_, mask_fn_, steps_[3];
  CUfunction                              function(const char *name) {
    CUfunction f;
    driver(cuModuleGetFunction(&f, module_.value, name));
    return f;
  }
  unsigned int constant(const char *name) {
    CUdeviceptr  ptr;
    size_t       size;
    unsigned int value;
    driver(cuModuleGetGlobal(&ptr, &size, module_.value, name));
    driver(cuMemcpyDtoH(&value, ptr, sizeof(value)));
    return value;
  }
  void launch(CUfunction f, unsigned int grid, unsigned int block, unsigned int shared, void **args,
              unsigned int rows = 1) {
    driver(cuLaunchKernel(f, grid, 1, 1, block, rows, 1, shared, stream_, args, nullptr));
  }
  void update_mask() {
    auto  m      = mask_.get();
    void *args[] = {&m, &settings_.r_inner, &settings_.r_outer, &settings_.s_inner,
                    &settings_.s_outer};
    launch(mask_fn_, (320 * 512 + 255) / 256, 256, 0, args);
  }

public:
  FusedFilter(const holoflow::core::TDesc &input, Filter2DSettings settings, cudaStream_t stream)
      : input_(input), settings_(settings), stream_(stream),
        planes_(static_cast<unsigned int>(input.num_elements() / (320 * 512))),
        a_(curaii::make_unique_device_ptr<unsigned int>(planes_ * 257 * 320)),
        b_(curaii::make_unique_device_ptr<unsigned int>(planes_ * 257 * 320)),
        mask_(curaii::make_unique_device_ptr<float>(320 * 512)),
        peaks_(curaii::make_unique_device_ptr<float>(planes_ * 64)),
        scales_(curaii::make_unique_device_ptr<int>(planes_ * 2)) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    auto ptx =
        compile(prop.major * 100 + prop.minor * 10, input.dtype == holoflow::core::DType::F16);
    driver(cuCtxGetCurrent(&context_));
    driver(cuModuleLoadData(&module_.value, ptx.data()));
    peak_     = function("peak_kernel");
    scale_    = function("scale_kernel");
    mask_fn_  = function("mask_kernel");
    steps_[0] = function("forward_x");
    steps_[1] = function("filter_y");
    steps_[2] = function("inverse_x");
    xs_       = constant("x_shared");
    ys_       = constant("y_shared");
    for (int i = 0; i < 3; ++i)
      driver(cuFuncSetAttribute(steps_[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                int(i == 1 ? ys_ : xs_)));
    update_mask();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
  ~FusedFilter() override { CUDA_CHECK_NT(cudaStreamSynchronize(stream_)); }
  bool update(const holoflow::core::TDesc &input, Filter2DSettings settings, cudaStream_t stream) {
    CUcontext context{};
    if (stream)
      driver(cuStreamGetCtx(stream, &context));
    else
      driver(cuCtxGetCurrent(&context));
    if (context != context_ || input.shape != input_.shape || input.strides != input_.strides ||
        input.dtype != input_.dtype || input.mem_loc != input_.mem_loc ||
        input.offset != input_.offset || settings.normalize != settings_.normalize ||
        settings.use_fused_f16 != settings_.use_fused_f16)
      return false;
    stream_ = stream;
    if (settings != settings_) {
      settings_ = settings;
      update_mask();
    }
    return true;
  }
  bool supports_cuda_graph() const noexcept override { return true; }
  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    holoflow::core::SyncCtx call{ctx.inputs, ctx.outputs, nullptr, nullptr, nullptr};
    (void)execute(call);
  }
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    void *in = ctx.inputs[0].data();
    auto  a = a_.get(), b = b_.get();
    auto  s            = scales_.get();
    auto  m            = mask_.get();
    auto  peaks        = peaks_.get();
    int   stage        = 0;
    void *peak_args[]  = {&in, &peaks, &stage};
    void *scale_args[] = {&peaks, &s, &stage};
    launch(peak_, planes_ * 64, 256, 0, peak_args);
    launch(scale_, planes_, 64, 0, scale_args);
    void *forward_args[] = {&in, &a, &s};
    launch(steps_[0], planes_ * 40, 64, xs_, forward_args, 4);
    stage = 1;
    in    = a;
    launch(peak_, planes_ * 64, 256, 0, peak_args);
    launch(scale_, planes_, 64, 0, scale_args);
    void *column_args[] = {&a, &b, &s, &m};
    launch(steps_[1], planes_ * 129, 16, ys_, column_args);
    void *out            = ctx.outputs[0].data();
    void *inverse_args[] = {&b, &out, &s, &settings_.normalize};
    launch(steps_[2], planes_ * 40, 64, xs_, inverse_args, 4);
    return holoflow::core::OpResult::Ok;
  }
};
} // namespace
bool supports_fused_f16_filter(const holoflow::core::TDesc &input) {
  if (input.rank() < 2 || input.shape[input.rank() - 2] != 320 || input.shape.back() != 512 ||
      input.num_elements() == 0 || input.num_elements() > size_t(2147483647) ||
      (input.dtype != holoflow::core::DType::F16 && input.dtype != holoflow::core::DType::F32))
    return false;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return (prop.major == 8 && prop.minor == 6) || (prop.major == 12 && prop.minor == 0);
}
std::unique_ptr<holoflow::core::ISyncTask>
create_fused_f16_filter(const holoflow::core::TDesc &input, Filter2DSettings settings,
                        cudaStream_t stream) {
  return std::make_unique<FusedFilter>(input, settings, stream);
}
bool update_fused_f16_filter(holoflow::core::ISyncTask &task, const holoflow::core::TDesc &input,
                             Filter2DSettings settings, cudaStream_t stream) {
  auto *f = dynamic_cast<FusedFilter *>(&task);
  return f && f->update(input, settings, stream);
}
} // namespace holotask::syncs
