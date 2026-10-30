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
#include "../fresnel_cuda_graph.cuh"
#include "curaii/cuda.hh"
#include "curaii/nvrtc.hh"
#include "fresnel_fused_f16_source.hh"
#include "logger.hh"
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
#include <windows.h>
namespace holotask::syncs {
namespace {
void driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(text ? text : "F16 Fresnel driver failure");
  }
}
std::filesystem::path asset_root() {
  wchar_t path[32768];
  auto    length = GetModuleFileNameW(nullptr, path, 32768);
  if (!length)
    throw std::runtime_error("Cannot locate runtime assets");
  return (std::filesystem::path(std::wstring(path, length)).parent_path() /
          HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR)
      .lexically_normal();
}
std::vector<char> compile(int sm, bool half, int components, bool reduce) {
  static std::mutex                               mutex;
  static std::map<std::string, std::vector<char>> cache;
  const auto      key = std::to_string(sm) + "/" + std::to_string(half) + "/" +
                        std::to_string(components) + "/" + std::to_string(reduce);
  std::lock_guard lock(mutex);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  auto root = asset_root();
  auto cuda = root / "cuda/include", dx = root / "mathdx/include";
  if (!std::filesystem::exists(cuda / "cuda_fp16.h"))
    cuda = HOLOFLOW_CUDA_INCLUDE_DIR;
  if (!std::filesystem::exists(dx / "cufftdx.hpp"))
    dx = HOLOFLOW_CUSOLVERDX_INCLUDE_DIR;
  curaii::NvrtcProgram      program(holotask::detail::fresnel_fused_f16_source, "fused_f16.cu");
  std::vector<std::string>  options = {"--std=c++17",
                                       "--device-as-default-execution-space",
                                       "--gpu-architecture=compute_" + std::to_string(sm / 10),
                                       "-DFFT_SM=" + std::to_string(sm),
                                       "-DINPUT_HALF=" + std::to_string(half),
                                       "-DCOMPONENTS=" + std::to_string(components),
                                       "-DREDUCE_MEAN=" + std::to_string(reduce),
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
class FusedFresnel final : public holoflow::core::ISyncTask {
  std::list<detail::FresnelCudaGraph>     graphs_;
  bool                                    graph_capture_enabled_ = true;
  Module                                  module_;
  cudaStream_t                            stream_;
  int                                     planes_, components_, groups_, output_count_;
  unsigned int                            xs_, ys_;
  bool                                    reduce_;
  curaii::unique_device_ptr<float2>       lens_;
  curaii::unique_device_ptr<int>          scales_;
  curaii::unique_device_ptr<unsigned int> intermediate_;
  curaii::unique_device_ptr<float>        partial_;
  curaii::unique_device_ptr<float>        scale_maxima_;
  CUfunction                              scale_partial_{}, scale_{}, x_{}, y_{}, finish_{};
  CUfunction                              function(const char *name) {
    CUfunction f;
    driver(cuModuleGetFunction(&f, module_.value, name));
    return f;
  }
  unsigned int constant(const char *name) {
    CUdeviceptr ptr;
    size_t      size;
    driver(cuModuleGetGlobal(&ptr, &size, module_.value, name));
    unsigned int v;
    driver(cuMemcpyDtoH(&v, ptr, sizeof(v)));
    return v;
  }
  void launch(CUfunction f, unsigned int grid, unsigned int bx, unsigned int by,
              unsigned int shared, void **args) {
    driver(cuLaunchKernel(f, grid, 1, 1, bx, by, 1, shared, stream_, args, nullptr));
  }

public:
  FusedFresnel(const holoflow::core::TDesc &input, const FresnelDiffractionSettings &settings,
               cudaStream_t stream)
      : stream_(stream), planes_(int(input.num_elements() / (320 * 512))),
        components_(int(input.shape[input.rank() - 3])),
        groups_(settings.reduce_magnitude ? (components_ + 3) / 4 : components_),
        output_count_((settings.reduce_magnitude ? planes_ / components_ : planes_) * 320 * 512),
        reduce_(settings.reduce_magnitude),
        lens_(curaii::make_unique_device_ptr<float2>(320 * 512)),
        scales_(curaii::make_unique_device_ptr<int>(planes_ * 2)),
        intermediate_(curaii::make_unique_device_ptr<unsigned int>(input.num_elements())),
        partial_(
            curaii::make_unique_device_ptr<float>((planes_ / components_) * groups_ * 320 * 512)) {
    scale_maxima_ = curaii::make_unique_device_ptr<float>(planes_ * 64);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    auto ptx = compile(prop.major * 100 + prop.minor * 10,
                       input.dtype == holoflow::core::DType::F16, components_, reduce_);
    driver(cuModuleLoadData(&module_.value, ptx.data()));
    scale_partial_ = function("fused_f16_scale_partial");
    scale_         = function("fused_f16_scale");
    x_             = function("fused_f16_fft_x");
    y_             = function("fused_f16_fft_y_mean");
    finish_        = function("fused_f16_finish");
    xs_            = constant("fused_x_shared");
    ys_            = constant("fused_y_shared");
    driver(cuFuncSetAttribute(x_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(xs_)));
    driver(cuFuncSetAttribute(y_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(ys_)));
    auto  lens       = lens_.get();
    float wavelength = settings.lambda, pitch = settings.dx, distance = settings.z;
    void *args[] = {&lens, &wavelength, &pitch, &distance};
    launch(function("fused_f16_lens"), (320 * 512 + 255) / 256, 256, 1, 0, args);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
  ~FusedFresnel() override {
    CUDA_CHECK_NT(cudaStreamSynchronize(stream_));
    graphs_.clear();
  }
  bool supports_cuda_graph() const noexcept override { return true; }
  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx call{ctx.inputs, ctx.outputs, &cancelled, nullptr, nullptr};
    execute(call);
  }
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (stream_ == nullptr)
      return enqueue(ctx);

    const detail::FresnelCudaGraph::Addresses addresses{ctx.inputs[0].data(),
                                                        ctx.outputs[0].data()};
    cudaStreamCaptureStatus                   capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream_, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone)
      return enqueue(ctx);

    const auto graph = std::find_if(graphs_.begin(), graphs_.end(),
                                    [&addresses](const detail::FresnelCudaGraph &candidate) {
                                      return candidate.matches(addresses);
                                    });
    if (graph != graphs_.end()) {
      graph->launch(stream_);
      graphs_.splice(graphs_.begin(), graphs_, graph);
      return holoflow::core::OpResult::Ok;
    }

    const auto result = enqueue(ctx);
    if (result == holoflow::core::OpResult::Ok && graph_capture_enabled_) {
      try {
        graphs_.emplace_front(stream_, addresses, [&]() { (void)enqueue(ctx); });
        if (graphs_.size() > 128)
          graphs_.pop_back();
      } catch (const std::exception &error) {
        graph_capture_enabled_ = false;
        logger()->warn("[FresnelDiffraction] CUDA Graph capture disabled: {}", error.what());
      }
    }
    return result;
  }

  holoflow::core::OpResult enqueue(holoflow::core::SyncCtx &ctx) {
    auto  input        = ctx.inputs[0].data();
    auto  output       = ctx.outputs[0].data();
    auto  lens         = lens_.get();
    auto  scales       = scales_.get();
    auto  intermediate = intermediate_.get();
    auto  partial      = partial_.get();
    auto  maxima       = scale_maxima_.get();
    void *pa[]         = {&input, &lens, &maxima};
    launch(scale_partial_, planes_ * 64, 256, 1, 0, pa);
    void *sa[] = {&maxima, &scales};
    launch(scale_, planes_, 256, 1, 0, sa);
    void *xa[] = {&input, &lens, &scales, &intermediate};
    launch(x_, planes_ * (320 / 8), 64, 4, xs_, xa);
    void *ya[] = {&intermediate, &scales, &partial};
    launch(y_, (planes_ / components_) * groups_ * (512 / 4), 16, 2, ys_, ya);
    void *fa[] = {&partial, &output, &output_count_};
    launch(finish_, (output_count_ + 255) / 256, 256, 1, 0, fa);
    return holoflow::core::OpResult::Ok;
  }
};
} // namespace
bool supports_fused_f16_fresnel(const holoflow::core::TDesc      &input,
                                const FresnelDiffractionSettings &settings) {
  using holoflow::core::DType;
  if (input.rank() < 3 || std::ranges::find(input.shape, size_t{0}) != input.shape.end() ||
      input.shape[input.rank() - 2] != 320 || input.shape.back() != 512 ||
      !settings.output_magnitude || settings.z == 0 ||
      (input.dtype != DType::F16 && input.dtype != DType::F32))
    return false;
  const holoflow::core::TDesc contiguous(input.shape, input.dtype, input.mem_loc, input.offset);
  if (input.strides != contiguous.strides ||
      (!settings.axes.empty() && settings.axes != std::vector<int>{-2, -1}))
    return false;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return (prop.major == 8 && prop.minor == 6) || (prop.major == 12 && prop.minor == 0);
}
std::unique_ptr<holoflow::core::ISyncTask>
create_fused_f16_fresnel(const holoflow::core::TDesc         &input,
                         const FresnelDiffractionSettings    &settings,
                         const holoflow::core::SyncCreateCtx &ctx) {
  return std::make_unique<FusedFresnel>(input, settings, ctx.stream);
}
} // namespace holotask::syncs
