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

#include "holonp/fft.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuComplex.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"

namespace holonp {

void to_json(nlohmann::json &j, const FFTSettings &s) {
  j = nlohmann::json{
      {"axis", s.axis},
      {"norm", s.norm},
  };
}

void from_json(const nlohmann::json &j, FFTSettings &s) {
  if (j.contains("axis")) {
    j.at("axis").get_to(s.axis);
  } else {
    s.axis = -1;
  }

  if (j.contains("norm")) {
    j.at("norm").get_to(s.norm);
  } else {
    s.norm = FftNorm::Backward;
  }
}

namespace {

constexpr int kMaxNDim = 16;

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("FFT: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0) {
    axis += ndim;
  }
  return axis;
}

inline float norm_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward) {
    return 1.0f;
  }
  const double n = static_cast<double>(n_fft);
  if (norm == FftNorm::Forward) {
    return static_cast<float>(1.0 / n);
  }
  return static_cast<float>(1.0 / std::sqrt(n));
}

struct FFTStoreCallbackInfo {
  float scale = 1.0f;
};

std::string get_compute_arch() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<std::string> get_nvrtc_args() {
  auto cuda_path = std::getenv("CUDA_PATH");
  check(cuda_path != nullptr, "CUDA_PATH environment variable not set");

  return {
      "-I" + std::string{cuda_path} + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };
}

std::vector<char> compile_source_to_lto(const std::string &source, const std::string &name) {
  const auto           args_string = get_nvrtc_args();
  curaii::NvrtcProgram prog(source.c_str(), name.c_str(), 0, nullptr, nullptr);

  std::vector<char *> args;
  args.reserve(args_string.size());
  for (const auto &arg : args_string) {
    args.push_back(const_cast<char *>(arg.c_str()));
  }

  try {
    NVRTC_CHECK(nvrtcCompileProgram(prog.get(), static_cast<int>(args.size()), args.data()));
    size_t code_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog.get(), &code_size));
    std::vector<char> lto(code_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog.get(), lto.data()));
    return lto;
  } catch (const curaii::NvrtcError &e) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog.get(), &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog.get(), log.data()));
    throw std::runtime_error(std::string(e.what()) + "\n" + log);
  }
}

std::vector<char> load_real_as_complex_lto() {
  std::string src = R"(
#include <cuComplex.h>

__device__ cuFloatComplex load_real_as_complex_callback(
    void *data, unsigned long long offset, void *callerInfo, void *sharedPtr) {
  return make_cuComplex(((float *)data)[offset], 0.0f);
}
)";

  return compile_source_to_lto(src, "load_real_as_complex_callback.cu");
}

std::vector<char> store_scaled_complex_lto() {
  std::string src = R"(
#include <cuComplex.h>

struct FFTStoreCallbackInfo {
  float scale;
};

__device__ void store_scaled_complex_callback(
    void *dataOut, unsigned long long offset, cuFloatComplex element, void *callerInfo,
    void *sharedPtr) {
  auto *info = (FFTStoreCallbackInfo *)callerInfo;
  ((cuFloatComplex *)dataOut)[offset] = make_cuComplex(element.x * info->scale,
                                                       element.y * info->scale);
}
)";

  return compile_source_to_lto(src, "store_scaled_complex_callback.cu");
}

class FFT : public holoflow::core::ISyncTask {
public:
  FFT(FFTSettings settings, holoflow::core::TDesc idesc, curaii::CufftHandle &&plan, size_t n_fft,
      size_t exec_count, size_t exec_stride, holoflow::core::DType input_dtype, cudaStream_t stream,
      std::vector<char> load_lto, std::vector<char> store_lto,
      DevPtr<FFTStoreCallbackInfo> d_store_info)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), plan_(std::move(plan)),
        n_fft_(n_fft), exec_count_(exec_count), exec_stride_(exec_stride),
        input_dtype_(input_dtype), stream_(stream), load_lto_(std::move(load_lto)),
        store_lto_(std::move(store_lto)), d_store_info_(std::move(d_store_info)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const FFTSettings           &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_ = stream;
      CUFFT_CHECK(cufftSetStream(plan_.get(), stream_));
    }
  }

private:
  FFTSettings                  settings_;
  holoflow::core::TDesc        idesc_;
  curaii::CufftHandle          plan_;
  size_t                       n_fft_;
  size_t                       exec_count_;
  size_t                       exec_stride_;
  holoflow::core::DType        input_dtype_;
  cudaStream_t                 stream_;
  std::vector<char>            load_lto_;
  std::vector<char>            store_lto_;
  DevPtr<FFTStoreCallbackInfo> d_store_info_;
};

} // namespace

holoflow::core::OpResult FFT::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = ctx.inputs[0].data();
  auto *odata = ctx.outputs[0].data();

  auto *in_f  = reinterpret_cast<float *>(idata);
  auto *in_c  = reinterpret_cast<cuFloatComplex *>(idata);
  auto *out_c = reinterpret_cast<cuFloatComplex *>(odata);

  for (size_t i = 0; i < exec_count_; ++i) {
    const auto offset = i * exec_stride_;
    auto *in_ptr  = input_dtype_ == holoflow::core::DType::F32 ? static_cast<void *>(in_f + offset)
                                                               : static_cast<void *>(in_c + offset);
    auto *out_ptr = out_c + offset;
    CUFFT_CHECK(cufftXtExec(plan_.get(), in_ptr, out_ptr, CUFFT_FORWARD));
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult FFTFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<FFTSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "input dtype must be F32 or CF32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  // holoflow::core::TDesc odesc = idesc;
  // odesc.dtype                 = holoflow::core::DType::CF32;

  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

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
FFTFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<FFTSettings>();

  const auto &idesc = input_descs[0];
  const int   ndim  = static_cast<int>(idesc.shape.size());

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto n_fft = idesc.shape[static_cast<size_t>(axis)];
  size_t     outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= idesc.shape[static_cast<size_t>(i)];
  }
  size_t inner = 1;
  for (int i = axis + 1; i < ndim; ++i) {
    inner *= idesc.shape[static_cast<size_t>(i)];
  }

  check(n_fft > 0, "invalid FFT length");
  check(n_fft <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT length exceeds cuFFT limits");

  const auto kIntMax = static_cast<size_t>(std::numeric_limits<int>::max());
  check(inner <= kIntMax, "FFT stride exceeds cuFFT limits");

  const bool can_loop_inner    = (outer <= kIntMax) && (n_fft <= kIntMax / inner);
  const bool prefer_loop_inner = inner <= outer;

  int    batch_i     = 0;
  int    istride     = static_cast<int>(inner);
  int    idist       = 0;
  size_t exec_count  = 0;
  size_t exec_stride = 0;

  if (can_loop_inner && prefer_loop_inner) {
    batch_i     = static_cast<int>(outer);
    idist       = static_cast<int>(n_fft * inner);
    exec_count  = inner;
    exec_stride = 1;
  } else {
    batch_i     = static_cast<int>(inner);
    idist       = 1;
    exec_count  = outer;
    exec_stride = n_fft * inner;
  }

  int           rank          = 1;
  long long int n[1]          = {static_cast<long long int>(n_fft)};
  long long int inembed[1]    = {static_cast<long long int>(n_fft)};
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[1]    = {static_cast<long long int>(n_fft)};
  int           ostride       = istride;
  int           odist         = idist;
  cudaDataType  outputtype    = CUDA_C_32F;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));

  std::vector<char> load_lto;
  if (idesc.dtype == holoflow::core::DType::F32) {
    load_lto = load_real_as_complex_lto();
    CUFFT_CHECK(cufftXtSetJITCallback(plan.get(), "load_real_as_complex_callback", load_lto.data(),
                                      load_lto.size(), CUFFT_CB_LD_COMPLEX, nullptr));
  }

  std::vector<char>            store_lto;
  DevPtr<FFTStoreCallbackInfo> d_store_info;
  const float                  scale = norm_scale(settings.norm, n_fft);
  if (scale != 1.0f) {
    store_lto    = store_scaled_complex_lto();
    d_store_info = curaii::make_unique_device_ptr<FFTStoreCallbackInfo>(1, ctx.stream);
    FFTStoreCallbackInfo info{
        .scale = scale,
    };
    CUDA_CHECK(cudaMemcpyAsync(d_store_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice,
                               ctx.stream));
    auto *d_store_info_ptr = reinterpret_cast<void *>(d_store_info.get());
    CUFFT_CHECK(cufftXtSetJITCallback(plan.get(), "store_scaled_complex_callback", store_lto.data(),
                                      store_lto.size(), CUFFT_CB_ST_COMPLEX, &d_store_info_ptr));
  }

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), rank, n, inembed, istride, idist, inputtype, onembed,
                                  ostride, odist, outputtype, batch_i, &work_size, executiontype));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFT(settings, idesc, std::move(plan), n_fft, exec_count, exec_stride, idesc.dtype,
              ctx.stream, std::move(load_lto), std::move(store_lto), std::move(d_store_info)));
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_fft = dynamic_cast<FFT *>(old_task.get());
  if (old_fft != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<FFTSettings>();
    if (settings == old_fft->settings() && same_desc(input_descs[0], old_fft->idesc())) {
      old_fft->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
