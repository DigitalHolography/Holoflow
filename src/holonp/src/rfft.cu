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

#include "holonp/rfft.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"

namespace holonp {

void to_json(nlohmann::json &j, const RFFTSettings &s) {
  j = nlohmann::json{
      {"axis", s.axis},
      {"norm", s.norm},
  };
}

void from_json(const nlohmann::json &j, RFFTSettings &s) {
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
    throw std::invalid_argument("RFFT: " + msg);
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

struct RFFTLoadCallbackInfo {
  const std::uint8_t *input                   = nullptr;
  const void         *dummy_input             = nullptr;
  unsigned long long  input_exec_stride       = 0;
  unsigned long long  dummy_exec_stride_bytes = 0;
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

std::vector<char> load_u8_as_real_lto() {
  std::string src = R"(
struct RFFTLoadCallbackInfo {
  const unsigned char *input;
  const void *dummy_input;
  unsigned long long input_exec_stride;
  unsigned long long dummy_exec_stride_bytes;
};

__device__ float load_u8_as_real_callback(
    void *data, unsigned long long offset, void *callerInfo, void *sharedPtr) {
  const auto *info = (const RFFTLoadCallbackInfo *)callerInfo;
  const auto dummy_offset =
      (unsigned long long)((const unsigned char *)data -
                           (const unsigned char *)info->dummy_input);
  const auto exec = dummy_offset / info->dummy_exec_stride_bytes;
  return (float)info->input[exec * info->input_exec_stride + offset];
}
)";

  return compile_source_to_lto(src, "load_u8_as_real_callback.cu");
}

__global__ void scale_cf32_kernel(cuFloatComplex *__restrict__ data, std::int64_t n, float scale) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

class RFFT : public holoflow::core::ISyncTask {
public:
  RFFT(RFFTSettings settings, holoflow::core::TDesc idesc, curaii::CufftHandle &&plan,
       size_t total_out, size_t n_fft, size_t exec_count, size_t exec_in_stride,
       size_t exec_out_stride, holoflow::core::DType input_dtype, cudaStream_t stream,
       std::vector<char> load_lto, DevPtr<RFFTLoadCallbackInfo> d_load_info)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), plan_(std::move(plan)),
        total_out_(total_out), n_fft_(n_fft), exec_count_(exec_count),
        exec_in_stride_(exec_in_stride), exec_out_stride_(exec_out_stride),
        input_dtype_(input_dtype), stream_(stream), load_lto_(std::move(load_lto)),
        d_load_info_(std::move(d_load_info)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const RFFTSettings          &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_          = stream;
      callback_input_  = nullptr;
      callback_output_ = nullptr;
      CUFFT_CHECK(cufftSetStream(plan_.get(), stream_));
    }
  }

private:
  RFFTSettings                 settings_;
  holoflow::core::TDesc        idesc_;
  curaii::CufftHandle          plan_;
  size_t                       total_out_;
  size_t                       n_fft_;
  size_t                       exec_count_;
  size_t                       exec_in_stride_;
  size_t                       exec_out_stride_;
  holoflow::core::DType        input_dtype_;
  cudaStream_t                 stream_;
  std::vector<char>            load_lto_;
  DevPtr<RFFTLoadCallbackInfo> d_load_info_;
  const void                  *callback_input_{nullptr};
  const void                  *callback_output_{nullptr};
};

} // namespace

holoflow::core::OpResult RFFT::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = ctx.inputs[0].data();
  auto *odata = ctx.outputs[0].data();

  auto *in_f  = reinterpret_cast<float *>(idata);
  auto *in_u8 = reinterpret_cast<std::uint8_t *>(idata);
  auto *out   = reinterpret_cast<cuFloatComplex *>(odata);

  // cuFFT validates the input allocation against the plan's F32 type before invoking the load
  // callback. The CF32 output is large enough to serve as a validated dummy input; callerInfo
  // redirects callback reads to the actual U8 source. Stable graph buffers avoid recurring copies.
  if (input_dtype_ == holoflow::core::DType::U8 &&
      (idata != callback_input_ || odata != callback_output_)) {
    const RFFTLoadCallbackInfo info{
        .input                   = in_u8,
        .dummy_input             = out,
        .input_exec_stride       = exec_in_stride_,
        .dummy_exec_stride_bytes = exec_out_stride_ * sizeof(cuFloatComplex),
    };
    CUDA_CHECK(
        cudaMemcpyAsync(d_load_info_.get(), &info, sizeof(info), cudaMemcpyHostToDevice, stream_));
    callback_input_  = idata;
    callback_output_ = odata;
  }

  for (size_t i = 0; i < exec_count_; ++i) {
    auto *out_ptr = out + i * exec_out_stride_;
    void *in_ptr  = input_dtype_ == holoflow::core::DType::U8
                        ? static_cast<void *>(out_ptr)
                        : static_cast<void *>(in_f + i * exec_in_stride_);
    CUFFT_CHECK(cufftXtExec(plan_.get(), in_ptr, out_ptr, CUFFT_FORWARD));
  }

  const float scale = norm_scale(settings_.norm, n_fft_);
  if (scale != 1.0f) {
    const auto    total_i64 = static_cast<std::int64_t>(total_out_);
    constexpr int block     = 256;
    const int     grid      = static_cast<int>((total_i64 + block - 1) / block);
    scale_cf32_kernel<<<grid, block, 0, stream_>>>(out, total_i64, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult RFFTFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<RFFTSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::F32,
        "input dtype must be U8 or F32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  const auto n_fft = idesc.shape[static_cast<size_t>(axis)];
  check(n_fft > 0, "invalid FFT length");

  // holoflow::core::TDesc odesc            = idesc;
  // odesc.dtype                            = holoflow::core::DType::CF32;
  // odesc.shape[static_cast<size_t>(axis)] = n_fft / 2 + 1;
  //

  auto odesc_shape                       = idesc.shape;
  odesc_shape[static_cast<size_t>(axis)] = n_fft / 2 + 1;
  holoflow::core::TDesc odesc(odesc_shape, holoflow::core::DType::CF32,
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
RFFTFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<RFFTSettings>();

  const auto &idesc    = input_descs[0];
  const auto  total_in = product_shape(idesc.shape);
  const int   ndim     = static_cast<int>(idesc.shape.size());

  const int axis = normalize_axis(settings.axis, ndim);
  check(axis >= 0 && axis < ndim, "axis out of range");

  const auto n_fft = idesc.shape[static_cast<size_t>(axis)];
  const auto n_out = n_fft / 2 + 1;
  check(n_fft > 0, "invalid FFT length");
  check(n_fft <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT length exceeds cuFFT limits");

  size_t outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= idesc.shape[static_cast<size_t>(i)];
  }
  size_t inner = 1;
  for (int i = axis + 1; i < ndim; ++i) {
    inner *= idesc.shape[static_cast<size_t>(i)];
  }

  const auto kIntMax = static_cast<size_t>(std::numeric_limits<int>::max());
  check(inner > 0, "invalid inner size");
  check(inner <= kIntMax, "FFT stride exceeds cuFFT limits");
  check(n_out <= kIntMax, "RFFT output length exceeds cuFFT limits");

  const bool can_loop_inner =
      (outer <= kIntMax) && (n_fft <= kIntMax / inner) && (n_out <= kIntMax / inner);
  const bool prefer_loop_inner = inner <= outer;

  int    batch_i         = 0;
  int    istride         = static_cast<int>(inner);
  int    idist           = 0;
  int    ostride         = static_cast<int>(inner);
  int    odist           = 0;
  size_t exec_count      = 0;
  size_t exec_in_stride  = 0;
  size_t exec_out_stride = 0;

  if (can_loop_inner && prefer_loop_inner) {
    batch_i         = static_cast<int>(outer);
    idist           = static_cast<int>(n_fft * inner);
    odist           = static_cast<int>(n_out * inner);
    exec_count      = inner;
    exec_in_stride  = 1;
    exec_out_stride = 1;
  } else {
    check(inner <= kIntMax, "FFT batch exceeds cuFFT limits");
    batch_i         = static_cast<int>(inner);
    idist           = 1;
    odist           = 1;
    exec_count      = outer;
    exec_in_stride  = n_fft * inner;
    exec_out_stride = n_out * inner;
  }

  int    rank       = 1;
  int    n[1]       = {static_cast<int>(n_fft)};
  int    inembed[1] = {static_cast<int>(n_fft)};
  int    onembed[1] = {static_cast<int>(n_out)};
  size_t work_size  = 0;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));

  std::vector<char>            load_lto;
  DevPtr<RFFTLoadCallbackInfo> d_load_info;
  if (idesc.dtype == holoflow::core::DType::U8) {
    load_lto              = load_u8_as_real_lto();
    d_load_info           = curaii::make_unique_device_ptr<RFFTLoadCallbackInfo>(1, ctx.stream);
    auto *d_load_info_ptr = reinterpret_cast<void *>(d_load_info.get());
    CUFFT_CHECK(cufftXtSetJITCallback(plan.get(), "load_u8_as_real_callback", load_lto.data(),
                                      load_lto.size(), CUFFT_CB_LD_REAL, &d_load_info_ptr));
  }

  CUFFT_CHECK(cufftMakePlanMany(plan.get(), rank, n, inembed, istride, idist, onembed, ostride,
                                odist, CUFFT_R2C, batch_i, &work_size));

  const size_t total_out = (total_in / n_fft) * n_out;

  return std::unique_ptr<holoflow::core::ISyncTask>(new RFFT(
      settings, idesc, std::move(plan), total_out, n_fft, exec_count, exec_in_stride,
      exec_out_stride, idesc.dtype, ctx.stream, std::move(load_lto), std::move(d_load_info)));
}

std::unique_ptr<holoflow::core::ISyncTask>
RFFTFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_rfft = dynamic_cast<RFFT *>(old_task.get());
  if (old_rfft != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<RFFTSettings>();
    if (settings == old_rfft->settings() && same_desc(input_descs[0], old_rfft->idesc())) {
      old_rfft->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
