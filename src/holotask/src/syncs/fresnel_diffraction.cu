#include "holotask/syncs/fresnel_diffraction.hh"

#include <limits>
#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds) {
  j = nlohmann::json{
      {"lambda", fds.lambda}, {"dx", fds.dx}, {"dy", fds.dy}, {"z", fds.z}, {"axes", fds.axes},
  };
}

void from_json(const nlohmann::json &j, FresnelDiffractionSettings &fds) {
  j.at("lambda").get_to(fds.lambda);
  j.at("dx").get_to(fds.dx);
  j.at("dy").get_to(fds.dy);
  j.at("z").get_to(fds.z);
  if (j.contains("axes")) {
    j.at("axes").get_to(fds.axes);
  } else {
    fds.axes = {-2, -1};
  }
}

FresnelDiffraction::FresnelDiffraction(const FresnelDiffractionSettings &settings,
                                       holoflow::core::TDesc             idesc,
                                       curaii::CufftHandle &&fft_handle, bool is_fast, bool is_real,
                                       DevPtr<cuFloatComplex> &&d_lens,
                                       DevPtr<void> &&d_caller_info, std::vector<char> &&lto)
    : settings_(settings), idesc_(std::move(idesc)), fft_handle_(std::move(fft_handle)),
      is_fast_(is_fast), is_real_(is_real), d_lens_(std::move(d_lens)),
      d_caller_info_(std::move(d_caller_info)), lto_(std::move(lto)) {}

holoflow::core::OpResult FresnelDiffraction::execute(holoflow::core::SyncCtx &ctx) {
  void *idata = ctx.inputs[0].data();
  void *odata = ctx.outputs[0].data();
  CUFFT_CHECK(cufftXtExec(fft_handle_.get(), idata, odata, CUFFT_FORWARD));
  return holoflow::core::OpResult::Ok;
}

namespace {

// Explicitly padded to guarantee identical 8-byte alignment between host and NVRTC
struct ApplyLensCallerInfo {
  unsigned int       width;
  unsigned int       _padding;
  unsigned long long idist;
  unsigned long long stride_h;
  unsigned long long istride;
  cuFloatComplex    *lens;
};

std::string get_compute_arch() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<std::string> get_nvrtc_args() {
  auto CUDA_PATH = std::getenv("CUDA_PATH");
  HOLOVIBES_CHECK(CUDA_PATH != nullptr, "CUDA_PATH environment variable not set");
  std::string cuda_path{CUDA_PATH};

  std::vector<std::string> args{
      "-I" + cuda_path + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };
  return args;
}

std::vector<char> compile_source_to_lto(const std::string &source, const std::string &name) {
  auto                 args_string = get_nvrtc_args();
  curaii::NvrtcProgram prog(source.c_str(), name.c_str(), 0, nullptr, nullptr);
  std::vector<char *>  args;
  std::ranges::transform(args_string, std::back_inserter(args),
                         [](const std::string &s) { return const_cast<char *>(s.c_str()); });

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
    logger()->error("[FresnelDiffraction] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

// Dynamically construct to guarantee only ONE identical symbol is exposed to cuFFT
std::vector<char> apply_lens_lto(bool is_fast, bool is_real) {
  std::string apply_lens_callback = R"(
  #include <cuComplex.h>

  struct ApplyLensCallerInfo {
    unsigned int width;
    unsigned int _padding; 
    unsigned long long idist;
    unsigned long long stride_h;
    unsigned long long istride;
    cuFloatComplex *lens;
  };

  __device__ cuFloatComplex apply_lens_callback2(
      void *data, size_t offset, void *callerInfo, void *sharedPtr) {
    auto *info = (ApplyLensCallerInfo *)callerInfo;
  )";

  if (is_real) {
    apply_lens_callback += R"(
    cuFloatComplex val = make_cuComplex(((float *)data)[offset], 0.0f);
    )";
  } else {
    apply_lens_callback += R"(
    cuFloatComplex val = ((cuFloatComplex *)data)[offset];
    )";
  }

  if (is_fast) {
    apply_lens_callback += R"(
    size_t lens_idx = offset % info->idist; 
    return cuCmulf(val, info->lens[lens_idx]);
  }
  )";
  } else {
    apply_lens_callback += R"(
    size_t local_offset = offset % info->idist;
    size_t row = local_offset / info->stride_h;
    size_t col = (local_offset % info->stride_h) / info->istride;
    size_t lens_idx = row * info->width + col;
    return cuCmulf(val, info->lens[lens_idx]);
  }
  )";
  }

  logger()->info("[FresnelDiffraction] Source code for JIT callback:\n{}", apply_lens_callback);

  return compile_source_to_lto(apply_lens_callback, "apply_lens_callback2.cu");
}

__global__ void quadratic_lens_kernel(cuFloatComplex *lens, int width, int height, float lambda,
                                      float z, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size     = width > height ? width : height;
  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * pixel_size;
  float y = ((row + offset_y) - size / 2.0f) * pixel_size;

  float phase     = CUDART_PI_F / (lambda * z) * (x * x + y * y);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

DevPtr<cuFloatComplex> make_quadratic_lens(int W, int H,
                                           const FresnelDiffractionSettings &settings) {
  auto bytes  = W * H * sizeof(cuFloatComplex);
  auto d_lens = curaii::make_unique_device_ptr<cuFloatComplex>(bytes);

  dim3 block_size(16, 16);
  dim3 grid_size((W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y);
  quadratic_lens_kernel<<<grid_size, block_size>>>(d_lens.get(), W, H, settings.lambda, settings.z,
                                                   settings.dx);

  CUDA_CHECK(cudaGetLastError());
  return d_lens;
}

} // namespace

holoflow::core::InferResult
FresnelDiffractionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[FresnelDiffractionFactory::infer] error: {}", msg);
      throw std::invalid_argument("FresnelDiffractionFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<FresnelDiffractionSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  auto &idesc = input_descs[0];
  check(idesc.rank() >= 2, "input must be a tensor of rank 2 or higher");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32,
        "input must be complex float32 or real float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");

  auto axes = settings.axes.empty() ? std::vector<int>{-2, -1} : settings.axes;
  check(axes.size() == 2, "axes must contain exactly two dimensions");
  int rank = static_cast<int>(idesc.rank());
  int ax0  = axes[0] < 0 ? axes[0] + rank : axes[0];
  int ax1  = axes[1] < 0 ? axes[1] + rank : axes[1];
  check(ax0 >= 0 && ax0 < rank && ax1 >= 0 && ax1 < rank, "axes out of bounds");
  check(ax0 != ax1, "axes must be distinct");

  if (!idesc.strides.empty()) {
    size_t stride_w = idesc.strides[ax1];
    size_t stride_h = idesc.strides[ax0];
    check(stride_h % stride_w == 0, "stride of ax0 (H) must be a multiple of stride of ax1 (W) for "
                                    "cuFFT to run without copying");
  }

  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");

  // auto odesc  = idesc;
  // odesc.dtype = holoflow::core::DType::CF32;

  holoflow::core::TDesc odesc;
  if (idesc.dtype == holoflow::core::DType::CF32) {
    odesc = idesc;
  } else {
    odesc = holoflow::core::TDesc(idesc.shape, holoflow::core::DType::CF32, idesc.mem_loc);
  }

  std::vector<holoflow::core::InPlace> in_place;
  if (idesc.dtype == holoflow::core::DType::CF32) {
    in_place = {{0, 0}};
  }

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = in_place,
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelDiffractionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                  const nlohmann::json                  &jsettings,
                                  const holoflow::core::SyncCreateCtx   &ctx) const {
  return update(nullptr, input_descs, jsettings, ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelDiffractionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                  std::span<const holoflow::core::TDesc>     input_descs,
                                  const nlohmann::json                      &jsettings,
                                  const holoflow::core::SyncCreateCtx       &ctx) const {
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<FresnelDiffractionSettings>();
  auto &idesc    = input_descs[0];

  int  rank = static_cast<int>(idesc.rank());
  auto axes = settings.axes.empty() ? std::vector<int>{-2, -1} : settings.axes;
  int  ax0  = axes[0] < 0 ? axes[0] + rank : axes[0];
  int  ax1  = axes[1] < 0 ? axes[1] + rank : axes[1];

  const int H = static_cast<int>(idesc.shape[ax0]);
  const int W = static_cast<int>(idesc.shape[ax1]);

  long long int batch = 1;
  for (int i = 0; i < rank; ++i) {
    if (i != ax0 && i != ax1)
      batch *= static_cast<long long int>(idesc.shape[i]);
  }

  bool          is_real   = (idesc.dtype == holoflow::core::DType::F32);
  long long int istride   = 1;
  long long int inembed_1 = W;
  long long int idist     = H * W;

  if (!idesc.strides.empty()) {
    size_t elem_size = is_real ? sizeof(float) : sizeof(cuFloatComplex);
    istride          = idesc.strides[ax1] / elem_size;
    inembed_1        = (idesc.strides[ax0] / elem_size) / istride;

    if (batch > 1) {
      long long int min_batch_stride = std::numeric_limits<long long int>::max();
      for (int i = 0; i < rank; ++i) {
        if (i != ax0 && i != ax1) {
          long long int s = idesc.strides[i] / elem_size;
          if (s < min_batch_stride)
            min_batch_stride = s;
        }
      }
      idist = min_batch_stride;
    } else {
      idist = inembed_1 * H * istride;
    }
  }

  bool                is_fast = (istride == 1 && inembed_1 == W);
  FresnelDiffraction *task    = nullptr;

  if (old_task) {
    task = dynamic_cast<FresnelDiffraction *>(old_task.release());
    HOLOVIBES_CHECK(task != nullptr, "old_task is not a FresnelDiffraction");
    CUFFT_CHECK(cufftSetStream(task->fft_handle_.get(), ctx.stream));
  }

  bool is_new = (task == nullptr);

  bool shape_changed =
      is_new || (task->idesc_.shape != idesc.shape) || (task->idesc_.strides != idesc.strides);
  bool axes_changed   = is_new || (task->settings_.axes != settings.axes);
  bool optics_changed = is_new || (task->settings_.lambda != settings.lambda) ||
                        (task->settings_.z != settings.z) || (task->settings_.dx != settings.dx);
  bool callback_changed = is_new || (task->is_fast_ != is_fast) || (task->is_real_ != is_real);

  bool lens_size_changed = false;
  if (!is_new) {
    int  old_rank = static_cast<int>(task->idesc_.rank());
    auto old_axes = task->settings_.axes.empty() ? std::vector<int>{-2, -1} : task->settings_.axes;
    int  old_ax0  = old_axes[0] < 0 ? old_axes[0] + old_rank : old_axes[0];
    int  old_ax1  = old_axes[1] < 0 ? old_axes[1] + old_rank : old_axes[1];
    int  old_H    = static_cast<int>(task->idesc_.shape[old_ax0]);
    int  old_W    = static_cast<int>(task->idesc_.shape[old_ax1]);
    lens_size_changed = (old_H != H) || (old_W != W);
  }

  if (is_new) {
    auto                d_lens = make_quadratic_lens(W, H, settings);
    ApplyLensCallerInfo info{static_cast<unsigned int>(W),
                             0,
                             static_cast<unsigned long long>(idist),
                             static_cast<unsigned long long>(inembed_1 * istride),
                             static_cast<unsigned long long>(istride),
                             d_lens.get()};

    auto d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
    CUDA_CHECK(
        cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream));

    auto lto = apply_lens_lto(is_fast, is_real);

    curaii::CufftHandle new_fft_handle;
    CUFFT_CHECK(cufftSetStream(new_fft_handle.get(), ctx.stream));

    int           fft_rank      = 2;
    long long int n[2]          = {static_cast<long long int>(H), static_cast<long long int>(W)};
    long long int inembed[2]    = {static_cast<long long int>(H),
                                   static_cast<long long int>(inembed_1)};
    long long int onembed[2]    = {static_cast<long long int>(H),
                                   static_cast<long long int>(inembed_1)};
    cudaDataType  inputtype     = CUDA_C_32F;
    cudaDataType  outputtype    = CUDA_C_32F;
    size_t        work_size     = 0;
    cudaDataType  executiontype = CUDA_C_32F;

    auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
    CUFFT_CHECK(cufftXtSetJITCallback(new_fft_handle.get(), "apply_lens_callback2", lto.data(),
                                      lto.size(), CUFFT_CB_LD_COMPLEX, &d_info_ptr));

    CUFFT_CHECK(cufftXtMakePlanMany(
        new_fft_handle.get(), fft_rank, n, inembed, static_cast<long long int>(istride),
        static_cast<long long int>(idist), inputtype, onembed, static_cast<long long int>(istride),
        static_cast<long long int>(idist), outputtype, static_cast<long long int>(batch),
        &work_size, executiontype));

    task = new FresnelDiffraction(settings, idesc, std::move(new_fft_handle), is_fast, is_real,
                                  std::move(d_lens), std::move(d_info), std::move(lto));

  } else {
    if (optics_changed || lens_size_changed || axes_changed) {
      task->d_lens_ = make_quadratic_lens(W, H, settings);
    }

    if (shape_changed || axes_changed || optics_changed) {
      ApplyLensCallerInfo info{static_cast<unsigned int>(W),
                               0,
                               static_cast<unsigned long long>(idist),
                               static_cast<unsigned long long>(inembed_1 * istride),
                               static_cast<unsigned long long>(istride),
                               task->d_lens_.get()};
      CUDA_CHECK(cudaMemcpyAsync(task->d_caller_info_.get(), &info, sizeof(info),
                                 cudaMemcpyHostToDevice, ctx.stream));
    }

    if (shape_changed || axes_changed || callback_changed) {
      if (callback_changed) {
        task->lto_ = apply_lens_lto(is_fast, is_real);
      }

      curaii::CufftHandle new_fft_handle;
      CUFFT_CHECK(cufftSetStream(new_fft_handle.get(), ctx.stream));

      int           fft_rank      = 2;
      long long int n[2]          = {static_cast<long long int>(H), static_cast<long long int>(W)};
      long long int inembed[2]    = {static_cast<long long int>(H),
                                     static_cast<long long int>(inembed_1)};
      long long int onembed[2]    = {static_cast<long long int>(H),
                                     static_cast<long long int>(inembed_1)};
      cudaDataType  inputtype     = CUDA_C_32F;
      cudaDataType  outputtype    = CUDA_C_32F;
      size_t        work_size     = 0;
      cudaDataType  executiontype = CUDA_C_32F;

      auto *d_info_ptr = reinterpret_cast<void *>(task->d_caller_info_.get());
      CUFFT_CHECK(cufftXtSetJITCallback(new_fft_handle.get(), "apply_lens_callback2",
                                        task->lto_.data(), task->lto_.size(), CUFFT_CB_LD_COMPLEX,
                                        &d_info_ptr));

      CUFFT_CHECK(cufftXtMakePlanMany(
          new_fft_handle.get(), fft_rank, n, inembed, static_cast<long long int>(istride),
          static_cast<long long int>(idist), inputtype, onembed,
          static_cast<long long int>(istride), static_cast<long long int>(idist), outputtype,
          static_cast<long long int>(batch), &work_size, executiontype));

      task->fft_handle_ = std::move(new_fft_handle);
    }

    task->settings_ = settings;
    task->idesc_    = idesc;
    task->is_fast_  = is_fast;
    task->is_real_  = is_real;
  }

  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs