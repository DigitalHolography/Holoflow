#include "angular_spectrum.hh"

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks::syncs {

void to_json(nlohmann::json &j, const AngularSpectrumSettings::Filter &f) {
  j = nlohmann::json{
      {"r_inner", f.r_inner},
      {"r_outer", f.r_outer},
      {"s_inner", f.s_inner},
      {"s_outer", f.s_outer},
  };
}

void from_json(const nlohmann::json &j, AngularSpectrumSettings::Filter &f) {
  j.at("r_inner").get_to(f.r_inner);
  j.at("r_outer").get_to(f.r_outer);
  j.at("s_inner").get_to(f.s_inner);
  j.at("s_outer").get_to(f.s_outer);
}

void to_json(nlohmann::json &j, const AngularSpectrumSettings &as) {
  j = nlohmann::json{
      {"lambda", as.lambda},
      {"dx", as.dx},
      {"dy", as.dy},
      {"z", as.z},
  };
  if (as.filter.has_value()) {
    j["filter"] = as.filter.value();
  }
}

void from_json(const nlohmann::json &j, AngularSpectrumSettings &as) {
  j.at("lambda").get_to(as.lambda);
  j.at("dx").get_to(as.dx);
  j.at("dy").get_to(as.dy);
  j.at("z").get_to(as.z);
  if (j.contains("filter")) {
    as.filter = AngularSpectrumSettings::Filter{};
    j.at("filter").get_to(as.filter.value());
  } else {
    as.filter = std::nullopt;
  }
}

AngularSpectrum::AngularSpectrum(const AngularSpectrumSettings &settings,
                                 curaii::CufftHandle &&fwd_plan, curaii::CufftHandle &&inv_plan,
                                 DevPtr<cuFloatComplex> &&d_lens, DevPtr<void> &&d_caller_info,
                                 std::vector<char> &&lto)
    : settings_(settings), fwd_plan_(std::move(fwd_plan)), inv_plan_(std::move(inv_plan)),
      d_lens_(std::move(d_lens)), d_caller_info_(std::move(d_caller_info)), lto_(std::move(lto)) {}

holoflow::core::OpResult AngularSpectrum::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data);
  CUFFT_CHECK(cufftXtExec(fwd_plan_.get(), idata, idata, CUFFT_FORWARD));
  CUFFT_CHECK(cufftXtExec(inv_plan_.get(), idata, odata, CUFFT_INVERSE));
  return holoflow::core::OpResult::Ok;
}

namespace {

struct ApplyLensCallerInfo {
  int             width;
  int             height;
  int             batch;
  cuFloatComplex *lens;
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
    logger()->error("[AngularSpectrum] NVRTC compilation log:\n{}", log);
    throw e;
  }
}

std::vector<char> apply_lens_lto() {
  std::string apply_lens_callback = R"(
  #include <cuComplex.h>

  struct ApplyLensCallerInfo {
    int             width;
    int             height;
    int             batch;
    cuFloatComplex *lens;
  };

  __device__ cuFloatComplex apply_lens_callback(
      void *data, size_t offset, void *callerInfo, void *sharedPtr) {
    auto  *info     = (ApplyLensCallerInfo *)callerInfo;
    size_t lens_idx = offset % (info->width * info->height);
    auto  *lens     = info->lens;
    auto   val      = ((cuFloatComplex *)data)[offset];
    auto   lens_val = lens[lens_idx];

    return cuCmulf(val, lens_val);
  }
  )";

  return compile_source_to_lto(apply_lens_callback, "apply_lens_callback.cu");
}

__global__ void spectral_lens_kernel(cuFloatComplex *lens, int width, int height, float lambda,
                                     float z, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  float du = 1.0f / (width * pixel_size);
  float dv = 1.0f / (height * pixel_size);
  float u  = (col - width / 2) * du;
  float v  = (row - height / 2) * dv;

  float tmp = 1.0f - (lambda * lambda) * (u * u + v * v);
  tmp       = fmaxf(tmp, 0.0f);

  float phase     = 2.0f * CUDART_PI_F * z / lambda * sqrt(tmp);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

__global__ void apply_filter_2d_kernel(cuFloatComplex *lens, const uint32_t width,
                                       const uint32_t height, const uint32_t r_inner,
                                       const uint32_t r_outer, const uint32_t smooth_inner,
                                       const uint32_t smooth_outer) {
  const uint32_t x   = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y   = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t idx = y * width + x;
  if (x >= width || y >= height)
    return;

  const float r_x  = static_cast<float>(x) - static_cast<float>(width) / 2;
  const float r_y  = static_cast<float>(y) - static_cast<float>(height) / 2;
  const float dist = hypotf(r_x, r_y);

  float a = 0.0f;
  if (dist < static_cast<float>(r_outer))
    a = 1.0f;
  else if (dist < static_cast<float>(r_outer + smooth_outer))
    a = cosf((dist - static_cast<float>(r_outer)) / static_cast<float>(smooth_outer) * CUDART_PI_F /
             2);

  float b = 0.0f;
  if (dist < static_cast<float>(r_inner))
    b = 1.0f;
  else if (dist < static_cast<float>(r_inner + smooth_inner))
    b = cosf((dist - static_cast<float>(r_inner)) / static_cast<float>(smooth_inner) * CUDART_PI_F /
             2);

  lens[idx].x *= a * (1 - b);
  lens[idx].y *= a * (1 - b);
}

__global__ void swap_corners_kernel(cuFloatComplex *in, cuFloatComplex *out, int width, int height,
                                    int batch_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int width_half  = width / 2;
  int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch_size) {
    return;
  }

  int             batch_offset = z * width * height;
  cuFloatComplex *in_frame     = in + batch_offset;
  cuFloatComplex *out_frame    = out + batch_offset;

  // --- Swap top-left with bottom-right ---
  int top_left_idx     = x + y * width;
  int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  cuFloatComplex tmp          = in_frame[top_left_idx];
  out_frame[top_left_idx]     = in_frame[bottom_right_idx];
  out_frame[bottom_right_idx] = tmp;

  // --- Swap top-right with bottom-left ---
  int top_right_idx   = (x + width_half) + y * width;
  int bottom_left_idx = x + (y + height_half) * width;

  tmp                        = in_frame[top_right_idx];
  out_frame[top_right_idx]   = in_frame[bottom_left_idx];
  out_frame[bottom_left_idx] = tmp;
}

DevPtr<cuFloatComplex> make_spectral_lens(int W, int H, const AngularSpectrumSettings &settings) {
  auto bytes  = W * H * sizeof(cuFloatComplex);
  auto d_lens = curaii::make_unique_device_ptr<cuFloatComplex>(bytes);

  dim3 block_size(16, 16);
  dim3 grid_size((W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y);
  spectral_lens_kernel<<<grid_size, block_size>>>(d_lens.get(), W, H, settings.lambda, settings.z,
                                                  settings.dx);

  if (settings.filter.has_value()) {
    auto &f = settings.filter.value();
    apply_filter_2d_kernel<<<grid_size, block_size>>>(d_lens.get(), W, H, f.r_inner, f.r_outer,
                                                      f.s_inner, f.s_outer);
  }

  swap_corners_kernel<<<grid_size, block_size>>>(d_lens.get(), d_lens.get(), W, H, 1);
  CUDA_CHECK(cudaPeekAtLastError());
  return d_lens;
}

} // namespace

holoflow::core::InferResult
AngularSpectrumFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[AngularSpectrumFactory::infer] error: {}", msg);
      throw std::invalid_argument("AngularSpectrumFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<AngularSpectrumSettings>();

  // Validate
  check(input_descs.size() == 1, "expected exactly one input");
  auto &idesc = input_descs[0];
  check(idesc.rank() == 3, "input must be a 3D tensor");
  check(idesc.dtype == holoflow::core::DType::CF32, "input must be complex float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");

  // Success
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AngularSpectrumFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<AngularSpectrumSettings>();
  auto &idesc    = input_descs[0];

  const int B = static_cast<int>(idesc.shape[0]);
  const int H = static_cast<int>(idesc.shape[1]);
  const int W = static_cast<int>(idesc.shape[2]);

  // LTO apply lens
  auto                d_lens = make_spectral_lens(W, H, settings);
  ApplyLensCallerInfo info{W, H, B, d_lens.get()};
  auto                d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));
  auto lto = apply_lens_lto();

  // FFT plans
  int           rank          = 2;
  long long int n[2]          = {H, W};
  long long int inembed[2]    = {H, W};
  int           istride       = 1;
  int           idist         = H * W;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {H, W};
  int           ostride       = 1;
  int           odist         = H * W;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = B;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle fwd_plan;
  curaii::CufftHandle inv_plan;
  CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "apply_lens_callback", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Success
  auto *task = new AngularSpectrum(settings, std::move(fwd_plan), std::move(inv_plan),
                                   std::move(d_lens), std::move(d_info), std::move(lto));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

std::unique_ptr<holoflow::core::ISyncTask>
AngularSpectrumFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                               std::span<const holoflow::core::TDesc>     input_descs,
                               const nlohmann::json                      &jsettings,
                               const holoflow::core::SyncCreateCtx       &ctx) const {
  // Validate
  auto infer       = this->infer(input_descs, jsettings);
  auto settings    = jsettings.get<AngularSpectrumSettings>();
  auto old_angular = dynamic_cast<AngularSpectrum *>(old_task.get());
  HOLOVIBES_CHECK(old_angular != nullptr, "old_task is not an AngularSpectrum");
  auto &idesc = input_descs[0];

  const int B = static_cast<int>(idesc.shape[0]);
  const int H = static_cast<int>(idesc.shape[1]);
  const int W = static_cast<int>(idesc.shape[2]);

  // LTO apply lens
  auto                d_lens = make_spectral_lens(W, H, settings);
  ApplyLensCallerInfo info{W, H, B, d_lens.get()};
  auto                d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));
  auto lto = std::move(old_angular->lto_);

  // FFT plans
  int           rank          = 2;
  long long int n[2]          = {H, W};
  long long int inembed[2]    = {H, W};
  int           istride       = 1;
  int           idist         = H * W;
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {H, W};
  int           ostride       = 1;
  int           odist         = H * W;
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch         = B;
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle fwd_plan;
  curaii::CufftHandle inv_plan;
  CUFFT_CHECK(cufftSetStream(fwd_plan.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(inv_plan.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(inv_plan.get(), "apply_lens_callback", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fwd_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  CUFFT_CHECK(cufftXtMakePlanMany(inv_plan.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Success
  auto *task = new AngularSpectrum(settings, std::move(fwd_plan), std::move(inv_plan),
                                   std::move(d_lens), std::move(d_info), std::move(lto));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::syncs
