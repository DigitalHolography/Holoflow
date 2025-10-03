#include "fresnel_diffraction.hh"

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks {

void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds) {
  j = nlohmann::json{
      {"lambda", fds.lambda},
      {"dx", fds.dx},
      {"dy", fds.dy},
      {"z", fds.z},
  };
}

void from_json(const nlohmann::json &j, FresnelDiffractionSettings &fds) {
  j.at("lambda").get_to(fds.lambda);
  j.at("dx").get_to(fds.dx);
  j.at("dy").get_to(fds.dy);
  j.at("z").get_to(fds.z);
}

FresnelDiffraction::FresnelDiffraction(const FresnelDiffractionSettings &settings,
                                       curaii::CufftHandle             &&fft_handle,
                                       DevPtr<cuFloatComplex>          &&d_lens,
                                       DevPtr<void> &&d_caller_info, std::vector<char> &&lto)
    : settings_(settings), fft_handle_(std::move(fft_handle)), d_lens_(std::move(d_lens)),
      d_caller_info_(std::move(d_caller_info)), lto_(std::move(lto)) {}

holoflow::core::OpResult FresnelDiffraction::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data);
  CUFFT_CHECK(cufftXtExec(fft_handle_.get(), idata, odata, CUFFT_FORWARD));
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
    logger()->error("[FresnelDiffraction] NVRTC compilation log:\n{}", log);
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

__global__ void quadratic_lens_kernel(cuFloatComplex *lens, int width, int height, float lambda,
                                      float z, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size = width > height ? width : height;

  // The intent with offsets is to support non-square images.
  // They are used to "center" the indexes as if the rectangle was extended to
  // a square.
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
FresnelDiffractionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                  const nlohmann::json                  &jsettings,
                                  const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<FresnelDiffractionSettings>();
  auto &idesc    = input_descs[0];

  const int B = static_cast<int>(idesc.shape[0]);
  const int H = static_cast<int>(idesc.shape[1]);
  const int W = static_cast<int>(idesc.shape[2]);

  // LTO apply lens
  auto                d_lens = make_quadratic_lens(W, H, settings);
  ApplyLensCallerInfo info{W, H, B, d_lens.get()};
  auto                d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));
  auto lto = apply_lens_lto();

  // FFT plan
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

  curaii::CufftHandle fft_handle;
  CUFFT_CHECK(cufftSetStream(fft_handle.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(fft_handle.get(), "apply_lens_callback", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fft_handle.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Success
  auto *task = new FresnelDiffraction(settings, std::move(fft_handle), std::move(d_lens),
                                      std::move(d_info), std::move(lto));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelDiffractionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                  std::span<const holoflow::core::TDesc>     input_descs,
                                  const nlohmann::json                      &jsettings,
                                  const holoflow::core::SyncCreateCtx       &ctx) const {
  // Validate
  auto infer       = this->infer(input_descs, jsettings);
  auto settings    = jsettings.get<FresnelDiffractionSettings>();
  auto old_fresnel = dynamic_cast<FresnelDiffraction *>(old_task.get());
  HOLOVIBES_CHECK(old_fresnel != nullptr, "old_task is not a FresnelDiffraction");
  auto &idesc = input_descs[0];

  const int B = static_cast<int>(idesc.shape[0]);
  const int H = static_cast<int>(idesc.shape[1]);
  const int W = static_cast<int>(idesc.shape[2]);

  // LTO apply lens
  auto                d_lens = make_quadratic_lens(W, H, settings);
  ApplyLensCallerInfo info{W, H, B, d_lens.get()};
  auto                d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));
  auto lto = std::move(old_fresnel->lto_);

  // FFT plan
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

  curaii::CufftHandle fft_handle;
  CUFFT_CHECK(cufftSetStream(fft_handle.get(), ctx.stream));

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(fft_handle.get(), "apply_lens_callback", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(fft_handle.get(), rank, n, inembed, istride, idist, inputtype,
                                  onembed, ostride, odist, outputtype, batch, &work_size,
                                  executiontype));

  // Success
  auto *task = new FresnelDiffraction(settings, std::move(fft_handle), std::move(d_lens),
                                      std::move(d_info), std::move(lto));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks