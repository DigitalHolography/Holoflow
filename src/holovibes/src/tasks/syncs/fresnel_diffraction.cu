#include "fresnel_diffraction.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks::syncs {

// -------------------------------------------------------------------------------------------------
// JSON Serialization
// -------------------------------------------------------------------------------------------------

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

// -------------------------------------------------------------------------------------------------
// FresnelDiffraction Implementation
// -------------------------------------------------------------------------------------------------

FresnelDiffraction::FresnelDiffraction(const FresnelDiffractionSettings &settings,
                                       curaii::CufftHandle             &&fft_handle,
                                       DevPtr<cuFloatComplex>          &&d_lens,
                                       DevPtr<void>                    &&d_caller_info,
                                       std::vector<char>               &&lto) :
    settings_(settings),
    fft_handle_(std::move(fft_handle)),
    d_lens_(std::move(d_lens)),
    d_caller_info_(std::move(d_caller_info)),
    lto_(std::move(lto)) {}

holoflow::core::OpResult FresnelDiffraction::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data);
  CUFFT_CHECK(cufftXtExec(fft_handle_.get(), idata, odata, CUFFT_FORWARD));
  return holoflow::core::OpResult::Ok;
}

namespace {

// -------------------------------------------------------------------------------------------------
// NVRTC / LTO Helpers
// -------------------------------------------------------------------------------------------------

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
  const char *cuda_path_env = std::getenv("CUDA_PATH");
  HOLOVIBES_CHECK(cuda_path_env != nullptr, "CUDA_PATH environment variable not set");

  std::string include_path = std::string("-I") + cuda_path_env + "/include";
  std::string arch_flag    = "-arch=" + get_compute_arch();

  return {std::move(include_path),
          std::move(arch_flag),
          "--std=c++20",
          "--relocatable-device-code=true",
          "-default-device",
          "-dlto"};
}

std::vector<char> compile_source_to_lto(const std::string &source, const std::string &name) {
  auto args_string = get_nvrtc_args();

  std::vector<const char *> args;
  args.reserve(args_string.size());
  for (const auto &s : args_string) {
    args.push_back(s.c_str());
  }

  curaii::NvrtcProgram prog(source.c_str(), name.c_str(), 0, nullptr, nullptr);

  try {
    NVRTC_CHECK(nvrtcCompileProgram(prog.get(), static_cast<int>(args.size()), args.data()));

    size_t code_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog.get(), &code_size));

    std::vector<char> lto(code_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog.get(), lto.data()));
    return lto;
  }

  catch (const curaii::NvrtcError &) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog.get(), &log_size));

    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog.get(), log.data()));

    logger()->error("[FresnelDiffraction] NVRTC compilation log:\n{}", log);
    throw;
  }
}

std::vector<char> generate_apply_lens_lto() {
  // Callback applied "inside" the FFT (quad in)
  constexpr const char *apply_lens_callback_src = R"(
    #include <cuComplex.h>

    struct ApplyLensCallerInfo {
      int             width;
      int             height;
      int             batch;
      cuFloatComplex *lens;
    };

    __device__ cuFloatComplex apply_lens_callback(void  *data,
                                                  size_t offset,
                                                  void  *callerInfo,
                                                  void  *sharedPtr) {
      const auto *info = (const ApplyLensCallerInfo *)callerInfo;

      const size_t plane_size = (size_t)info->width * info->height;
      const size_t lens_idx   = offset % plane_size;

      const cuFloatComplex val      = ((const cuFloatComplex *)data)[offset];
      const cuFloatComplex lens_val = info->lens[lens_idx];

      return cuCmulf(val, lens_val);
    }
  )";

  return compile_source_to_lto(apply_lens_callback_src, "apply_lens_callback.cu");
}

// -------------------------------------------------------------------------------------------------
// Quadratic Lens Generation
// -------------------------------------------------------------------------------------------------

/**
 * @brief Generates the quadratic phase lens on the Host CPU.
 * Formula: exp(i * (pi / (lambda * z)) * (x^2 + y^2))
 */
std::vector<cuFloatComplex>
generate_quadratic_lens_host(int width, int height, const FresnelDiffractionSettings &settings) {
  std::vector<cuFloatComplex> lens(width * height);

  // Constant factor: pi / (lambda * z)
  const float factor = std::numbers::pi_v<float> / (settings.lambda * settings.z);

  const float half_w = width * 0.5f;
  const float half_h = height * 0.5f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Physical coordinates centered at 0
      const float phys_y = (y - half_h) * settings.dy;
      const float y_sq   = phys_y * phys_y;
      const float phys_x = (x - half_w) * settings.dx;
      const float x_sq   = phys_x * phys_x;

      // Calculate complex exponential: cos(phase) + i*sin(phase)
      const float phase   = factor * (x_sq + y_sq);
      lens[y * width + x] = make_cuComplex(std::cos(phase), std::sin(phase));
    }
  }

  return lens;
}

// -------------------------------------------------------------------------------------------------
// FFT Helpers
// -------------------------------------------------------------------------------------------------

struct FftPlanParams {
  int           rank;
  long long int n[2];
  long long int inembed[2];
  int           istride;
  int           idist;
  long long int onembed[2];
  int           ostride;
  int           odist;
  int           batch;
};

curaii::CufftHandle create_configured_fft_handle(const FftPlanParams         &params,
                                                 const std::vector<char>     &lto_code,
                                                 DevPtr<ApplyLensCallerInfo> &d_caller_info_ptr,
                                                 cudaStream_t                 stream) {
  curaii::CufftHandle handle;
  CUFFT_CHECK(cufftSetStream(handle.get(), stream));

  void *info_ptr = d_caller_info_ptr.get();
  CUFFT_CHECK(cufftXtSetJITCallback(handle.get(),
                                    "apply_lens_callback",
                                    const_cast<char *>(lto_code.data()),
                                    lto_code.size(),
                                    CUFFT_CB_LD_COMPLEX,
                                    &info_ptr));

  size_t     work_size   = 0;
  long long *n_ptr       = const_cast<long long *>(params.n);
  long long *inembed_ptr = const_cast<long long *>(params.inembed);
  long long *onembed_ptr = const_cast<long long *>(params.onembed);

  CUFFT_CHECK(cufftXtMakePlanMany(handle.get(),
                                  params.rank,
                                  n_ptr,
                                  inembed_ptr,
                                  params.istride,
                                  params.idist,
                                  CUDA_C_32F,
                                  onembed_ptr,
                                  params.ostride,
                                  params.odist,
                                  CUDA_C_32F,
                                  params.batch,
                                  &work_size,
                                  CUDA_C_32F));

  return handle;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// FresnelDiffractionFactory Implementation
// -------------------------------------------------------------------------------------------------

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
  check(input_descs.size() == 1, "Expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() == 3, "Input must be a 3D tensor (Batch, Height, Width)");
  check(idesc.dtype == holoflow::core::DType::CF32, "Input must be complex float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input must be in device memory");
  check(settings.lambda > 0.0f, "Wavelength (lambda) must be positive");
  check(settings.dx > 0.0f && settings.dy > 0.0f, "Pixel pitch (dx, dy) must be positive");
  constexpr float epsilon = 1e-6f;
  check(std::abs(settings.dx - settings.dy) < epsilon, "dx must equal dy (square pixels required)");

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

  // Generate lens
  auto lens_bytes = W * H * sizeof(cuFloatComplex);
  auto h_lens     = generate_quadratic_lens_host(W, H, settings);
  auto d_lens     = curaii::make_unique_device_ptr<cuFloatComplex>(lens_bytes);
  CUDA_CHECK(cudaMemcpy(d_lens.get(), h_lens.data(), lens_bytes, cudaMemcpyHostToDevice));

  // Prepare caller info
  using Info    = ApplyLensCallerInfo;
  auto lto_code = generate_apply_lens_lto();
  Info info{W, H, B, d_lens.get()};
  auto d_info = curaii::make_unique_device_ptr<Info>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

  // Prepare FFT plan
  FftPlanParams params{.rank    = 2,
                       .n       = {H, W},
                       .inembed = {H, W},
                       .istride = 1,
                       .idist   = H * W,
                       .onembed = {H, W},
                       .ostride = 1,
                       .odist   = H * W,
                       .batch   = B};

  auto fft_handle = create_configured_fft_handle(params, lto_code, d_info, ctx.stream);

  // Success
  auto *task = new FresnelDiffraction(settings,
                                      std::move(fft_handle),
                                      std::move(d_lens),
                                      std::move(d_info),
                                      std::move(lto_code));
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

  // Recreate lens
  auto lens_bytes = W * H * sizeof(cuFloatComplex);
  auto h_lens     = generate_quadratic_lens_host(W, H, settings);
  auto d_lens     = curaii::make_unique_device_ptr<cuFloatComplex>(lens_bytes);
  CUDA_CHECK(cudaMemcpy(d_lens.get(), h_lens.data(), lens_bytes, cudaMemcpyHostToDevice));

  // Recreate caller info (reuse LTO)
  using Info    = ApplyLensCallerInfo;
  auto lto_code = old_fresnel->lto_;
  Info info{W, H, B, d_lens.get()};
  auto d_info = curaii::make_unique_device_ptr<Info>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

  // Recreate FFT plan
  FftPlanParams params{.rank    = 2,
                       .n       = {H, W},
                       .inembed = {H, W},
                       .istride = 1,
                       .idist   = H * W,
                       .onembed = {H, W},
                       .ostride = 1,
                       .odist   = H * W,
                       .batch   = B};

  auto fft_handle = create_configured_fft_handle(params, lto_code, d_info, ctx.stream);

  // Success
  auto *task = new FresnelDiffraction(settings,
                                      std::move(fft_handle),
                                      std::move(d_lens),
                                      std::move(d_info),
                                      std::move(lto_code));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::syncs