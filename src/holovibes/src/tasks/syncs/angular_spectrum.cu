#include "angular_spectrum.hh"

#include <cmath>
#include <math_constants.h>
#include <numbers>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks::syncs {

// -------------------------------------------------------------------------------------------------
// JSON Serialization
// -------------------------------------------------------------------------------------------------

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

// -------------------------------------------------------------------------------------------------
// AngularSpectrum Implementation
// -------------------------------------------------------------------------------------------------

AngularSpectrum::AngularSpectrum(const AngularSpectrumSettings &settings,
                                 curaii::CufftHandle          &&fwd_plan,
                                 curaii::CufftHandle          &&inv_plan,
                                 DevPtr<cuFloatComplex>       &&d_lens,
                                 DevPtr<void>                 &&d_caller_info,
                                 std::vector<char>            &&lto) :
    settings_(settings),
    fwd_plan_(std::move(fwd_plan)),
    inv_plan_(std::move(inv_plan)),
    d_lens_(std::move(d_lens)),
    d_caller_info_(std::move(d_caller_info)),
    lto_(std::move(lto)) {}

holoflow::core::OpResult AngularSpectrum::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = reinterpret_cast<cuFloatComplex *>(ctx.inputs[0].data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data);
  CUFFT_CHECK(cufftXtExec(fwd_plan_.get(), idata, idata, CUFFT_FORWARD));
  CUFFT_CHECK(cufftXtExec(inv_plan_.get(), idata, odata, CUFFT_INVERSE));
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

    logger()->error("[AngularSpectrum] NVRTC compilation log:\n{}", log);
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

/// @brief Computes the frequencies of the discrete Fourier transform.
/// @details Generates a vector of frequencies for a given sample spacing,
/// useful for interpreting FFT results in physical units.
/// @param n The window length or number of samples.
/// @param d The sample spacing (inverse of the sampling frequency).
/// @return A vector of length n containing the frequencies in FFT order.
std::vector<float> fftfreq(size_t n, float d) {
  std::vector<float>   freqs(n);
  const float          factor = 1.0f / (static_cast<float>(n) * d);
  const std::ptrdiff_t n_half = static_cast<std::ptrdiff_t>(n) / 2;

  for (size_t i = 0; i < n; ++i) {
    std::ptrdiff_t k   = static_cast<std::ptrdiff_t>(i);
    std::ptrdiff_t idx = (k <= n_half) ? k : (k - static_cast<std::ptrdiff_t>(n));
    freqs[i]           = static_cast<float>(idx) * factor;
  }

  return freqs;
}

/**
 * @brief Performs a 2D FFT shift operation on complex data.
 * 
 * Rearranges the zero-frequency component to the center of the spectrum,
 * swapping the quadrants of the 2D array. This is typically used after
 * FFT computation to center the DC component for visualization or further processing.
 * 
 * @param data Pointer to the input/output cuFloatComplex array in device memory.
 *             The array will be modified in-place.
 * @param width The width (number of columns) of the 2D data array.
 * @param height The height (number of rows) of the 2D data array.
 * 
 * @note This function operates on CUDA device memory. Ensure that the data pointer
 *       is allocated on the GPU and that width*height represents the total number
 *       of complex elements in the array.
 * 
 * @see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Shift_theorem
 */
void fftshift2d(cuFloatComplex *data, size_t width, size_t height) {
  const size_t hx = width / 2;
  const size_t hy = height / 2;

  // For odd sizes, leave the center row/column untouched.
  const size_t wA = hx;
  const size_t hA = hy;

  // Quadrant order before shift:
  // A B
  // C D
  //
  // After shift:
  // D C
  // B A

  // Swap A <-> D
  for (size_t y = 0; y < hA; ++y) {
    size_t rowA = y * width;
    size_t rowD = (y + hy) * width;
    for (size_t x = 0; x < wA; ++x) {
      std::swap(data[rowA + x], data[rowD + (x + hx)]);
    }
  }

  // Swap B <-> C
  for (size_t y = 0; y < hA; ++y) {
    size_t rowB = y * width;
    size_t rowC = (y + hy) * width;
    for (size_t x = 0; x < hx && x + hx < width; ++x) {
      std::swap(data[rowB + (x + hx)], data[rowC + x]);
    }
  }
}

/**
 * @brief Generates a lens transfer function on the host for angular spectrum propagation.
 *
 * Angular spectrum propagation equation:
 * H(fx, fy, z) = exp(i * kz * z)
 * where kz = sqrt(k^2 - (2*pi*fx)^2 - (2*pi*fy)^2)
 * and k = 2*pi/lambda
 *
 * @param width The width of the lens function output (in pixels).
 * @param height The height of the lens function output (in pixels).
 * @param settings Reference to AngularSpectrumSettings containing lens parameters
 *                 such as focal length, wavelength, and pixel size.
 *
 * @return A host-allocated array containing the complex-valued lens transfer function
 *         with dimensions [height][width].
 *
 * @note This function allocates memory on the host that should be freed by the caller.
 * @note The lens function is computed based on the angular spectrum method for
 *       optical wave propagation and focusing.
 */
std::vector<cuFloatComplex>
generate_lens_host(int width, int height, const AngularSpectrumSettings &settings) {
  std::vector<cuFloatComplex> lens(width * height);

  const float k  = 2.0f * std::numbers::pi_v<float> / settings.lambda;
  const auto  fx = fftfreq(static_cast<size_t>(width), settings.dx);
  const auto  fy = fftfreq(static_cast<size_t>(height), settings.dy);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float u = fx[x];
      const float v = fy[y];

      const float kx    = u * 2.0f * std::numbers::pi_v<float>;
      const float ky    = v * 2.0f * std::numbers::pi_v<float>;
      const float kz_sq = k * k - kx * kx - ky * ky;
      const float kz    = (kz_sq > 0.0f) ? std::sqrt(kz_sq) : 0.0f;
      const float phase = kz * settings.z;

      // Calculate complex exponential: cos(phase) + i*sin(phase)
      lens[y * width + x] = make_cuComplex(std::cos(phase), std::sin(phase));
    }
  }

  return lens;
}

void generate_filter(cuFloatComplex *filter,
                     uint32_t        width,
                     uint32_t        height,
                     uint32_t        r_inner,
                     uint32_t        r_outer,
                     uint32_t        smooth_inner,
                     uint32_t        smooth_outer) {
  const float cx = static_cast<float>(width) / 2.0f;
  const float cy = static_cast<float>(height) / 2.0f;

  const float f_r_inner      = static_cast<float>(r_inner);
  const float f_r_outer      = static_cast<float>(r_outer);
  const float f_smooth_inner = static_cast<float>(smooth_inner);
  const float f_smooth_outer = static_cast<float>(smooth_outer);

  for (uint32_t y = 0; y < height; ++y) {
    const float ry = static_cast<float>(y) - cy;
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t idx = y * width + x;

      const float rx   = static_cast<float>(x) - cx;
      const float dist = std::hypot(rx, ry);

      float a = 0.0f;
      if (dist < f_r_outer) {
        a = 1.0f;
      } else if (dist < f_r_outer + f_smooth_outer) {
        a = std::cos((dist - f_r_outer) / f_smooth_outer * std::numbers::pi_v<float> / 2.0f);
      }

      float b = 0.0f;
      if (dist < f_r_inner) {
        b = 1.0f;
      } else if (dist < f_r_inner + f_smooth_inner) {
        b = std::cos((dist - f_r_inner) / f_smooth_inner * std::numbers::pi_v<float> / 2.0f);
      }

      const float v = a * (1.0f - b);
      filter[idx].x = v;
      filter[idx].y = v;
    }
  }
}

std::vector<cuFloatComplex>
generate_lens_with_filter(size_t Nx, size_t Ny, const AngularSpectrumSettings &settings) {
  // Generate lens on host
  auto lens = generate_lens_host(static_cast<int>(Nx), static_cast<int>(Ny), settings);

  // Apply filter if specified
  if (settings.filter.has_value()) {
    auto                       &f = settings.filter.value();
    std::vector<cuFloatComplex> filter(Nx * Ny);
    generate_filter(filter.data(),
                    static_cast<uint32_t>(Nx),
                    static_cast<uint32_t>(Ny),
                    f.r_inner,
                    f.r_outer,
                    f.s_inner,
                    f.s_outer);

    fftshift2d(filter.data(), Nx, Ny);

    for (size_t i = 0; i < Nx * Ny; ++i) {
      lens[i] = cuCmulf(lens[i], filter[i]);
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

curaii::CufftHandle create_configured_lto_fft_handle(const FftPlanParams         &params,
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

curaii::CufftHandle create_configured_fft_handle(const FftPlanParams &params, cudaStream_t stream) {
  curaii::CufftHandle handle;
  CUFFT_CHECK(cufftSetStream(handle.get(), stream));

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
// AngularSpectrumFactory Implementation
// -------------------------------------------------------------------------------------------------

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
AngularSpectrumFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto  infer    = this->infer(input_descs, jsettings);
  auto  settings = jsettings.get<AngularSpectrumSettings>();
  auto &idesc    = input_descs[0];

  const size_t B = idesc.shape[0];
  const size_t H = idesc.shape[1];
  const size_t W = idesc.shape[2];

  // Generate lens
  auto lens_bytes = W * H * sizeof(cuFloatComplex);
  auto h_lens     = generate_lens_with_filter(W, H, settings);
  auto d_lens     = curaii::make_unique_device_ptr<cuFloatComplex>(lens_bytes);
  CUDA_CHECK(cudaMemcpy(d_lens.get(), h_lens.data(), lens_bytes, cudaMemcpyHostToDevice));

  // Prepare caller info
  using Info    = ApplyLensCallerInfo;
  auto lto_code = generate_apply_lens_lto();
  Info info{static_cast<int>(W), static_cast<int>(H), static_cast<int>(B), d_lens.get()};
  auto d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

  // Prepare FFT plans
  FftPlanParams params{.rank    = 2,
                       .n       = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .inembed = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .istride = 1,
                       .idist   = static_cast<int>(H * W),
                       .onembed = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .ostride = 1,
                       .odist   = static_cast<int>(H * W),
                       .batch   = static_cast<int>(B)};

  auto fwd_plan = create_configured_fft_handle(params, ctx.stream);
  auto inv_plan = create_configured_lto_fft_handle(params, lto_code, d_info, ctx.stream);

  // Success
  auto *task = new AngularSpectrum(settings,
                                   std::move(fwd_plan),
                                   std::move(inv_plan),
                                   std::move(d_lens),
                                   std::move(d_info),
                                   std::move(lto_code));
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

  const size_t B = idesc.shape[0];
  const size_t H = idesc.shape[1];
  const size_t W = idesc.shape[2];

  // Recreate lens
  auto lens_bytes = W * H * sizeof(cuFloatComplex);
  auto h_lens     = generate_lens_with_filter(W, H, settings);
  auto d_lens     = curaii::make_unique_device_ptr<cuFloatComplex>(lens_bytes);
  CUDA_CHECK(cudaMemcpy(d_lens.get(), h_lens.data(), lens_bytes, cudaMemcpyHostToDevice));

  // Recreate caller info (reuse LTO)
  using Info    = ApplyLensCallerInfo;
  auto lto_code = old_angular->lto_;
  Info info{static_cast<int>(W), static_cast<int>(H), static_cast<int>(B), d_lens.get()};
  auto d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(sizeof(info));
  CUDA_CHECK(cudaMemcpy(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice));

  // Recreate FFT plan
  FftPlanParams params{.rank    = 2,
                       .n       = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .inembed = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .istride = 1,
                       .idist   = static_cast<int>(H * W),
                       .onembed = {static_cast<long long int>(H), static_cast<long long int>(W)},
                       .ostride = 1,
                       .odist   = static_cast<int>(H * W),
                       .batch   = static_cast<int>(B)};

  auto fwd_plan = create_configured_fft_handle(params, ctx.stream);
  auto inv_plan = create_configured_lto_fft_handle(params, lto_code, d_info, ctx.stream);

  // Success
  auto *task = new AngularSpectrum(settings,
                                   std::move(fwd_plan),
                                   std::move(inv_plan),
                                   std::move(d_lens),
                                   std::move(d_info),
                                   std::move(lto_code));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::syncs
