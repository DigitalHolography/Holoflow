#include "holotask/sources/fresnel_qin.hh"

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holotask::sources {

void to_json(nlohmann::json &j, const FresnelQinSettings &fqs) {
  j = nlohmann::json{
      {"lambda", fqs.lambda}, {"dx", fqs.dx}, {"dy", fqs.dy},
      {"z", fqs.z},           {"nx", fqs.nx}, {"ny", fqs.ny},
  };
}

void from_json(const nlohmann::json &j, FresnelQinSettings &fqs) {
  j.at("lambda").get_to(fqs.lambda);
  j.at("dx").get_to(fqs.dx);
  j.at("dy").get_to(fqs.dy);
  j.at("z").get_to(fqs.z);
  j.at("nx").get_to(fqs.nx);
  j.at("ny").get_to(fqs.ny);
}

FresnelQin::FresnelQin(const FresnelQinSettings &settings, DevPtr<cuFloatComplex> &&d_lens,
                       cudaStream_t stream)
    : settings_(settings), d_lens_(std::move(d_lens)), stream_(stream) {}

holoflow::core::OpResult FresnelQin::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata = d_lens_.get();
  auto *odata = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
  auto  bytes = settings_.nx * settings_.ny * sizeof(cuFloatComplex);
  CUDA_CHECK(cudaMemcpyAsync(odata, idata, bytes, cudaMemcpyDeviceToDevice, stream_));
  return holoflow::core::OpResult::Ok;
}

namespace {

__global__ void quadratic_lens_kernel(cuFloatComplex *lens, int width, int height, float lambda,
                                      float z, float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size = width > height ? width : height;

  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * pixel_size;
  float y = ((row + offset_y) - size / 2.0f) * pixel_size;

  float phase     = CUDART_PI_F / (lambda * z) * (x * x + y * y);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

DevPtr<cuFloatComplex> make_quadratic_lens(const FresnelQinSettings &settings) {
  auto bytes  = settings.nx * settings.ny * sizeof(cuFloatComplex);
  auto d_lens = curaii::make_unique_device_ptr<cuFloatComplex>(bytes);
  auto nx     = static_cast<int>(settings.nx);
  ;
  auto ny = static_cast<int>(settings.ny);

  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
  quadratic_lens_kernel<<<grid_size, block_size>>>(d_lens.get(), nx, ny, settings.lambda,
                                                   settings.z, settings.dx);

  CUDA_CHECK(cudaGetLastError());
  return d_lens;
}

} // namespace

holoflow::core::InferResult
FresnelQinFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[FresnelQinFactory::infer] error: {}", msg);
      throw std::invalid_argument("FresnelQinFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<FresnelQinSettings>();

  check(input_descs.size() == 0, "expected zero input tensors");
  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");
  check(settings.z != 0.0f, "propagation distance z must be non-zero");
  check(settings.nx > 0, "nx must be positive");
  check(settings.ny > 0, "ny must be positive");

  holoflow::core::TDesc odesc({settings.ny, settings.nx}, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelQinFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings,
                          const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<FresnelQinSettings>();

  auto  d_lens = make_quadratic_lens(settings);
  auto *task   = new FresnelQin(settings, std::move(d_lens), ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::sources