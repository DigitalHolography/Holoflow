#include "tasks/fft_shift.hh"

#include <cuComplex.h>
#include <set>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks {

void to_json(nlohmann::json &j, const FFTShiftSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, FFTShiftSettings &) {}

namespace {

template <typename T>
__global__ void swap_corner_kernel(const T *idata, T *odata, int width, int height, int batch) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int width_half  = width / 2;
  int height_half = height / 2;

  if (x >= width_half || y >= height_half || z >= batch) {
    return;
  }

  int      batch_offset = z * width * height;
  const T *in_frame     = idata + batch_offset;
  T       *out_frame    = odata + batch_offset;

  // --- Swap top-left with bottom-right ---
  int top_left_idx     = x + y * width;
  int bottom_right_idx = (x + width_half) + (y + height_half) * width;

  T tmp                       = in_frame[top_left_idx];
  out_frame[top_left_idx]     = in_frame[bottom_right_idx];
  out_frame[bottom_right_idx] = tmp;

  // --- Swap top-right with bottom-left ---
  int top_right_idx   = (x + width_half) + y * width;
  int bottom_left_idx = x + (y + height_half) * width;

  tmp                        = in_frame[top_right_idx];
  out_frame[top_right_idx]   = in_frame[bottom_left_idx];
  out_frame[bottom_left_idx] = tmp;
}

template <class T> void launch(cudaStream_t s, const void *in, void *out, int W, int H, int B) {
  dim3      block(16, 16, 1);
  const int W2 = W >> 1, H2 = H >> 1;
  dim3      grid((W2 + block.x - 1) / block.x, (H2 + block.y - 1) / block.y,
                 (B + block.z - 1) / block.z);
  swap_corner_kernel<T>
      <<<grid, block, 0, s>>>(static_cast<const T *>(in), static_cast<T *>(out), W, H, B);
}

using Launcher = void (*)(cudaStream_t, const void *, void *, int, int, int);

constexpr int dtype_index(holoflow::core::DType dt) {
  switch (dt) {
  case holoflow::core::DType::U8:
    return 0;
  case holoflow::core::DType::U16:
    return 1;
  case holoflow::core::DType::F32:
    return 2;
  case holoflow::core::DType::CF32:
    return 3;
  default:
    return -1;
  }
}

static constexpr Launcher DISPATCH[4] = {
    &launch<uint8_t>,
    &launch<uint16_t>,
    &launch<float>,
    &launch<cuFloatComplex>,
};
} // namespace

FFTShift::FFTShift(const FFTShiftSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult FFTShift::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  int B               = static_cast<int>(idesc.shape[0]);
  int H               = static_cast<int>(idesc.shape[1]);
  int W               = static_cast<int>(idesc.shape[2]);
  int dt_idx          = dtype_index(idesc.dtype);
  HOLOVIBES_CHECK(dt_idx >= 0, "Unsupported dtype {}", static_cast<int>(idesc.dtype));
  auto f = DISPATCH[dt_idx];
  f(stream_, idata, odata, W, H, B);
  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
FFTShiftFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[FFTShiftFactory::infer] error: {}", msg);
      throw std::invalid_argument("FFTShiftFactory inference error: " + msg);
    }
  };

  const std::set<holoflow::core::DType> supported_dtypes = {
      holoflow::core::DType::U8,
      holoflow::core::DType::U16,
      holoflow::core::DType::F32,
      holoflow::core::DType::CF32,
  };

  // Validate
  auto settings = jsettings.get<FFTShiftSettings>();
  check(input_descs.size() == 1, "expected exactly one input");
  auto idesc = input_descs[0];
  check(idesc.shape.size() == 3, "expected 3D input");
  check(supported_dtypes.contains(idesc.dtype), "unsupported input dtype");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");

  // Success
  return {
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTShiftFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto infer = this->infer(input_descs, jsettings);
  (void)infer;
  auto settings = jsettings.get<FFTShiftSettings>();

  // Success
  auto *task = new FFTShift(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks