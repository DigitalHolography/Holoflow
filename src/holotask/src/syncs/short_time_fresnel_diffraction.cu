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

#include "holotask/syncs/short_time_fresnel_diffraction.hh"

#include <algorithm>
#include <cstdlib>
#include <ranges>
#include <string>
#include <vector>

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ShortTimeFresnelDiffractionSettings &s) {
  j = {{"lambda", s.lambda},
       {"dx", s.dx},
       {"dy", s.dy},
       {"z", s.z},
       {"win_h", s.win_h},
       {"win_w", s.win_w},
       {"stride_y", s.stride_y},
       {"stride_x", s.stride_x},
       {"phase_ref", s.phase_ref},
       {"skip_phase_shift", s.skip_phase_shift},
       {"axes", s.axes}};
}

void from_json(const nlohmann::json &j, ShortTimeFresnelDiffractionSettings &s) {
  j.at("lambda").get_to(s.lambda);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("z").get_to(s.z);
  j.at("win_h").get_to(s.win_h);
  j.at("win_w").get_to(s.win_w);
  j.at("stride_y").get_to(s.stride_y);
  j.at("stride_x").get_to(s.stride_x);
  if (j.contains("phase_ref"))
    j.at("phase_ref").get_to(s.phase_ref);
  if (j.contains("skip_phase_shift"))
    j.at("skip_phase_shift").get_to(s.skip_phase_shift);
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
}

// -------------------------------------------------------------------------------------------------
// Internal types (shared layout with NVRTC source — must stay in sync)
// -------------------------------------------------------------------------------------------------

namespace {

struct LaunchOffset {
  size_t in_bytes;
  size_t out_bytes;
};

// Caller-info struct embedded verbatim into the NVRTC source string.
// Alignment note: all 4-byte fields come first, then the 8-byte field, then floats.
struct STFTCallerInfo {
  unsigned int       win_w;
  unsigned int       win_h;
  unsigned int       nx_win;
  unsigned int       ny_win;
  unsigned int       stride_x;
  unsigned int       stride_y;
  unsigned int       field_W;
  unsigned int       _pad;        // keeps field_idist 8-byte aligned
  unsigned long long field_idist; // elements between consecutive batch items in the original field
  float              dx;
  float              dy;
  float              pi_over_lz; // π / (λ * z)
  float              x_origin;   // physical x-coord of pixel index 0 in the chosen frame
  float              y_origin;   // physical y-coord of pixel index 0 in the chosen frame
};

// Shared header for both callback variants.
constexpr const char *k_stft_callback_header = R"(
#include <cuComplex.h>

struct STFTCallerInfo {
  unsigned int win_w, win_h, nx_win, ny_win;
  unsigned int stride_x, stride_y, field_W, _pad;
  unsigned long long field_idist;
  float dx, dy, pi_over_lz, x_origin, y_origin;
};

// Decode a flat cuFFT offset into window grid and local pixel coordinates.
// offset = (b_inner * ny_win * nx_win + gy * nx_win + gx) * win_h * win_w + j * win_w + i
__device__ __forceinline__ void stft_decode(
    const STFTCallerInfo *info, size_t offset,
    unsigned long long &b, unsigned int &gx, unsigned int &gy,
    unsigned int &i,  unsigned int &j)
{
  unsigned long long tile  = (unsigned long long)info->win_w * info->win_h;
  unsigned long long grid  = (unsigned long long)info->nx_win * info->ny_win;
  unsigned long long lf    = offset % tile;
  i  = (unsigned int)(lf % info->win_w);
  j  = (unsigned int)(lf / info->win_w);
  unsigned long long sf = (offset / tile) % grid;
  gx = (unsigned int)(sf % info->nx_win);
  gy = (unsigned int)(sf / info->nx_win);
  b  = offset / (tile * grid);
}
)";

// LOCAL callback — lens centered on the window (same as classic fresnel_diffraction).
constexpr const char *k_stft_load_local_body = R"(
__device__ cuFloatComplex stft_load_local(
    void *data, size_t offset, void *callerInfo, void *sharedPtr)
{
  auto *info = (STFTCallerInfo *)callerInfo;
  unsigned long long b; unsigned int gx, gy, i, j;
  stft_decode(info, offset, b, gx, gy, i, j);

  unsigned long long src = b * info->field_idist
                         + (unsigned long long)(gy * info->stride_y + j) * info->field_W
                         + (unsigned long long)(gx * info->stride_x + i);
  cuFloatComplex val = ((cuFloatComplex *)data)[src];

  // Phase: π/(λz) * (x_local² + y_local²), window centered.
  float x = (float)i * info->dx + info->x_origin;
  float y = (float)j * info->dy + info->y_origin;
  float s, c;
  sincosf(info->pi_over_lz * (x * x + y * y), &s, &c);
  return cuCmulf(val, make_cuComplex(c, s));
}
)";

// GLOBAL callback — lens evaluated at global field coordinates.
// Combined phase = π/(λz) * (x_global² + y_global²) which is the standard global quadratic lens.
constexpr const char *k_stft_load_global_body = R"(
__device__ cuFloatComplex stft_load_global(
    void *data, size_t offset, void *callerInfo, void *sharedPtr)
{
  auto *info = (STFTCallerInfo *)callerInfo;
  unsigned long long b; unsigned int gx, gy, i, j;
  stft_decode(info, offset, b, gx, gy, i, j);

  unsigned long long src = b * info->field_idist
                         + (unsigned long long)(gy * info->stride_y + j) * info->field_W
                         + (unsigned long long)(gx * info->stride_x + i);
  cuFloatComplex val = ((cuFloatComplex *)data)[src];

  // Phase: π/(λz) * (x_global² + y_global²).
  // x_origin encodes the physical coordinate of global pixel index 0.
  float x = (float)(gx * info->stride_x + i) * info->dx + info->x_origin;
  float y = (float)(gy * info->stride_y + j) * info->dy + info->y_origin;
  float s, c;
  sincosf(info->pi_over_lz * (x * x + y * y), &s, &c);
  return cuCmulf(val, make_cuComplex(c, s));
}
)";

// -------------------------------------------------------------------------------------------------
// NVRTC helpers  (mirrors fresnel_diffraction.cu)
// -------------------------------------------------------------------------------------------------

std::string get_compute_arch() {
  int            device{};
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return "compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<char> compile_lto(const std::string &src, const std::string &name) {
  auto CUDA_PATH = std::getenv("CUDA_PATH");
  HOLOVIBES_CHECK(CUDA_PATH != nullptr, "CUDA_PATH environment variable not set");

  std::vector<std::string> args_str = {
      "-I" + std::string{CUDA_PATH} + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };

  curaii::NvrtcProgram prog(src.c_str(), name.c_str(), 0, nullptr, nullptr);
  std::vector<char *>  args;
  std::ranges::transform(args_str, std::back_inserter(args),
                         [](const std::string &s) { return const_cast<char *>(s.c_str()); });
  try {
    NVRTC_CHECK(nvrtcCompileProgram(prog.get(), static_cast<int>(args.size()), args.data()));
    size_t sz = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog.get(), &sz));
    std::vector<char> lto(sz);
    NVRTC_CHECK(nvrtcGetLTOIR(prog.get(), lto.data()));
    return lto;
  } catch (const curaii::NvrtcError &) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog.get(), &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog.get(), log.data()));
    logger()->error("[ShortTimeFresnelDiffraction] NVRTC log:\n{}", log);
    throw;
  }
}

std::vector<char> build_stft_lto(bool is_global) {
  std::string src = k_stft_callback_header;
  src += is_global ? k_stft_load_global_body : k_stft_load_local_body;
  const char *name = is_global ? "stft_load_global.cu" : "stft_load_local.cu";
  return compile_lto(src, name);
}

// -------------------------------------------------------------------------------------------------
// Batch-grouping helpers  (identical logic to fresnel_diffraction.cu)
// -------------------------------------------------------------------------------------------------

struct BatchGroup {
  size_t           size      = 1;
  long long int    in_idist  = 0;
  long long int    out_idist = 0;
  std::vector<int> dims;
};

std::vector<size_t> strides_bytes(const holoflow::core::TDesc &d) {
  if (!d.strides.empty())
    return d.strides;
  std::vector<size_t> s(d.shape.size());
  size_t              acc = holoflow::core::size_of(d.dtype);
  for (size_t i = d.shape.size(); i-- > 0;) {
    s[i] = acc;
    acc *= d.shape[i];
  }
  return s;
}

BatchGroup select_batch_group(const std::vector<size_t> &shape, const std::vector<size_t> &in_sb,
                              const std::vector<size_t> &out_sb, const std::vector<int> &outer_dims,
                              size_t in_esz, size_t out_esz, long long int def_in_idist,
                              long long int def_out_idist) {
  BatchGroup best{1, def_in_idist, def_out_idist, {}};
  if (outer_dims.empty())
    return best;

  auto sorted = outer_dims;
  std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return in_sb[a] < in_sb[b]; });

  for (size_t start = 0; start < sorted.size(); ++start) {
    int        d = sorted[start];
    BatchGroup cur;
    cur.size      = shape[d];
    cur.in_idist  = static_cast<long long>(in_sb[d] / in_esz);
    cur.out_idist = static_cast<long long>(out_sb[d] / out_esz);
    cur.dims.push_back(d);

    if (cur.in_idist < def_in_idist || cur.out_idist < def_out_idist)
      continue;
    if (cur.size > best.size)
      best = cur;

    size_t acc = cur.size;
    for (size_t nxt = start + 1; nxt < sorted.size(); ++nxt) {
      int  nd  = sorted[nxt];
      auto nii = static_cast<long long>(in_sb[nd] / in_esz);
      auto noi = static_cast<long long>(out_sb[nd] / out_esz);
      if (nii != cur.in_idist * static_cast<long long>(acc))
        break;
      if (noi != cur.out_idist * static_cast<long long>(acc))
        break;
      acc *= shape[nd];
      cur.size = acc;
      cur.dims.push_back(nd);
      if (cur.size > best.size)
        best = cur;
    }
  }
  return best;
}

void gen_offsets(const std::vector<size_t> &shape, const std::vector<size_t> &in_sb,
                 const std::vector<size_t> &out_sb, const std::vector<int> &dims, size_t dim_idx,
                 size_t cur_in, size_t cur_out, std::vector<LaunchOffset> &out) {
  if (dim_idx == dims.size()) {
    out.push_back({cur_in, cur_out});
    return;
  }
  int d = dims[dim_idx];
  for (size_t i = 0; i < shape[d]; ++i)
    gen_offsets(shape, in_sb, out_sb, dims, dim_idx + 1, cur_in + i * in_sb[d],
                cur_out + i * out_sb[d], out);
}

// -------------------------------------------------------------------------------------------------
// Output-plane quadratic phase kernel  (identical to fresnel_diffraction.cu)
// -------------------------------------------------------------------------------------------------

__global__ void stft_output_phase_kernel(cuFloatComplex *output, const cuFloatComplex *lens,
                                         size_t batch, int height, int width, long long int idist,
                                         long long int stride_h, long long int istride) {
  auto lid   = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = static_cast<unsigned long long>(batch) * height * width;
  if (lid >= total)
    return;
  auto b   = lid / (static_cast<unsigned long long>(height) * width);
  auto loc = lid % (static_cast<unsigned long long>(height) * width);
  auto row = static_cast<int>(loc / width);
  auto col = static_cast<int>(loc % width);
  auto idx = b * idist + static_cast<unsigned long long>(row) * stride_h +
             static_cast<unsigned long long>(col) * istride;
  output[idx] = cuCmulf(output[idx], lens[static_cast<unsigned long long>(row) * width + col]);
}

// Build the output-plane quadratic lens for a window (same convention as classic Fresnel).
DevPtr<cuFloatComplex> make_win_lens(int win_w, int win_h, float lambda, float z, float dx) {
  int   size  = win_w > win_h ? win_w : win_h;
  int   off_x = (size - win_w) / 2;
  int   off_y = (size - win_h) / 2;
  float pil   = CUDART_PI_F / (lambda * z);

  auto d = curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(win_w) * win_h);

  // Host-side fill (small lens) — for large windows a kernel would be better, but consistency
  // with fresnel_diffraction.cu (which uses a kernel) is maintained by calling the same pattern.
  // Here we use a simple thrust-free approach via a temporary host vector.
  std::vector<cuFloatComplex> h(static_cast<size_t>(win_w) * win_h);
  for (int row = 0; row < win_h; ++row) {
    for (int col = 0; col < win_w; ++col) {
      float x              = ((col + off_x) - size / 2.0f) * dx;
      float y              = ((row + off_y) - size / 2.0f) * dx;
      float phase          = pil * (x * x + y * y);
      h[row * win_w + col] = make_cuComplex(cosf(phase), sinf(phase));
    }
  }
  CUDA_CHECK(
      cudaMemcpy(d.get(), h.data(), h.size() * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  return d;
}

// -------------------------------------------------------------------------------------------------
// Axis helpers
// -------------------------------------------------------------------------------------------------

void check(bool cond, const std::string &msg) {
  if (!cond) {
    logger()->error("[ShortTimeFresnelDiffractionFactory] {}", msg);
    throw std::invalid_argument("ShortTimeFresnelDiffractionFactory: " + msg);
  }
}

std::pair<int, int> normalize_axes(const ShortTimeFresnelDiffractionSettings &s, int rank) {
  auto axes = s.axes.empty() ? std::vector<int>{-2, -1} : s.axes;
  check(axes.size() == 2, "axes must contain exactly two elements");
  int ax0 = axes[0] < 0 ? axes[0] + rank : axes[0];
  int ax1 = axes[1] < 0 ? axes[1] + rank : axes[1];
  check(ax0 >= 0 && ax0 < rank && ax1 >= 0 && ax1 < rank, "axes out of bounds");
  check(ax0 != ax1, "axes must be distinct");
  return {ax0, ax1};
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Impl
// -------------------------------------------------------------------------------------------------

struct ShortTimeFresnelDiffraction::Impl {
  ShortTimeFresnelDiffractionSettings settings;
  holoflow::core::TDesc               idesc;
  curaii::CufftHandle                 fft_handle;

  // Launch geometry
  std::vector<LaunchOffset> offsets;
  size_t                    inner_batch; // batch items per cufftXtExec (folded outer dims)
  size_t                    ny_win, nx_win;
  int                       win_h, win_w;
  long long int             out_idist; // win_h * win_w (elements between windows in output)
  long long int             out_stride_h;
  long long int             out_istride;

  // Device resources
  cudaStream_t           stream;
  DevPtr<cuFloatComplex> d_win_lens; // output-plane quadratic lens [win_h, win_w]
  DevPtr<void>           d_caller_info;
  std::vector<char>      lto;
};

// -------------------------------------------------------------------------------------------------
// ShortTimeFresnelDiffraction
// -------------------------------------------------------------------------------------------------

ShortTimeFresnelDiffraction::ShortTimeFresnelDiffraction(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ShortTimeFresnelDiffraction::~ShortTimeFresnelDiffraction() = default;

const holoflow::core::TDesc &ShortTimeFresnelDiffraction::get_idesc() const { return impl_->idesc; }
const ShortTimeFresnelDiffractionSettings &ShortTimeFresnelDiffraction::get_settings() const {
  return impl_->settings;
}

void ShortTimeFresnelDiffraction::update_stream(cudaStream_t stream) {
  if (impl_->stream != stream) {
    impl_->stream = stream;
    CUFFT_CHECK(cufftSetStream(impl_->fft_handle.get(), stream));
  }
}

holoflow::core::OpResult ShortTimeFresnelDiffraction::execute(holoflow::core::SyncCtx &ctx) {
  auto &im = *impl_;

  auto *idata = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *odata = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());

  for (const auto &off : im.offsets) {
    auto *in_ptr  = idata + off.in_bytes;
    auto *out_ptr = reinterpret_cast<cuFloatComplex *>(odata + off.out_bytes);

    CUFFT_CHECK(cufftXtExec(im.fft_handle.get(), in_ptr, out_ptr, CUFFT_FORWARD));

    if (!im.settings.skip_phase_shift) {
      constexpr int block_size = 256;
      auto          total_wins = im.inner_batch * im.ny_win * im.nx_win;
      auto          total      = total_wins * static_cast<size_t>(im.win_h) * im.win_w;
      int           grid_size  = static_cast<int>((total + block_size - 1) / block_size);
      stft_output_phase_kernel<<<grid_size, block_size, 0, im.stream>>>(
          out_ptr, im.d_win_lens.get(), total_wins, im.win_h, im.win_w, im.out_idist,
          im.out_stride_h, im.out_istride);
    }
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// Factory: infer
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ShortTimeFresnelDiffractionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                          const nlohmann::json                  &jsettings) const {
  auto s = jsettings.get<ShortTimeFresnelDiffractionSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &id = input_descs[0];
  check(id.rank() >= 2, "input must be rank ≥ 2");
  check(id.dtype == holoflow::core::DType::CF32, "input must be CF32");
  check(id.mem_loc == holoflow::core::MemLoc::Device, "input must be on device");
  check(s.lambda > 0.f, "lambda must be positive");
  check(s.dx > 0.f, "dx must be positive");
  check(s.dy > 0.f, "dy must be positive");
  check(s.dx == s.dy, "dx must equal dy");
  check(s.z != 0.f, "z must be non-zero");
  check(s.win_h >= 1 && s.win_w >= 1, "window dimensions must be ≥ 1");
  check(s.stride_y >= 1 && s.stride_x >= 1, "strides must be ≥ 1");

  auto [ax0, ax1] = normalize_axes(s, static_cast<int>(id.rank()));
  const size_t H  = id.shape[ax0];
  const size_t W  = id.shape[ax1];
  check(s.win_h <= H && s.win_w <= W, "window dimensions exceed field size");

  const size_t ny_win = (H - s.win_h) / s.stride_y + 1;
  const size_t nx_win = (W - s.win_w) / s.stride_x + 1;

  // Output: replace the two spatial dims with [ny_win, nx_win, win_h, win_w].
  std::vector<size_t> out_shape(id.shape.begin(), id.shape.end() - 2);
  out_shape.push_back(ny_win);
  out_shape.push_back(nx_win);
  out_shape.push_back(s.win_h);
  out_shape.push_back(s.win_w);

  holoflow::core::TDesc odesc(out_shape, holoflow::core::DType::CF32,
                              holoflow::core::MemLoc::Device);

  return {.input_descs   = {id},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

// -------------------------------------------------------------------------------------------------
// Factory: create
// -------------------------------------------------------------------------------------------------

std::unique_ptr<holoflow::core::ISyncTask>
ShortTimeFresnelDiffractionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                           const nlohmann::json                  &jsettings,
                                           const holoflow::core::SyncCreateCtx   &ctx) const {
  auto inf = this->infer(input_descs, jsettings);
  auto s   = jsettings.get<ShortTimeFresnelDiffractionSettings>();

  const auto &idesc = input_descs[0];
  const auto &odesc = inf.output_descs[0];
  const int   rank  = static_cast<int>(idesc.rank());
  auto [ax0, ax1]   = normalize_axes(s, rank);

  const int    H      = static_cast<int>(idesc.shape[ax0]);
  const int    W      = static_cast<int>(idesc.shape[ax1]);
  const int    win_h  = static_cast<int>(s.win_h);
  const int    win_w  = static_cast<int>(s.win_w);
  const size_t ny_win = (H - win_h) / s.stride_y + 1;
  const size_t nx_win = (W - win_w) / s.stride_x + 1;

  constexpr size_t esz = sizeof(cuFloatComplex);

  auto in_sb  = strides_bytes(idesc);
  auto out_sb = strides_bytes(odesc);

  // Outer dims: everything except ax0 and ax1.
  std::vector<int> outer_dims;
  outer_dims.reserve(static_cast<size_t>(rank) - 2);
  for (int i = 0; i < rank; ++i)
    if (i != ax0 && i != ax1)
      outer_dims.push_back(i);

  // Per-field memory span in the original field (real input side).
  // Per-field output span: all window data for one field.
  const long long def_in_idist  = static_cast<long long>(in_sb[ax0] / esz) * H;
  const long long def_out_idist = static_cast<long long>(ny_win * nx_win * win_h * win_w);

  auto group = select_batch_group(idesc.shape, in_sb, out_sb, outer_dims, esz, esz, def_in_idist,
                                  def_out_idist);

  // Dims not folded into the inner batch → one cufftXtExec per combination.
  std::vector<int> launch_dims;
  for (int d : outer_dims)
    if (std::find(group.dims.begin(), group.dims.end(), d) == group.dims.end())
      launch_dims.push_back(d);

  std::vector<LaunchOffset> offsets;
  gen_offsets(idesc.shape, in_sb, out_sb, launch_dims, 0, 0, 0, offsets);
  if (offsets.empty())
    offsets.push_back({0, 0});

  // -- Output strides (contiguous layout for the output array) -----------------------------------
  const long long out_win_idist = static_cast<long long>(win_h) * win_w;
  const long long out_stride_h  = static_cast<long long>(win_w);
  const long long out_istride   = 1LL;

  // -- Caller-info struct -----------------------------------------------------------------------
  // Physical coordinate of pixel 0 in x/y, in the chosen phase frame.
  float x_origin, y_origin;
  if (s.phase_ref == STFDPhaseReference::LOCAL) {
    // Frame: window centered. Convention identical to make_quadratic_lens in FresnelDiffraction.
    int loc_size  = win_w > win_h ? win_w : win_h;
    int loc_off_x = (loc_size - win_w) / 2;
    int loc_off_y = (loc_size - win_h) / 2;
    x_origin      = (loc_off_x - loc_size / 2.0f) * s.dx;
    y_origin      = (loc_off_y - loc_size / 2.0f) * s.dy;
  } else {
    // Frame: full field centered. Same convention: largest dimension sets the scale.
    int glo_size  = W > H ? W : H;
    int glo_off_x = (glo_size - W) / 2;
    int glo_off_y = (glo_size - H) / 2;
    x_origin      = (glo_off_x - glo_size / 2.0f) * s.dx;
    y_origin      = (glo_off_y - glo_size / 2.0f) * s.dy;
  }

  STFTCallerInfo info{
      .win_w       = static_cast<unsigned>(win_w),
      .win_h       = static_cast<unsigned>(win_h),
      .nx_win      = static_cast<unsigned>(nx_win),
      .ny_win      = static_cast<unsigned>(ny_win),
      .stride_x    = static_cast<unsigned>(s.stride_x),
      .stride_y    = static_cast<unsigned>(s.stride_y),
      .field_W     = static_cast<unsigned>(W),
      ._pad        = 0,
      .field_idist = static_cast<unsigned long long>(group.in_idist),
      .dx          = s.dx,
      .dy          = s.dy,
      .pi_over_lz  = CUDART_PI_F / (s.lambda * s.z),
      .x_origin    = x_origin,
      .y_origin    = y_origin,
  };
  auto d_info = curaii::make_unique_device_ptr<STFTCallerInfo>(1);
  CUDA_CHECK(
      cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream));

  // -- LTO compilation --------------------------------------------------------------------------
  bool is_global = (s.phase_ref == STFDPhaseReference::GLOBAL);
  auto lto       = build_stft_lto(is_global);

  // -- Window-lens for output phase shift -------------------------------------------------------
  auto d_win_lens = make_win_lens(win_w, win_h, s.lambda, s.z, s.dx);

  // -- cuFFT plan -------------------------------------------------------------------------------
  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));

  void       *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  const char *cb_name    = is_global ? "stft_load_global" : "stft_load_local";
  CUFFT_CHECK(cufftXtSetJITCallback(plan.get(), cb_name, lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  // Virtual input layout (overridden by callback): [inner_batch*ny_win*nx_win, win_h, win_w].
  long long int n[2]       = {win_h, win_w};
  long long int inembed[2] = {win_h, win_w}; // unused by callback but must be valid
  long long int onembed[2] = {win_h, win_w};
  size_t        work_size  = 0;

  long long batch = static_cast<long long>(group.size) * static_cast<long long>(ny_win * nx_win);

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, n, inembed, 1LL,
                                  static_cast<long long>(win_h * win_w),
                                  CUDA_C_32F,                              // input (virtual)
                                  onembed, 1LL, out_win_idist, CUDA_C_32F, // output (real)
                                  batch, &work_size, CUDA_C_32F));

  auto impl = std::make_unique<ShortTimeFresnelDiffraction::Impl>(ShortTimeFresnelDiffraction::Impl{
      .settings      = s,
      .idesc         = idesc,
      .fft_handle    = std::move(plan),
      .offsets       = std::move(offsets),
      .inner_batch   = group.size,
      .ny_win        = ny_win,
      .nx_win        = nx_win,
      .win_h         = win_h,
      .win_w         = win_w,
      .out_idist     = out_win_idist,
      .out_stride_h  = out_stride_h,
      .out_istride   = out_istride,
      .stream        = ctx.stream,
      .d_win_lens    = std::move(d_win_lens),
      .d_caller_info = std::move(d_info),
      .lto           = std::move(lto),
  });

  return std::make_unique<ShortTimeFresnelDiffraction>(std::move(impl));
}

// -------------------------------------------------------------------------------------------------
// Factory: update
// -------------------------------------------------------------------------------------------------

std::unique_ptr<holoflow::core::ISyncTask>
ShortTimeFresnelDiffractionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                           std::span<const holoflow::core::TDesc>     input_descs,
                                           const nlohmann::json                      &jsettings,
                                           const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old = dynamic_cast<ShortTimeFresnelDiffraction *>(old_task.get());
  if (old == nullptr || input_descs.size() != 1)
    return create(input_descs, jsettings, ctx);

  const auto &nid = input_descs[0];
  const auto &oid = old->get_idesc();
  auto        s   = jsettings.get<ShortTimeFresnelDiffractionSettings>();

  bool reusable = (s == old->get_settings()) && (nid.shape == oid.shape) &&
                  (nid.strides == oid.strides) && (nid.dtype == oid.dtype) &&
                  (nid.mem_loc == oid.mem_loc);

  if (reusable) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
