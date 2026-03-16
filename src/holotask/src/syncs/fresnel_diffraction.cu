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

#include "holotask/syncs/fresnel_diffraction.hh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <ranges>
#include <string>
#include <vector>

#include <math_constants.h>

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds) {
  j = nlohmann::json{
      {"lambda", fds.lambda}, {"dx", fds.dx},     {"dy", fds.dy},
      {"z", fds.z},           {"axes", fds.axes}, {"skip_phase_shift", fds.skip_phase_shift},
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
  if (j.contains("skip_phase_shift")) {
    j.at("skip_phase_shift").get_to(fds.skip_phase_shift);
  } else {
    fds.skip_phase_shift = true;
  }
}

namespace {

struct ApplyLensCallerInfo {
  unsigned int       width;
  unsigned int       _padding;
  unsigned long long idist;
  unsigned long long stride_h;
  unsigned long long istride;
  cuFloatComplex    *lens;
};

struct BatchGroup {
  size_t           size      = 1;
  long long int    in_idist  = 0;
  long long int    out_idist = 0;
  std::vector<int> dims;
};

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[FresnelDiffractionFactory::infer] error: {}", msg);
    throw std::invalid_argument("FresnelDiffractionFactory inference error: " + msg);
  }
}

std::vector<size_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty())
    return desc.strides;

  std::vector<size_t> strides(desc.shape.size());
  size_t              acc = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
}

std::pair<int, int> normalize_axes(const FresnelDiffractionSettings &settings, int rank) {
  auto axes = settings.axes.empty() ? std::vector<int>{-2, -1} : settings.axes;
  check(axes.size() == 2, "axes must contain exactly two dimensions");

  int ax0 = axes[0] < 0 ? axes[0] + rank : axes[0];
  int ax1 = axes[1] < 0 ? axes[1] + rank : axes[1];
  check(ax0 >= 0 && ax0 < rank && ax1 >= 0 && ax1 < rank, "axes out of bounds");
  check(ax0 != ax1, "axes must be distinct");
  return {ax0, ax1};
}

void generate_offsets_recursive(const std::vector<size_t> &shape,
                                const std::vector<size_t> &in_strides,
                                const std::vector<size_t> &out_strides,
                                const std::vector<int> &outer_dims, size_t dim_idx,
                                size_t current_in, size_t current_out,
                                std::vector<LaunchOffset> &out_offsets) {
  if (dim_idx == outer_dims.size()) {
    out_offsets.push_back({current_in, current_out});
    return;
  }

  int d = outer_dims[dim_idx];
  for (size_t i = 0; i < shape[d]; ++i) {
    generate_offsets_recursive(shape, in_strides, out_strides, outer_dims, dim_idx + 1,
                               current_in + i * in_strides[d], current_out + i * out_strides[d],
                               out_offsets);
  }
}

BatchGroup select_best_batch_group(const std::vector<size_t> &shape,
                                   const std::vector<size_t> &in_strides_bytes,
                                   const std::vector<size_t> &out_strides_bytes,
                                   const std::vector<int> &outer_dims, size_t in_elem_size,
                                   size_t out_elem_size, long long int default_in_idist,
                                   long long int default_out_idist) {
  BatchGroup best{
      .size      = 1,
      .in_idist  = default_in_idist,
      .out_idist = default_out_idist,
      .dims      = {},
  };

  if (outer_dims.empty())
    return best;

  auto sorted_outer_dims = outer_dims;
  std::sort(sorted_outer_dims.begin(), sorted_outer_dims.end(),
            [&](int lhs, int rhs) { return in_strides_bytes[lhs] < in_strides_bytes[rhs]; });

  for (size_t start = 0; start < sorted_outer_dims.size(); ++start) {
    BatchGroup current;
    int        dim = sorted_outer_dims[start];
    current.size      = shape[dim];
    current.in_idist  = static_cast<long long int>(in_strides_bytes[dim] / in_elem_size);
    current.out_idist = static_cast<long long int>(out_strides_bytes[dim] / out_elem_size);

    // FIX: Prevent batching if the dimension is interleaved inside the spatial dimensions
    if (current.in_idist < default_in_idist || current.out_idist < default_out_idist) {
      continue;
    }

    current.dims.push_back(dim);

    if (current.size > best.size)
      best = current;

    size_t current_size = current.size;
    for (size_t next_idx = start + 1; next_idx < sorted_outer_dims.size(); ++next_idx) {
      int           next_dim      = sorted_outer_dims[next_idx];
      long long int next_in_idist =
          static_cast<long long int>(in_strides_bytes[next_dim] / in_elem_size);
      long long int next_out_idist =
          static_cast<long long int>(out_strides_bytes[next_dim] / out_elem_size);

      if (next_in_idist != current.in_idist * static_cast<long long int>(current_size) ||
          next_out_idist != current.out_idist * static_cast<long long int>(current_size)) {
        break;
      }

      current_size *= shape[next_dim];
      current.size = current_size;
      current.dims.push_back(next_dim);

      if (current.size > best.size)
        best = current;
    }
  }

  return best;
}

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

  return {
      "-I" + cuda_path + "/include",
      "-arch=" + get_compute_arch(),
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto",
  };
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

  logger()->info("[FresnelDiffraction] apply_lens_callback source code:\n{}", apply_lens_callback);

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

  float phase             = CUDART_PI_F / (lambda * z) * (x * x + y * y);
  lens[row * width + col] = make_cuComplex(cosf(phase), sinf(phase));
}

DevPtr<cuFloatComplex> make_quadratic_lens(int width, int height,
                                           const FresnelDiffractionSettings &settings) {
  auto d_lens = curaii::make_unique_device_ptr<cuFloatComplex>(static_cast<size_t>(width) * height);

  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  quadratic_lens_kernel<<<grid_size, block_size>>>(d_lens.get(), width, height, settings.lambda,
                                                   settings.z, settings.dx);
  return d_lens;
}

__global__ void apply_output_phase_shift_kernel(cuFloatComplex *output, const cuFloatComplex *lens,
                                                size_t batch, int height, int width,
                                                long long int idist, long long int stride_h,
                                                long long int istride) {
  auto linear_idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto plane_size = static_cast<unsigned long long>(height) * width;
  auto total_size = static_cast<unsigned long long>(batch) * plane_size;
  if (linear_idx >= total_size)
    return;

  auto batch_idx = linear_idx / plane_size;
  auto local_idx = linear_idx % plane_size;
  int  row       = static_cast<int>(local_idx / width);
  int  col       = static_cast<int>(local_idx % width);

  auto out_idx = static_cast<unsigned long long>(batch_idx) * idist +
                 static_cast<unsigned long long>(row) * stride_h +
                 static_cast<unsigned long long>(col) * istride;

  output[out_idx] =
      cuCmulf(output[out_idx], lens[static_cast<unsigned long long>(row) * width + col]);
}

} // namespace

FresnelDiffraction::FresnelDiffraction(const FresnelDiffractionSettings &settings,
                                       holoflow::core::TDesc             idesc,
                                       curaii::CufftHandle             &&fft_handle,
                                       std::vector<LaunchOffset> offsets, size_t inner_batch,
                                       int height, int width, long long out_idist,
                                       long long out_stride_h, long long out_istride,
                                       cudaStream_t stream, DevPtr<cuFloatComplex> &&d_lens,
                                       DevPtr<void> &&d_caller_info, std::vector<char> &&lto)
    : settings_(settings), idesc_(std::move(idesc)), fft_handle_(std::move(fft_handle)),
      offsets_(std::move(offsets)), inner_batch_(inner_batch), stream_(stream), height_(height),
      width_(width), out_idist_(out_idist), out_stride_h_(out_stride_h), out_istride_(out_istride),
      d_lens_(std::move(d_lens)), d_caller_info_(std::move(d_caller_info)), lto_(std::move(lto)) {}

void FresnelDiffraction::update_stream(cudaStream_t stream) {
  if (stream_ != stream) {
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(fft_handle_.get(), stream_));
  }
}

holoflow::core::OpResult FresnelDiffraction::execute(holoflow::core::SyncCtx &ctx) {
  auto *idata_base = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *odata_base = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());

  for (const auto &offset : offsets_) {
    auto *in_ptr  = idata_base + offset.in_bytes;
    auto *out_ptr = reinterpret_cast<cuFloatComplex *>(odata_base + offset.out_bytes);
    CUFFT_CHECK(cufftXtExec(fft_handle_.get(), in_ptr, out_ptr, CUFFT_FORWARD));

    if (!settings_.skip_phase_shift) {
      constexpr int block_size = 256;
      auto          total      = inner_batch_ * static_cast<size_t>(height_) * width_;
      int           grid_size  = static_cast<int>((total + block_size - 1) / block_size);
      apply_output_phase_shift_kernel<<<grid_size, block_size, 0, stream_>>>(
          out_ptr, d_lens_.get(), inner_batch_, height_, width_, out_idist_, out_stride_h_,
          out_istride_);
    }
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
FresnelDiffractionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings) const {
  auto settings = jsettings.get<FresnelDiffractionSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() >= 2, "input must be a tensor of rank 2 or higher");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32,
        "input must be complex float32 or real float32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");

  auto [ax0, ax1] = normalize_axes(settings, static_cast<int>(idesc.rank()));
  auto in_strides = get_strides_bytes(idesc);
  check(
      in_strides[ax0] % in_strides[ax1] == 0,
      "stride of ax0 (H) must be a multiple of stride of ax1 (W) for cuFFT to run without copying");

  check(settings.lambda > 0.0f, "wavelength must be positive");
  check(settings.dx > 0.0f, "dx must be positive");
  check(settings.dy > 0.0f, "dy must be positive");
  check(settings.dx == settings.dy, "dx must equal dy");

  holoflow::core::TDesc odesc;
  if (idesc.dtype == holoflow::core::DType::CF32) {
    odesc = idesc;
  } else {
    odesc = holoflow::core::TDesc(idesc.shape, holoflow::core::DType::CF32, idesc.mem_loc);
  }

  std::vector<holoflow::core::InPlace> in_place;
  // if (idesc.dtype == holoflow::core::DType::CF32) {
  //   in_place = {{0, 0}};
  // }

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
  auto        infer    = this->infer(input_descs, jsettings);
  auto        settings = jsettings.get<FresnelDiffractionSettings>();
  const auto &idesc    = input_descs[0];
  const auto &odesc    = infer.output_descs[0];
  const int   rank     = static_cast<int>(idesc.rank());
  auto [ax0, ax1]      = normalize_axes(settings, rank);

  const int H = static_cast<int>(idesc.shape[ax0]);
  const int W = static_cast<int>(idesc.shape[ax1]);

  const bool   is_real       = (idesc.dtype == holoflow::core::DType::F32);
  const size_t in_elem_size  = is_real ? sizeof(float) : sizeof(cuFloatComplex);
  const size_t out_elem_size = sizeof(cuFloatComplex);

  auto in_strides_bytes  = get_strides_bytes(idesc);
  auto out_strides_bytes = get_strides_bytes(odesc);

  check(in_strides_bytes[ax0] % in_strides_bytes[ax1] == 0,
        "input strides for the selected axes are incompatible");
  check(out_strides_bytes[ax0] % out_strides_bytes[ax1] == 0,
        "output strides for the selected axes are incompatible");

  long long int in_istride   = static_cast<long long int>(in_strides_bytes[ax1] / in_elem_size);
  long long int in_stride_h  = static_cast<long long int>(in_strides_bytes[ax0] / in_elem_size);
  long long int inembed_h    = in_stride_h / in_istride;
  long long int transform_in = in_stride_h * H;

  long long int out_istride   = static_cast<long long int>(out_strides_bytes[ax1] / out_elem_size);
  long long int out_stride_h  = static_cast<long long int>(out_strides_bytes[ax0] / out_elem_size);
  long long int onembed_h     = out_stride_h / out_istride;
  long long int transform_out = out_stride_h * H;

  std::vector<int> outer_dims;
  outer_dims.reserve(static_cast<size_t>(rank) - 2);
  for (int i = 0; i < rank; ++i) {
    if (i != ax0 && i != ax1) {
      outer_dims.push_back(i);
    }
  }

  auto best_group =
      select_best_batch_group(idesc.shape, in_strides_bytes, out_strides_bytes, outer_dims,
                              in_elem_size, out_elem_size, transform_in, transform_out);

  std::vector<int> launch_dims;
  launch_dims.reserve(outer_dims.size());
  for (int dim : outer_dims) {
    if (std::find(best_group.dims.begin(), best_group.dims.end(), dim) == best_group.dims.end()) {
      launch_dims.push_back(dim);
    }
  }

  std::vector<LaunchOffset> offsets;
  generate_offsets_recursive(idesc.shape, in_strides_bytes, out_strides_bytes, launch_dims, 0, 0, 0,
                             offsets);
  if (offsets.empty()) {
    offsets.push_back({0, 0});
  }

  auto d_lens = make_quadratic_lens(W, H, settings);

  ApplyLensCallerInfo info{
      .width    = static_cast<unsigned int>(W),
      ._padding = 0,
      .idist    = static_cast<unsigned long long>(best_group.in_idist),
      .stride_h = static_cast<unsigned long long>(in_stride_h),
      .istride  = static_cast<unsigned long long>(in_istride),
      .lens     = d_lens.get(),
  };

  auto d_info = curaii::make_unique_device_ptr<ApplyLensCallerInfo>(1);
  CUDA_CHECK(
      cudaMemcpyAsync(d_info.get(), &info, sizeof(info), cudaMemcpyHostToDevice, ctx.stream));

  bool is_fast = (in_istride == 1 && inembed_h == W);
  auto lto     = apply_lens_lto(is_fast, is_real);

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));

  size_t        work_size     = 0;
  long long int n[2]          = {static_cast<long long int>(H), static_cast<long long int>(W)};
  long long int inembed[2]    = {static_cast<long long int>(H), inembed_h};
  long long int onembed[2]    = {static_cast<long long int>(H), onembed_h};
  cudaDataType  inputtype     = CUDA_C_32F;
  cudaDataType  outputtype    = CUDA_C_32F;
  cudaDataType  executiontype = CUDA_C_32F;

  auto *d_info_ptr = reinterpret_cast<void *>(d_info.get());
  CUFFT_CHECK(cufftXtSetJITCallback(plan.get(), "apply_lens_callback2", lto.data(), lto.size(),
                                    CUFFT_CB_LD_COMPLEX, &d_info_ptr));

  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), 2, n, inembed, in_istride, best_group.in_idist,
                                  inputtype, onembed, out_istride, best_group.out_idist, outputtype,
                                  static_cast<long long int>(best_group.size), &work_size,
                                  executiontype));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FresnelDiffraction(settings, idesc, std::move(plan), std::move(offsets), best_group.size,
                             H, W, best_group.out_idist, out_stride_h, out_istride, ctx.stream,
                             std::move(d_lens), std::move(d_info), std::move(lto)));
}

std::unique_ptr<holoflow::core::ISyncTask>
FresnelDiffractionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                  std::span<const holoflow::core::TDesc>     input_descs,
                                  const nlohmann::json                      &jsettings,
                                  const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old_fresnel = dynamic_cast<FresnelDiffraction *>(old_task.get());
  if (old_fresnel != nullptr && input_descs.size() == 1) {
    const auto &new_idesc = input_descs[0];
    const auto &old_idesc = old_fresnel->get_idesc();
    auto        settings  = jsettings.get<FresnelDiffractionSettings>();

    bool can_reuse =
        (settings == old_fresnel->get_settings()) && (new_idesc.shape == old_idesc.shape) &&
        (new_idesc.strides == old_idesc.strides) && (new_idesc.dtype == old_idesc.dtype) &&
        (new_idesc.mem_loc == old_idesc.mem_loc);

    if (can_reuse) {
      old_fresnel->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
