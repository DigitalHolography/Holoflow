#include "registration.hh"

#include <cub/cub.cuh>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

using curaii::make_unique_device_ptr;
using curaii::make_unique_host_ptr;

namespace holovibes::tasks::syncs {

void to_json(nlohmann::json &j, const RegistrationSettings &s) {
  j = nlohmann::json{{"radius", s.radius}};
}

void from_json(const nlohmann::json &j, RegistrationSettings &s) {
  j.at("radius").get_to(s.radius);
}

namespace {

__global__ void cf32_conjugate_kernel(cuFloatComplex *odata, const cuFloatComplex *idata,
                                      int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  auto c     = idata[idx];
  c.y        = -c.y;
  odata[idx] = c;
}

__global__ void cf32_hadamard_kernel(cuFloatComplex *odata, const cuFloatComplex *idata1,
                                     const cuFloatComplex *idata2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = cuCmulf(idata1[idx], idata2[idx]);
}

__global__ void cf32_normalize_kernel(cuFloatComplex *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  cuFloatComplex val = data[idx];
  float          mag = sqrtf(val.x * val.x + val.y * val.y);
  if (mag > 1e-12f) {
    data[idx].x = val.x / mag;
    data[idx].y = val.y / mag;
  } else {
    data[idx].x = 0.0f;
    data[idx].y = 0.0f;
  }
}

__global__ void f32_shift_subpixel_kernel(float *odata, const float *idata, float shift_x,
                                          float shift_y, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  float src_x = static_cast<float>(x) + shift_x;
  float src_y = static_cast<float>(y) + shift_y;

  int x0 = static_cast<int>(floor(src_x));
  int y0 = static_cast<int>(floor(src_y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float dx = src_x - static_cast<float>(x0);
  float dy = src_y - static_cast<float>(y0);

  int dst_idx = y * w + x;

  auto wrap_coord = [](int coord, int size) {
    coord = coord % size;
    if (coord < 0)
      coord += size;
    return coord;
  };

  int x0_wrap = wrap_coord(x0, w);
  int x1_wrap = wrap_coord(x1, w);
  int y0_wrap = wrap_coord(y0, h);
  int y1_wrap = wrap_coord(y1, h);

  float i00 = idata[y0_wrap * w + x0_wrap];
  float i10 = idata[y0_wrap * w + x1_wrap];
  float i01 = idata[y1_wrap * w + x0_wrap];
  float i11 = idata[y1_wrap * w + x1_wrap];

  odata[dst_idx] = (1.0f - dx) * (1.0f - dy) * i00 + dx * (1.0f - dy) * i10 +
                   (1.0f - dx) * dy * i01 + dx * dy * i11;
}

__device__ bool in_ellipse(int x, int y, int width, int height, float radius) {
  const float cx     = 0.5f * (width - 1);
  const float cy     = 0.5f * (height - 1);
  const float minDim = static_cast<float>(min(width, height));
  const float r      = radius * 0.5f * minDim;
  const float r2     = r * r;
  const float sx     = minDim / static_cast<float>(width);
  const float sy     = minDim / static_cast<float>(height);
  const float dx     = static_cast<float>(x) - cx;
  const float dy     = static_cast<float>(y) - cy;
  const float dxs    = dx * sx;
  const float dys    = dy * sy;
  return (dxs * dxs + dys * dys) <= r2;
}

__global__ void f32_sub_mean_kernel(float *odata, const float *idata, int count, float mean) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  odata[idx] = idata[idx] - mean;
}

__global__ void f32_ellipse_mask_kernel(float *odata, const float *idata, int width, int height,
                                        float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const bool   inside = in_ellipse(x, y, width, height, radius);
  const size_t idx    = static_cast<size_t>(y) * width + static_cast<size_t>(x);
  odata[idx]          = inside ? idata[idx] : 0.0f;
}

__global__ void ellipse_mask_kernel(uint8_t *oroi, int width, int height, float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const bool   inside = in_ellipse(x, y, width, height, radius);
  const size_t idx    = static_cast<size_t>(y) * width + static_cast<size_t>(x);
  oroi[idx]           = inside;
}

__global__ void extract_3x3_kernel(float *d_output, const float *d_input, int peak_x, int peak_y,
                                   int w, int h) {
  int idx = threadIdx.y * 3 + threadIdx.x;
  if (threadIdx.x < 3 && threadIdx.y < 3) {
    int dx = threadIdx.x - 1;
    int dy = threadIdx.y - 1;

    int x = max(0, min(w - 1, peak_x + dx));
    int y = max(0, min(h - 1, peak_y + dy));

    d_output[idx] = d_input[y * w + x];
  }
}

} // namespace

Registration::Registration(RegistrationSettings settings, const holoflow::core::TDesc &input_desc,
                           const holoflow::core::TDesc &output_desc, cudaStream_t stream,
                           DevPtr<float> d_mean_centered, bool ref_initialized, size_t freq_size,
                           curaii::CufftHandle r2c_handle, curaii::CufftHandle c2r_handle,
                           DevPtr<float> d_ref, DevPtr<float> d_xcorr,
                           DevPtr<cuFloatComplex> d_freq1, DevPtr<cuFloatComplex> d_freq2,
                           size_t sum_tmp_bytes, DevPtr<uint8_t> d_sum_tmp, DevPtr<float> d_sum,
                           size_t amax_tmp_bytes, DevPtr<uint8_t> d_amax_tmp, DevPtr<float> d_max,
                           DevPtr<int64_t> d_max_idx, size_t select_tmp_bytes,
                           DevPtr<uint8_t> d_select_tmp, DevPtr<int> d_select_count,
                           DevPtr<uint8_t> d_select_roi, DevPtr<float> d_selected)
    : settings_(std::move(settings)), input_desc_(input_desc), output_desc_(output_desc),
      stream_(stream), d_mean_centered_(std::move(d_mean_centered)),
      ref_initialized_(ref_initialized), freq_size_(freq_size), r2c_handle_(std::move(r2c_handle)),
      c2r_handle_(std::move(c2r_handle)), d_ref_(std::move(d_ref)), d_xcorr_(std::move(d_xcorr)),
      d_freq1_(std::move(d_freq1)), d_freq2_(std::move(d_freq2)), sum_tmp_bytes_(sum_tmp_bytes),
      d_sum_tmp_(std::move(d_sum_tmp)), d_sum_(std::move(d_sum)), amax_tmp_bytes_(amax_tmp_bytes),
      d_amax_tmp_(std::move(d_amax_tmp)), d_max_(std::move(d_max)),
      d_max_idx_(std::move(d_max_idx)), select_tmp_bytes_(select_tmp_bytes),
      d_select_tmp_(std::move(d_select_tmp)), d_select_count_(std::move(d_select_count)),
      d_select_roi_(std::move(d_select_roi)), d_selected_(std::move(d_selected)) {}

holoflow::core::OpResult Registration::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
    return holoflow::core::OpResult::Cancelled;
  }

  if (ctx.inputs.empty() || ctx.outputs.empty()) {
    return holoflow::core::OpResult::NotReady;
  }

  const auto &input_view  = ctx.inputs[0];
  auto       &output_view = ctx.outputs[0];

  const float *input_data  = reinterpret_cast<const float *>(input_view.data);
  float       *output_data = reinterpret_cast<float *>(output_view.data);

  auto width  = input_desc_.shape.back();
  auto height = input_desc_.shape.size() > 1 ? input_desc_.shape[input_desc_.shape.size() - 2] : 1;
  auto batch  = input_desc_.shape.size() > 2 ? input_desc_.shape[input_desc_.shape.size() - 3] : 1;

  if (batch != 1) {
    return holoflow::core::OpResult::NotReady;
  }

  center_mean(d_mean_centered_.get(), input_data, batch, height, width);

  if (!ref_initialized_) {
    CUDA_CHECK(cudaMemcpyAsync(d_ref_.get(), d_mean_centered_.get(), input_desc_.num_bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
    cudaDeviceSynchronize();
    ref_initialized_ = true;
  }

  xcorr(d_xcorr_.get(), d_mean_centered_.get());
  auto [shift_x, shift_y] = get_shifts_subpixel(d_xcorr_.get(), width, height);

  apply_shifts(output_data, input_data, shift_x, shift_y, batch, height, width);

  return holoflow::core::OpResult::Ok;
}

void Registration::xcorr(float *odata, const float *idata) {
  auto *idata_nc = const_cast<float *>(idata);
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata_nc, d_freq1_.get()));
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), d_ref_.get(), d_freq2_.get()));

  int block_size = 256;
  int grid_size  = (freq_size_ + block_size - 1) / block_size;

  cf32_conjugate_kernel<<<grid_size, block_size, 0, stream_>>>(d_freq2_.get(), d_freq2_.get(),
                                                               static_cast<int>(freq_size_));

  cf32_hadamard_kernel<<<grid_size, block_size, 0, stream_>>>(
      d_freq1_.get(), d_freq2_.get(), d_freq1_.get(), static_cast<int>(freq_size_));

  cf32_normalize_kernel<<<grid_size, block_size, 0, stream_>>>(d_freq1_.get(),
                                                               static_cast<int>(freq_size_));

  CUFFT_CHECK(cufftExecC2R(c2r_handle_.get(), d_freq1_.get(), odata));
  CUDA_CHECK(cudaGetLastError());
}

void Registration::mask_circle(float *odata, const float *idata, std::size_t b, std::size_t h,
                               std::size_t w) {
  if (b != 1)
    return;

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y);

  f32_ellipse_mask_kernel<<<grid_size, block_size, 0, stream_>>>(
      odata, idata, static_cast<int>(w), static_cast<int>(h), settings_.radius);
}

void Registration::center_mean(float *odata, const float *idata, std::size_t b, std::size_t h,
                               std::size_t w) {
  if (b != 1)
    return;

  const size_t num_pixels = w * h;

  CUDA_CHECK(cudaMemsetAsync(d_selected_.get(), 0, num_pixels * sizeof(float), stream_));

  CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp_.get(), select_tmp_bytes_, idata,
                                        d_select_roi_.get(), d_selected_.get(),
                                        d_select_count_.get(), num_pixels, stream_));

  int select_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&select_count, d_select_count_.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  if (select_count == 0)
    return;

  CUDA_CHECK(cub::DeviceReduce::Sum(d_sum_tmp_.get(), sum_tmp_bytes_, d_selected_.get(),
                                    d_sum_.get(), select_count, stream_));

  float sum_val = 0.0f;
  CUDA_CHECK(
      cudaMemcpyAsync(&sum_val, d_sum_.get(), sizeof(float), cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  float mean = sum_val / static_cast<float>(select_count);

  const int    block_dim = 256;
  const size_t grid_dim  = (num_pixels + block_dim - 1) / block_dim;

  f32_sub_mean_kernel<<<static_cast<unsigned int>(grid_dim), block_dim, 0, stream_>>>(
      odata, idata, static_cast<int>(num_pixels), mean);

  CUDA_CHECK(cudaGetLastError());
}

std::pair<int64_t, int64_t> Registration::get_shifts(float *xcorr, std::size_t b, std::size_t h,
                                                     std::size_t w) {
  if (b != 1)
    return {0, 0};

  CUDA_CHECK(cub::DeviceReduce::ArgMax(d_amax_tmp_.get(), amax_tmp_bytes_, xcorr, d_max_.get(),
                                       d_max_idx_.get(), w * h, stream_));

  int64_t h_max_idx = 0;
  CUDA_CHECK(cudaMemcpyAsync(&h_max_idx, d_max_idx_.get(), sizeof(int64_t), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  int64_t x = h_max_idx % w;
  int64_t y = h_max_idx / w;

  int half_h = static_cast<int>(h) / 2;
  int half_w = static_cast<int>(w) / 2;

  int dy = (y < half_h) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(h);
  int dx = (x < half_w) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(w);

  return {dx, dy};
}

void Registration::apply_shifts(float *odata, const float *idata, float shift_x, float shift_y,
                                std::size_t b, std::size_t h, std::size_t w) {
  if (b != 1)
    return;

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y);

  f32_shift_subpixel_kernel<<<grid_size, block_size, 0, stream_>>>(
      odata, idata, shift_x, shift_y, static_cast<int>(w), static_cast<int>(h));
}

std::pair<float, float> Registration::get_shifts_subpixel(float *xcorr, std::size_t w,
                                                          std::size_t h) {
  auto [shift_x, shift_y] = get_shifts(xcorr, 1, h, w);

  int peak_x = static_cast<int>((shift_x < 0) ? shift_x + w : shift_x);
  int peak_y = static_cast<int>((shift_y < 0) ? shift_y + h : shift_y);

  float *d_samples;
  float  h_samples[9];
  CUDA_CHECK(cudaMalloc(&d_samples, 9 * sizeof(float)));

  dim3 block_size(3, 3);
  extract_3x3_kernel<<<1, block_size, 0, stream_>>>(d_samples, xcorr, peak_x, peak_y,
                                                    static_cast<int>(w), static_cast<int>(h));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(
      cudaMemcpyAsync(h_samples, d_samples, 9 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaFree(d_samples));

  float refined_dx = (h_samples[5] - h_samples[3]) /
                     (2.0f * (h_samples[3] + h_samples[5] - 2.0f * h_samples[4] + 1e-12f));
  float refined_dy = (h_samples[7] - h_samples[1]) /
                     (2.0f * (h_samples[1] + h_samples[7] - 2.0f * h_samples[4] + 1e-12f));

  return {shift_x + refined_dx, shift_y + refined_dy};
}

holoflow::core::InferResult
RegistrationFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings) const {
  const auto check = [](bool condition, const char *message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };

  check(!input_descs.empty(), "No input descriptors provided");
  check(input_descs.size() == 1, "Registration expects exactly one input");

  const auto &input_desc = input_descs[0];
  auto        settings   = jsettings.get<RegistrationSettings>();

  check(input_desc.rank() >= 2, "Input must have at least 2 dimensions");
  check(input_desc.dtype == holoflow::core::DType::F32, "Input must be F32 type");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "Input must be in device memory");
  check(settings.radius >= 0.0f && settings.radius <= 1.0f, "radius not in [0, 1]");

  holoflow::core::TDesc output_desc = input_desc;

  return holoflow::core::InferResult{
      .input_descs   = {input_desc},
      .output_descs  = {output_desc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
RegistrationFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings,
                            const holoflow::core::SyncCreateCtx   &ctx) const {
  logger()->info("Creating Registration sync task");
  auto result   = infer(input_descs, jsettings);
  auto settings = jsettings.get<RegistrationSettings>();

  const auto &input_desc = input_descs[0];

  const auto H = input_desc.shape.size() > 1 ? input_desc.shape[input_desc.shape.size() - 2] : 1;
  const auto W = input_desc.shape.back();

  // Center mean
  auto d_mean_centered = make_unique_device_ptr<float>(W * H);

  // Cross correlation
  bool   ref_initialized = false;
  size_t freq_size       = H * W;

  auto r2c_handle = curaii::CufftHandle();
  auto c2r_handle = curaii::CufftHandle();

  size_t r2c_ws = 0;
  size_t c2r_ws = 0;
  CUFFT_CHECK(cufftMakePlan2d(r2c_handle.get(), static_cast<int>(W), static_cast<int>(H), CUFFT_R2C,
                              &r2c_ws));
  CUFFT_CHECK(cufftMakePlan2d(c2r_handle.get(), static_cast<int>(W), static_cast<int>(H), CUFFT_C2R,
                              &c2r_ws));
  CUFFT_CHECK(cufftSetStream(r2c_handle.get(), ctx.stream));
  CUFFT_CHECK(cufftSetStream(c2r_handle.get(), ctx.stream));

  auto d_ref   = make_unique_device_ptr<float>(W * H);
  auto d_xcorr = make_unique_device_ptr<float>(W * H);
  auto d_freq1 = make_unique_device_ptr<cuFloatComplex>(freq_size);
  auto d_freq2 = make_unique_device_ptr<cuFloatComplex>(freq_size);

  // CUB argmax
  size_t amax_tmp_bytes = 0;
  auto   d_max          = make_unique_device_ptr<float>(1);
  auto   d_max_idx      = make_unique_device_ptr<int64_t>(1);
  CUDA_CHECK(cub::DeviceReduce::ArgMax((void *)nullptr, amax_tmp_bytes, (float *)nullptr,
                                       d_max.get(), d_max_idx.get(), W * H, ctx.stream));
  auto d_amax_tmp = make_unique_device_ptr<uint8_t>(amax_tmp_bytes);

  // CUB select
  size_t select_tmp_bytes = 0;
  auto   d_select_count   = make_unique_device_ptr<int>(1);
  auto   d_select_roi     = make_unique_device_ptr<uint8_t>(W * H);
  auto   d_selected       = make_unique_device_ptr<float>(W * H);

  dim3 block_size(16, 16);
  dim3 grid_size((W + block_size.x - 1) / block_size.x, (H + block_size.y - 1) / block_size.y);

  ellipse_mask_kernel<<<grid_size, block_size, 0, ctx.stream>>>(
      d_select_roi.get(), static_cast<int>(W), static_cast<int>(H), settings.radius);
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cub::DeviceSelect::Flagged(static_cast<void *>(nullptr), select_tmp_bytes,
                                        static_cast<float *>(nullptr), d_select_roi.get(),
                                        d_selected.get(), d_select_count.get(), W * H, ctx.stream));
  auto d_select_tmp = make_unique_device_ptr<uint8_t>(select_tmp_bytes);

  // CUB sum
  CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp.get(), select_tmp_bytes, d_mean_centered.get(),
                                        d_select_roi.get(), d_selected.get(), d_select_count.get(),
                                        W * H, ctx.stream));

  int select_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&select_count, d_select_count.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, ctx.stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

  size_t sum_tmp_bytes = 0;
  auto   d_sum         = make_unique_device_ptr<float>(1);
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, sum_tmp_bytes, static_cast<float *>(nullptr),
                                    d_sum.get(), select_count, ctx.stream));
  auto d_sum_tmp = make_unique_device_ptr<uint8_t>(sum_tmp_bytes);

  // Assemble task
  return std::unique_ptr<holoflow::core::ISyncTask>(new Registration(
      settings, input_desc, result.output_descs[0], ctx.stream, std::move(d_mean_centered),
      ref_initialized, freq_size, std::move(r2c_handle), std::move(c2r_handle), std::move(d_ref),
      std::move(d_xcorr), std::move(d_freq1), std::move(d_freq2), sum_tmp_bytes,
      std::move(d_sum_tmp), std::move(d_sum), amax_tmp_bytes, std::move(d_amax_tmp),
      std::move(d_max), std::move(d_max_idx), select_tmp_bytes, std::move(d_select_tmp),
      std::move(d_select_count), std::move(d_select_roi), std::move(d_selected)));
}

} // namespace holovibes::tasks::syncs