#include "holovibes/tasks/registration.hh"

#include <cub/cub.cuh>

#include "holovibes/holovibes.hh"

namespace holovibes::tasks {

// ==========================================================================
//                     Registration Implementation
// ==========================================================================

namespace {

__global__ void cf32_conjugate_kernel(cuFloatComplex       *odata,
                                      const cuFloatComplex *idata, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  auto c     = idata[idx];
  c.y        = -c.y;
  odata[idx] = c;
}

__global__ void cf32_hadamard_kernel(cuFloatComplex       *odata,
                                     const cuFloatComplex *idata1,
                                     const cuFloatComplex *idata2, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  odata[idx] = cuCmulf(idata1[idx], idata2[idx]);
}

__global__ void f32_shift_kernel(float *odata, const float *idata,
                                 int64_t shift_x, int64_t shift_y, int w,
                                 int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int src_x   = x - shift_x;
  int src_y   = y - shift_y;
  int dst_idx = y * w + x;

  if (src_x < 0 || src_x >= w || src_y < 0 || src_y >= h) {
    odata[dst_idx] = 0.0f;
  } else {
    int src_idx    = src_y * w + src_x;
    odata[dst_idx] = idata[src_idx];
  }
}

__device__ bool in_ellipse(int x, int y, int width, int height, float radius) {
  // center of the image
  const float cx = 0.5f * (width - 1);
  const float cy = 0.5f * (height - 1);

  // side length of the eventual square
  const float minDim = static_cast<float>(min(width, height));

  // radius (in pixels) of the circle in that square
  const float r  = radius * 0.5f * minDim;
  const float r2 = r * r;

  // how the rectangle will be squeezed/stretched to a square
  const float sx = minDim / static_cast<float>(width);
  const float sy = minDim / static_cast<float>(height);

  // offset from center, then pre-scale
  const float dx  = static_cast<float>(x) - cx;
  const float dy  = static_cast<float>(y) - cy;
  const float dxs = dx * sx;
  const float dys = dy * sy;

  // ellipse test in pre-scaled space
  return (dxs * dxs + dys * dys) <= r2;
}

__global__ void f32_sub_mean_kernel(float *odata, const float *idata, int count,
                                    float *sum, int in_roi) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  __shared__ float mean;
  if (threadIdx.x == 0) {
    mean = (*sum) / static_cast<float>(in_roi);
  }
  __syncthreads();

  odata[idx] = idata[idx] - mean;
}

__global__ void f32_ellipse_mask_kernel(float *odata, const float *idata,
                                        int width, int height, float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const bool   inside = in_ellipse(x, y, width, height, radius);
  const size_t idx    = static_cast<size_t>(y) * width + static_cast<size_t>(x);
  odata[idx]          = inside ? idata[idx] : 0.0f;
}

} // namespace

Registration::Registration(
    const dh::TaskMeta &meta, cudaStream_t stream,
    // Config
    float radius,
    // Center mean
    DevPtr<float> d_mean_centered,
    // Cross correlation
    bool ref_initialized, size_t freq_size, CufftHandle r2c_handle,
    CufftHandle c2r_handle, DevPtr<float> d_ref, DevPtr<float> d_xcorr,
    DevPtr<cuFloatComplex> d_freq1, DevPtr<cuFloatComplex> d_freq2,
    // CUB sum
    size_t sum_tmp_bytes, DevPtr<uint8_t> d_sum_tmp, DevPtr<float> d_sum,
    // CUB argmax
    size_t amax_tmp_bytes, DevPtr<uint8_t> d_amax_tmp, DevPtr<float> d_max,
    DevPtr<int64_t> d_max_idx,
    // CUB select
    size_t select_tmp_bytes, DevPtr<uint8_t> d_select_tmp,
    DevPtr<int> d_select_count, DevPtr<uint8_t> d_select_roi,
    DevPtr<float> d_selected)
    : dh::Task(meta, stream), radius_(radius),
      d_mean_centered_(std::move(d_mean_centered)),
      ref_initialized_(ref_initialized), freq_size_(freq_size),
      r2c_handle_(std::move(r2c_handle)), c2r_handle_(std::move(c2r_handle)),
      d_ref_(std::move(d_ref)), d_xcorr_(std::move(d_xcorr)),
      d_freq1_(std::move(d_freq1)), d_freq2_(std::move(d_freq2)),
      sum_tmp_bytes_(sum_tmp_bytes), d_sum_tmp_(std::move(d_sum_tmp)),
      d_sum_(std::move(d_sum)), amax_tmp_bytes_(amax_tmp_bytes),
      d_amax_tmp_(std::move(d_amax_tmp)), d_max_(std::move(d_max)),
      d_max_idx_(std::move(d_max_idx)), select_tmp_bytes_(select_tmp_bytes),
      d_select_tmp_(std::move(d_select_tmp)),
      d_select_count_(std::move(d_select_count)),
      d_select_roi_(std::move(d_select_roi)),
      d_selected_(std::move(d_selected)) {}

void Registration::run(dh::TensorView input, dh::TensorView output) {
  size_t w     = imeta().shape().at(2);
  size_t h     = imeta().shape().at(1);
  auto  *idata = static_cast<float *>(input.data());
  auto  *odata = static_cast<float *>(output.data());

  // mask_circle(d_mean_centered_.get(), idata);
  center_mean(d_mean_centered_.get(), idata);
  mask_circle(d_mean_centered_.get(), d_mean_centered_.get());

  // The reference image is the first frame.
  if (!ref_initialized_) {
    ref_initialized_ = true;
    CUDA_CHECK(cudaMemcpyAsync(d_ref_.get(), d_mean_centered_.get(),
                               input.size_in_bytes(), cudaMemcpyDeviceToDevice,
                               stream_));
  }

  // apply_shifts(d_mean_centered_.get(), idata, 0, 1);
  // center_mean(d_mean_centered_.get(), d_mean_centered_.get());
  // mask_circle(d_mean_centered_.get(), d_mean_centered_.get());

  cross_correlation(d_xcorr_.get(), d_mean_centered_.get());
  auto [shift_x, shift_y] = get_shifts(d_xcorr_.get());
  apply_shifts(odata, idata, shift_x, shift_y);

  // apply_shifts(odata, d_ref_.get(), 0, 0);

  // CUDA_CHECK(cudaMemcpyAsync(d_ref_.get(), d_mean_centered_.get(),
  //                            input.size_in_bytes(), cudaMemcpyDeviceToDevice,
  //                            stream_));
  // cudaMemcpyAsync(odata, d_ref_.get(), imeta().size_in_bytes(),
  //                 cudaMemcpyDeviceToDevice, stream_);
}

void Registration::cross_correlation(float *odata, const float *idata) {
  auto *idata_nc = const_cast<float *>(idata);
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata_nc, d_freq1_.get()));
  CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), d_ref_.get(), d_freq2_.get()));
  // CUFFT_CHECK(cufftExecR2C(r2c_handle_.get(), idata_nc, d_freq2_.get()));

  int block_size = 256;
  int grid_size  = (freq_size_ + block_size - 1) / block_size;
  cf32_conjugate_kernel<<<grid_size, block_size, 0, stream_>>>(
      d_freq2_.get(), d_freq2_.get(), freq_size_);
  cf32_hadamard_kernel<<<grid_size, block_size, 0, stream_>>>(
      d_freq1_.get(), d_freq2_.get(), d_freq1_.get(), freq_size_);

  CUFFT_CHECK(cufftExecC2R(c2r_handle_.get(), d_freq1_.get(), odata));
}

void Registration::mask_circle(float *odata, const float *idata) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x,
                 (h + block_size.y - 1) / block_size.y);

  f32_ellipse_mask_kernel<<<grid_size, block_size, 0, stream_>>>(odata, idata,
                                                                 w, h, radius_);
}

void Registration::center_mean(float *odata, const float *idata) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  CUDA_CHECK(cub::DeviceSelect::Flagged(
      d_select_tmp_.get(), select_tmp_bytes_, idata, d_select_roi_.get(),
      d_selected_.get(), d_select_count_.get(), w * h, stream_));

  int select_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&select_count, d_select_count_.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  CUDA_CHECK(cub::DeviceReduce::Sum(d_sum_tmp_.get(), sum_tmp_bytes_,
                                    d_selected_.get(), d_sum_.get(),
                                    select_count, stream_));

  int block_dim = 256;
  int grid_dim  = (w * h + block_dim - 1) / block_dim;
  f32_sub_mean_kernel<<<grid_dim, block_dim, 0, stream_>>>(
      odata, idata, w * h, d_sum_.get(), select_count);
}

std::pair<int64_t, int64_t> Registration::get_shifts(float *xcorr) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  CUDA_CHECK(cub::DeviceReduce::ArgMax(d_amax_tmp_.get(), amax_tmp_bytes_,
                                       xcorr, d_max_.get(), d_max_idx_.get(),
                                       w * h, stream_));

  int64_t h_max_idx = 0;
  CUDA_CHECK(cudaMemcpyAsync(&h_max_idx, d_max_idx_.get(), sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  int64_t x = h_max_idx % w;
  int64_t y = h_max_idx / w;

  // FFT places the zero-shift component first (0, 0) instead of the center. To
  // get proper cross-correlation, one would need to perform fftshift. We only
  // need to do it on the found coordinate.
  int half_h = h / 2;
  int half_w = w / 2;

  int dy = (y < half_h) ? y : y - h;
  int dx = (x < half_w) ? x : x - w;

  return {-dx, -dy};
}

void Registration::apply_shifts(float *odata, const float *idata,
                                int64_t shift_x, int64_t shift_y) {
  size_t w = imeta().shape().at(2);
  size_t h = imeta().shape().at(1);

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x,
                 (h + block_size.y - 1) / block_size.y);

  f32_shift_kernel<<<grid_size, block_size, 0, stream_>>>(odata, idata, shift_x,
                                                          shift_y, w, h);
}

// ==========================================================================
//                     RegistrationFactory Implementation
// ==========================================================================

namespace {

__global__ void ellipse_mask_kernel(uint8_t *oroi, int width, int height,
                                    float radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const bool   inside = in_ellipse(x, y, width, height, radius);
  const size_t idx    = static_cast<size_t>(y) * width + static_cast<size_t>(x);
  oroi[idx]           = inside;
}

} // namespace

dh::TaskMeta RegistrationFactory::type_check(const dh::TensorMeta &imeta,
                                             const json           &jparams) {
  // Unpack parameters
  const auto params = jparams.get<RegistrationParams>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // Parameter sanity
  check(params.radius >= 0.0f && params.radius <= 1.0f, "radius not in [0, 1]");

  // Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.shape().at(0) == 1, "tensor dim 0 != 1");
  check(imeta.data_type() == dh::DataType::F32, "tensor data_type != F32");
  check(imeta.memory_location() == dh::MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  // Success
  return dh::TaskMeta(imeta, imeta, false);
}

std::unique_ptr<dh::Task>
RegistrationFactory::create(const dh::TensorMeta &imeta, const json &jparams,
                            cudaStream_t stream) {
  using namespace curaii::cuda;

  // Validate
  auto meta   = type_check(imeta, jparams);
  auto params = jparams.get<RegistrationParams>();
  auto w      = imeta.shape().at(2);
  auto h      = imeta.shape().at(1);

  // Center mean
  auto d_mean_centered = make_unique_device_ptr<float>(w * h);

  // Cross correlation
  bool   ref_initialized = false;
  size_t freq_size       = h * w;
  size_t r2c_ws          = 0;
  size_t c2r_ws          = 0;
  auto   r2c_handle      = curaii::cufft::Handle();
  auto   c2r_handle      = curaii::cufft::Handle();
  CUFFT_CHECK(cufftMakePlan2d(r2c_handle.get(), w, h, CUFFT_R2C, &r2c_ws));
  CUFFT_CHECK(cufftMakePlan2d(c2r_handle.get(), w, h, CUFFT_C2R, &c2r_ws));
  CUFFT_CHECK(cufftSetStream(r2c_handle.get(), stream));
  CUFFT_CHECK(cufftSetStream(c2r_handle.get(), stream));
  auto d_ref   = make_unique_device_ptr<float>(w * h);
  auto d_xcorr = make_unique_device_ptr<float>(w * h);
  auto d_freq1 = make_unique_device_ptr<cuFloatComplex>(freq_size);
  auto d_freq2 = make_unique_device_ptr<cuFloatComplex>(freq_size);

  // CUB argmax
  size_t amax_tmp_bytes = 0;
  auto   d_max          = make_unique_device_ptr<float>(1);
  auto   d_max_idx      = make_unique_device_ptr<int64_t>(1);
  CUDA_CHECK(cub::DeviceReduce::ArgMax((void *)nullptr, amax_tmp_bytes,
                                       (float *)nullptr, d_max.get(),
                                       d_max_idx.get(), w * h, stream));
  auto d_amax_tmp = make_unique_device_ptr<uint8_t>(amax_tmp_bytes);

  // CUB select
  size_t select_tmp_bytes = 0;
  auto   d_select_count   = make_unique_device_ptr<int>(1);
  auto   d_select_roi     = make_unique_device_ptr<uint8_t>(w * h);
  auto   d_selected       = make_unique_device_ptr<float>(w * h);

  dim3 block_size(16, 16);
  dim3 grid_size((w + block_size.x - 1) / block_size.x,
                 (h + block_size.y - 1) / block_size.y);

  ellipse_mask_kernel<<<grid_size, block_size, 0, stream>>>(
      d_select_roi.get(), w, h, params.radius);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cub::DeviceSelect::Flagged(
      (void *)nullptr, select_tmp_bytes, (float *)nullptr, d_select_roi.get(),
      d_selected.get(), d_select_count.get(), w * h, stream));
  auto d_select_tmp = make_unique_device_ptr<uint8_t>(select_tmp_bytes);

  // CUB sum
  CUDA_CHECK(cub::DeviceSelect::Flagged(d_select_tmp.get(), select_tmp_bytes,
                                        d_mean_centered.get(),
                                        d_select_roi.get(), d_selected.get(),
                                        d_select_count.get(), w * h, stream));

  int select_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&select_count, d_select_count.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t sum_tmp_bytes = 0;
  auto   d_sum         = make_unique_device_ptr<float>(1);
  CUDA_CHECK(cub::DeviceReduce::Sum((void *)nullptr, sum_tmp_bytes,
                                    (float *)nullptr, d_sum.get(), select_count,
                                    stream));
  auto d_sum_tmp = make_unique_device_ptr<uint8_t>(sum_tmp_bytes);

  // Assemble task
  auto *task = new Registration(
      meta, stream, params.radius, std::move(d_mean_centered), ref_initialized,
      freq_size, std::move(r2c_handle), std::move(c2r_handle), std::move(d_ref),
      std::move(d_xcorr), std::move(d_freq1), std::move(d_freq2), sum_tmp_bytes,
      std::move(d_sum_tmp), std::move(d_sum), amax_tmp_bytes,
      std::move(d_amax_tmp), std::move(d_max), std::move(d_max_idx),
      select_tmp_bytes, std::move(d_select_tmp), std::move(d_select_count),
      std::move(d_select_roi), std::move(d_selected));

  return std::unique_ptr<Registration>(task);
}

} // namespace holovibes::tasks