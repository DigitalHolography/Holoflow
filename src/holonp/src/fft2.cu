#include "holonp/fft2.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const FFT2Settings &s) {
  j = nlohmann::json{
      {"axes", s.axes},
      {"norm", s.norm},
  };
}

void from_json(const nlohmann::json &j, FFT2Settings &s) {
  if (j.contains("axes")) {
    j.at("axes").get_to(s.axes);
  } else {
    s.axes.clear();
  }

  if (j.contains("norm")) {
    j.at("norm").get_to(s.norm);
  } else {
    s.norm = FftNorm::Backward;
  }
}

namespace {

constexpr int kMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("FFT2: " + msg);
  }
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

inline std::array<int, 2> normalize_axes(const std::vector<int> &axes, int ndim) {
  std::array<int, 2> out{};
  if (axes.empty()) {
    out = {ndim - 2, ndim - 1};
  } else {
    check(axes.size() == 2, "axes must have length 2");
    out = {axes[0], axes[1]};
  }

  for (auto &a : out) {
    if (a < 0) {
      a += ndim;
    }
    check(a >= 0 && a < ndim, "axis out of range");
  }

  check(out[0] != out[1], "axes must be distinct");

  std::array<int, 2> sorted = out;
  std::sort(sorted.begin(), sorted.end());
  check(sorted[0] == ndim - 2 && sorted[1] == ndim - 1,
        "only the last two axes are supported for now");
  return out;
}

inline float norm_scale(FftNorm norm, size_t n_fft) {
  if (norm == FftNorm::Backward) {
    return 1.0f;
  }
  const double n = static_cast<double>(n_fft);
  if (norm == FftNorm::Forward) {
    return static_cast<float>(1.0 / n);
  }
  return static_cast<float>(1.0 / std::sqrt(n));
}

__global__ void real_to_complex_kernel(const float *__restrict__ in,
                                       cuFloatComplex *__restrict__ out, std::int64_t n) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    out[idx] = make_cuFloatComplex(in[idx], 0.0f);
  }
}

__global__ void scale_cf32_kernel(cuFloatComplex *__restrict__ data, std::int64_t n, float scale) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

} // namespace

FFT2::FFT2(const FFT2Settings &settings, curaii::CufftHandle &&plan, size_t total_elems,
           size_t n_fft, holoflow::core::DType input_dtype, cudaStream_t stream,
           DevPtr<cuFloatComplex> d_tmp)
    : settings_(settings), plan_(std::move(plan)), total_elems_(total_elems), n_fft_(n_fft),
      input_dtype_(input_dtype), stream_(stream), d_tmp_(std::move(d_tmp)) {}

holoflow::core::OpResult FFT2::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  (void)idesc;
  (void)odesc;

  const auto total_i64 = static_cast<std::int64_t>(total_elems_);
  const int  block     = 256;
  const int  grid      = static_cast<int>((total_i64 + block - 1) / block);

  const cuFloatComplex *in = nullptr;
  if (input_dtype_ == holoflow::core::DType::F32) {
    auto *in_f = reinterpret_cast<const float *>(idata);
    real_to_complex_kernel<<<grid, block, 0, stream_>>>(in_f, d_tmp_.get(), total_i64);
    in = d_tmp_.get();
  } else {
    in = reinterpret_cast<const cuFloatComplex *>(idata);
  }

  auto *out = reinterpret_cast<cuFloatComplex *>(odata);
  CUFFT_CHECK(cufftXtExec(plan_.get(), const_cast<cuFloatComplex *>(in), out, CUFFT_FORWARD));

  const float scale = norm_scale(settings_.norm, n_fft_);
  if (scale != 1.0f) {
    scale_cf32_kernel<<<grid, block, 0, stream_>>>(out, total_i64, scale);
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult FFT2Factory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<FFT2Settings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "input dtype must be F32 or CF32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim >= 2, "input ndim must be >= 2");
  check(ndim <= kMaxNDim, "input ndim too large");

  (void)normalize_axes(settings.axes, ndim);

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  holoflow::core::TDesc odesc = idesc;
  odesc.dtype                 = holoflow::core::DType::CF32;

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FFT2Factory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  const auto settings  = jsettings.get<FFT2Settings>();
  (void)infer_res;

  const auto &idesc = input_descs[0];
  const auto  total = product_shape(idesc.shape);
  const int   ndim  = static_cast<int>(idesc.shape.size());

  (void)normalize_axes(settings.axes, ndim);

  const auto n0    = idesc.shape[static_cast<size_t>(ndim - 2)];
  const auto n1    = idesc.shape[static_cast<size_t>(ndim - 1)];
  const auto n_fft = n0 * n1;

  check(n0 > 0 && n1 > 0, "invalid FFT dimensions");
  check(n0 <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT dimension exceeds cuFFT limits");
  check(n1 <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT dimension exceeds cuFFT limits");
  check(n_fft <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT size exceeds cuFFT limits");
  check(total % n_fft == 0, "input shape is not contiguous along the FFT axes");

  const auto batch = total / n_fft;
  check(batch <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "FFT batch size exceeds cuFFT limits");

  int           rank          = 2;
  long long int n[2]          = {static_cast<long long int>(n0), static_cast<long long int>(n1)};
  long long int inembed[2]    = {static_cast<long long int>(n0), static_cast<long long int>(n1)};
  int           istride       = 1;
  int           idist         = static_cast<int>(n_fft);
  cudaDataType  inputtype     = CUDA_C_32F;
  long long int onembed[2]    = {static_cast<long long int>(n0), static_cast<long long int>(n1)};
  int           ostride       = 1;
  int           odist         = static_cast<int>(n_fft);
  cudaDataType  outputtype    = CUDA_C_32F;
  int           batch_i       = static_cast<int>(batch);
  size_t        work_size     = 0;
  cudaDataType  executiontype = CUDA_C_32F;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), rank, n, inembed, istride, idist, inputtype, onembed,
                                  ostride, odist, outputtype, batch_i, &work_size, executiontype));

  DevPtr<cuFloatComplex> d_tmp;
  if (idesc.dtype == holoflow::core::DType::F32) {
    d_tmp = curaii::make_unique_device_ptr<cuFloatComplex>(total, ctx.stream);
  }

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFT2(settings, std::move(plan), total, n_fft, idesc.dtype, ctx.stream, std::move(d_tmp)));
}

} // namespace holonp
