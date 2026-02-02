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

constexpr size_t kMaxNDim = 16;

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

// Ensure strides are present and in bytes
inline std::vector<size_t> ensure_strides(const holoflow::core::TDesc &desc) {
  if (!desc.strides.empty()) {
    return desc.strides;
  }
  // Generate contiguous byte strides if missing
  std::vector<size_t> strides(desc.shape.size());
  size_t acc = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= desc.shape[i];
  }
  return strides;
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

__global__ void scale_cf32_kernel(cuFloatComplex *__restrict__ data, size_t n, float scale) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

} // namespace

FFT2::FFT2(const FFT2Settings &settings, curaii::CufftHandle &&plan, size_t n_fft,
           cudaStream_t stream)
    : settings_(settings), plan_(std::move(plan)), n_fft_(n_fft), stream_(stream) {}

holoflow::core::OpResult FFT2::execute(holoflow::core::SyncCtx &ctx) {
  // Get raw pointers.
  // Note: data() includes the offset. 
  // Since we did not use the 'offset' param in cufftXtMakePlanMany (it doesn't have one),
  // we rely on the pointer itself pointing to the start of the data.
  auto *idata_base = reinterpret_cast<uint8_t *>(ctx.inputs[0].data());
  auto *odata      = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());

  // Input is guaranteed CF32 by factory
  auto *fft_in  = const_cast<cuFloatComplex *>(reinterpret_cast<const cuFloatComplex *>(idata_base));
  auto *fft_out = odata;

  // Execute Plan
  CUFFT_CHECK(cufftXtExec(plan_.get(), fft_in, fft_out, CUFFT_FORWARD));

  // Normalize
  const size_t total_out_elems = ctx.outputs[0].desc.num_elements();
  const float  scale           = norm_scale(settings_.norm, n_fft_);

  if (scale != 1.0f) {
    const int block = 256;
    const int grid  = static_cast<int>((total_out_elems + block - 1) / block);
    scale_cf32_kernel<<<grid, block, 0, stream_>>>(fft_out, total_out_elems, scale);
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
  check(idesc.dtype == holoflow::core::DType::CF32, "input dtype must be CF32");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim >= 2, "input ndim must be >= 2");
  check(ndim <= kMaxNDim, "input ndim too large");

  (void)normalize_axes(settings.axes, ndim);

  const auto total = product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  // holoflow::core::TDesc odesc = idesc;
  // // Output is always contiguous CF32
  // odesc.strides.clear();
  // odesc.offset = 0;
  
  holoflow::core::TDesc odesc(idesc.shape, holoflow::core::DType::CF32,
                             holoflow::core::MemLoc::Device);

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
                    const nlohmann::json                   &jsettings,
                    const holoflow::core::SyncCreateCtx    &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  const auto settings  = jsettings.get<FFT2Settings>();
  (void)infer_res;

  const auto &idesc = input_descs[0];
  const auto  total = product_shape(idesc.shape);
  const int   ndim  = static_cast<int>(idesc.shape.size());

  (void)normalize_axes(settings.axes, ndim);

  // FFT Dimensions (Last 2)
  const auto n0    = idesc.shape[static_cast<size_t>(ndim - 2)]; // Height
  const auto n1    = idesc.shape[static_cast<size_t>(ndim - 1)]; // Width
  const auto n_fft = n0 * n1;
  const auto batch = total / n_fft;

  // Stride Analysis
  const auto   strides_bytes = ensure_strides(idesc);
  const size_t elem_size     = holoflow::core::size_of(idesc.dtype);

  // 1. Check Element Alignment
  std::vector<long long int> s_el(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (strides_bytes[i] % elem_size != 0) {
      throw std::invalid_argument("FFT2: Input strides are not aligned to element size");
    }
    s_el[i] = static_cast<long long int>(strides_bytes[i] / elem_size);
  }

  // 2. Check 2D Axis Hierarchy compatibility
  // In a standard 2D layout suitable for cuFFT's 'inembed':
  // stride(Outer) must be K * stride(Inner).
  const long long int s_inner = s_el[ndim - 1]; // Stride of width (x)
  const long long int s_outer = s_el[ndim - 2]; // Stride of height (y)

  if (s_outer % s_inner != 0) {
    throw std::invalid_argument("FFT2: Incompatible strides for 2D FFT axes (outer stride must "
                                "be multiple of inner stride)");
  }

  // 3. Check Batch Linearity
  // cuFFT allows a single 'idist' for batch stride.
  // This means the batch dimensions must form a linear sequence in memory with a constant step.
  // Effectively, stride[Batch_k] must be stride[Batch_k+1] * shape[Batch_k+1].
  
  // Default for single batch (batch=1)
  int idist = static_cast<int>(n_fft);

  if (ndim > 2) {
    // The "base" stride for the batch is the stride of the innermost batch dimension (dim ndim-3).
    idist = static_cast<int>(s_el[ndim - 3]);

    // Check hierarchy upwards
    long long int expected_stride = idist;
    for (int i = ndim - 3; i >= 0; --i) {
      if (s_el[i] != expected_stride) {
        throw std::invalid_argument("FFT2: Batch dimensions are not contiguous/linear in memory");
      }
      // Prepare expected stride for the next higher dimension (i-1)
      if (i > 0) {
        expected_stride *= static_cast<long long int>(idesc.shape[i]);
      }
    }
  }

  // --- Layout Supported: Map to Plan Parameters ---

  int           rank        = 2;
  long long int n[2]        = {static_cast<long long int>(n0), static_cast<long long int>(n1)};
  
  // Input Layout
  // istride: distance between two successive input elements in the least significant dimension
  int istride = static_cast<int>(s_inner);
  
  // inembed: storage dimensions.
  // inembed[1] is the stride of the outer dimension divided by the stride of the inner dimension.
  // Address = b*idist + (y * inembed[1] + x) * istride
  long long int inembed[2];
  // inembed[0] = static_cast<long long int>(n0); // Not strictly used for stride logic in 2D
  inembed[0] = 320;
  inembed[1] = s_outer / s_inner;

  // Output Layout (Always Standard/Contiguous)
  long long int onembed[2] = {static_cast<long long int>(n0), static_cast<long long int>(n1)};
  int           ostride    = 1;
  int           odist      = static_cast<int>(n_fft);

  size_t work_size = 0;

  curaii::CufftHandle plan;
  CUFFT_CHECK(cufftSetStream(plan.get(), ctx.stream));
  CUFFT_CHECK(cufftXtMakePlanMany(plan.get(), rank, n, inembed, istride, idist, CUDA_C_32F,
                                  onembed, ostride, odist, CUDA_C_32F, static_cast<int>(batch),
                                  &work_size, CUDA_C_32F));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFT2(settings, std::move(plan), n_fft, ctx.stream));
}

} // namespace holonp