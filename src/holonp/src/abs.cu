#include "holonp/abs.hh"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const AbsSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, AbsSettings &) {}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Abs: " + msg);
  }
}

template <typename T> __device__ T abs_val(T v) { return v < 0 ? -v : v; }
template <> __device__ std::uint8_t abs_val(std::uint8_t v) { return v; }
template <> __device__ std::uint16_t abs_val(std::uint16_t v) { return v; }
template <> __device__ float         abs_val(float v) { return fabsf(v); }

template <typename InT, typename OutT>
__global__ void abs_kernel(const InT *__restrict__ in, OutT *__restrict__ out, std::int64_t n) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    out[idx] = static_cast<OutT>(abs_val(in[idx]));
  }
}

__global__ void abs_kernel_cf32(const cuFloatComplex *__restrict__ in, float *__restrict__ out,
                                std::int64_t n) {
  const std::int64_t idx =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (idx < n) {
    const auto v = in[idx];
    out[idx]     = sqrtf(v.x * v.x + v.y * v.y);
  }
}

} // namespace

Abs::Abs(const AbsSettings &settings, cudaStream_t stream) : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Abs::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  (void)odesc;

  const auto n = static_cast<std::int64_t>(idesc.num_elements());
  if (n == 0) {
    return holoflow::core::OpResult::Ok;
  }

  constexpr int block = 256;
  const int     grid  = static_cast<int>((n + block - 1) / block);

  switch (idesc.dtype) {
  case holoflow::core::DType::U8: {
    auto *in  = reinterpret_cast<const std::uint8_t *>(idata);
    auto *out = reinterpret_cast<std::uint8_t *>(odata);
    abs_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::U16: {
    auto *in  = reinterpret_cast<const std::uint16_t *>(idata);
    auto *out = reinterpret_cast<std::uint16_t *>(odata);
    abs_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::F32: {
    auto *in  = reinterpret_cast<const float *>(idata);
    auto *out = reinterpret_cast<float *>(odata);
    abs_kernel<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  case holoflow::core::DType::CF32: {
    auto *in  = reinterpret_cast<const cuFloatComplex *>(idata);
    auto *out = reinterpret_cast<float *>(odata);
    abs_kernel_cf32<<<grid, block, 0, stream_>>>(in, out, n);
    break;
  }
  default:
    logger()->error("[Abs::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult AbsFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  (void)jsettings;

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::U8 || idesc.dtype == holoflow::core::DType::U16 ||
            idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");
  check(idesc.num_elements() > 0, "input tensor has zero elements");

  // holoflow::core::TDesc odesc = idesc;
  // if (idesc.dtype == holoflow::core::DType::CF32) {
  //   odesc.dtype = holoflow::core::DType::F32;
  // }
  // 

  holoflow::core::TDesc odesc(
      idesc.shape,
      idesc.dtype == holoflow::core::DType::CF32 ? holoflow::core::DType::F32 : idesc.dtype,
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
AbsFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto infer_res = this->infer(input_descs, jsettings);
  (void)infer_res;

  const auto settings = jsettings.get<AbsSettings>();
  return std::unique_ptr<holoflow::core::ISyncTask>(new Abs(settings, ctx.stream));
}

} // namespace holonp
