#include "holonp/arange.hh"

#include <cuComplex.h>

namespace holonp {

void to_json(nlohmann::json &j, const ArangeSettings &s) {
  j = nlohmann::json{{"start", s.start}, {"stop", s.stop}, {"step", s.step}};

  if (s.dtype) {
    j["dtype"] = *s.dtype;
  }

  if (s.device) {
    j["device"] = *s.device;
  }
}

void from_json(const nlohmann::json &j, ArangeSettings &s) {
  j.at("start").get_to(s.start);
  j.at("stop").get_to(s.stop);
  j.at("step").get_to(s.step);

  if (j.contains("dtype")) {
    s.dtype = j.at("dtype").get<holoflow::core::DType>();
  } else {
    s.dtype = std::nullopt;
  }

  if (j.contains("device")) {
    s.device = j.at("device").get<holoflow::core::MemLoc>();
  } else {
    s.device = std::nullopt;
  }
}

namespace {

template <class T>
__global__ void arange_kernel_scalar(T *out, std::int64_t n, double start, double step) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const double v = start + step * static_cast<double>(idx);
    out[idx]       = static_cast<T>(v);
  }
}

__global__ void arange_kernel_cf32(cuFloatComplex *out, std::int64_t n, double start, double step) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = static_cast<float>(start + step * static_cast<double>(idx));
    out[idx]      = make_cuFloatComplex(v, 0.0f);
  }
}

} // namespace

Arange::Arange(const ArangeSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Arange::execute(holoflow::core::SyncCtx &ctx) {
  auto *odata = ctx.outputs[0].data();
  auto  odesc = ctx.outputs[0].desc;
  const auto dtype    = settings_.dtype.value_or(holoflow::core::DType::F32);
  const auto n        = static_cast<std::int64_t>(odesc.num_elements());

  if (n <= 0) {
    logger()->error("[Arange::execute] output tensor has non-positive length");
    std::abort();
  }

  constexpr int block_size = 256;
  const int     grid_size  = static_cast<int>((n + block_size - 1) / block_size);

  switch (dtype) {
  case holoflow::core::DType::U8: {
    auto *out = reinterpret_cast<std::uint8_t *>(odata);
    arange_kernel_scalar<<<grid_size, block_size, 0, stream_>>>(out, n, settings_.start,
                                                                settings_.step);
    break;
  }
  case holoflow::core::DType::U16: {
    auto *out = reinterpret_cast<std::uint16_t *>(odata);
    arange_kernel_scalar<<<grid_size, block_size, 0, stream_>>>(out, n, settings_.start,
                                                                settings_.step);
    break;
  }
  case holoflow::core::DType::F32: {
    auto *out = reinterpret_cast<float *>(odata);
    arange_kernel_scalar<<<grid_size, block_size, 0, stream_>>>(out, n, settings_.start,
                                                                settings_.step);
    break;
  }
  case holoflow::core::DType::CF32: {
    auto *out = reinterpret_cast<cuFloatComplex *>(odata);
    arange_kernel_cf32<<<grid_size, block_size, 0, stream_>>>(out, n, settings_.start,
                                                              settings_.step);
    break;
  }
  default:
    logger()->error("[Arange::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

namespace {

size_t arange_len(double start, double stop, double step) {
  const bool forward = step > 0.0;
  if (forward && stop <= start) {
    return 0;
  } else if (!forward && stop >= start) {
    return 0;
  }

  const double span = stop - start;
  const double n_d  = std::ceil(span / step);

  if (!std::isfinite(n_d) || n_d <= 0.0) {
    throw std::invalid_argument("Arange: invalid resulting length");
  }

  const double max_u64 = static_cast<double>(std::numeric_limits<size_t>::max());
  if (n_d > max_u64) {
    throw std::invalid_argument("Arange: resulting length too large");
  }

  return static_cast<std::int64_t>(n_d);
}

} // namespace

holoflow::core::InferResult ArangeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      throw std::invalid_argument("ArangeFactory inference error: " + msg);
    }
  };

  const auto settings = jsettings.get<ArangeSettings>();
  const auto dtype    = settings.dtype.value_or(holoflow::core::DType::F32);
  const auto memloc   = settings.device.value_or(holoflow::core::MemLoc::Device);

  check(input_descs.empty(), "expected zero inputs");
  check(memloc == holoflow::core::MemLoc::Device, "only Device output is supported (for now)");
  check(settings.step != 0.0, "step must be non-zero");

  const size_t n = arange_len(settings.start, settings.stop, settings.step);

  holoflow::core::TDesc odesc({n}, dtype, memloc);

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
ArangeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<ArangeSettings>();
  (void)infer;

  auto *task = new Arange(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holonp