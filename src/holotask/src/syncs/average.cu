#include "holotask/syncs/average.hh"

#include <cuComplex.h>
#include <set>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const AverageSettings &s) {
  j = nlohmann::json{
      {"axis", s.axis},
      {"start", s.start},
      {"end", s.end},
  };
}

void from_json(const nlohmann::json &j, AverageSettings &s) {
  j.at("axis").get_to(s.axis);
  j.at("start").get_to(s.start);
  j.at("end").get_to(s.end);
}

namespace {

template <typename T> struct AvgTraits {
  using Acc = float;
  __device__ static void add(Acc &a, T v) { a += static_cast<float>(v); }
  __device__ static T    finish(Acc a, float n) { return static_cast<T>(a / n); }
};

template <> struct AvgTraits<cuFloatComplex> {
  using Acc = cuFloatComplex;
  __device__ static void           add(Acc &a, cuFloatComplex v) { a = cuCaddf(a, v); }
  __device__ static cuFloatComplex finish(Acc a, float n) {
    return make_cuFloatComplex(a.x / n, a.y / n);
  }
};

template <int Axis, class T>
__global__ void average_kernel(const T *__restrict__ in, T *__restrict__ out, int B, int H, int W,
                               int begin, int end) {
  // Output geometry per axis:
  // Axis 0 -> out(H,W):  tx=x in [0,W), ty=y in [0,H)
  // Axis 1 -> out(B,W):  tx=x in [0,W), ty=z in [0,B)
  // Axis 2 -> out(B,H):  tx=y in [0,H), ty=z in [0,B)
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int pitchZ = W * H;

  if constexpr (Axis == 0) {
    if (tx >= W || ty >= H) {
      return;
    }
    const int x = tx, y = ty;
    using Acc = typename AvgTraits<T>::Acc;
    Acc       sum{};
    const int baseHW = y * W + x;
    for (int z = begin; z < end; ++z) {
      sum = (AvgTraits<T>::add(sum, in[z * pitchZ + baseHW]), sum);
    }
    out[y * W + x] = AvgTraits<T>::finish(sum, float(end - begin));
  }

  else if constexpr (Axis == 1) {
    if (tx >= W || ty >= B) {
      return;
    }
    const int x = tx, z = ty;
    using Acc = typename AvgTraits<T>::Acc;
    Acc       sum{};
    const int baseZ = z * pitchZ;
    for (int y = begin; y < end; ++y) {
      sum = (AvgTraits<T>::add(sum, in[baseZ + y * W + x]), sum);
    }
    out[z * W + x] = AvgTraits<T>::finish(sum, float(end - begin));
  }

  else if constexpr (Axis == 2) {
    if (tx >= H || ty >= B) {
      return;
    }
    const int y = tx, z = ty;
    using Acc = typename AvgTraits<T>::Acc;
    Acc       sum{};
    const int baseZY = z * pitchZ + y * W;
    for (int x = begin; x < end; ++x) {
      sum = (AvgTraits<T>::add(sum, in[baseZY + x]), sum);
    }
    out[z * H + y] = AvgTraits<T>::finish(sum, float(end - begin));
  }

  else {
    static_assert(Axis >= 0 && Axis <= 2, "Invalid axis");
  }
}

template <int Axis, class T>
void launch_avg(cudaStream_t s, const void *in, void *out, int B, int H, int W, int begin,
                int end) {
  dim3 block(16, 16);
  dim3 grid;
  if constexpr (Axis == 0) {
    grid = dim3((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  } else if constexpr (Axis == 1) {
    grid = dim3((W + block.x - 1) / block.x, (B + block.y - 1) / block.y);
  } else if constexpr (Axis == 2) {
    grid = dim3((H + block.x - 1) / block.x, (B + block.y - 1) / block.y);
  } else {
    static_assert(Axis >= 0 && Axis <= 2, "Invalid axis");
  }
  average_kernel<Axis, T><<<grid, block, 0, s>>>(static_cast<const T *>(in), static_cast<T *>(out),
                                                 B, H, W, begin, end);
}

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

using Launcher = void (*)(cudaStream_t s, const void *in, void *out, int B, int H, int W, int begin,
                          int end);

static constexpr Launcher DISPATCH[3][4] = {
    {launch_avg<0, uint8_t>, launch_avg<0, uint16_t>, launch_avg<0, float>,
     launch_avg<0, cuFloatComplex>},
    {launch_avg<1, uint8_t>, launch_avg<1, uint16_t>, launch_avg<1, float>,
     launch_avg<1, cuFloatComplex>},
    {launch_avg<2, uint8_t>, launch_avg<2, uint16_t>, launch_avg<2, float>,
     launch_avg<2, cuFloatComplex>},
};

} // namespace

Average::Average(const AverageSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Average::execute(holoflow::core::SyncCtx &ctx) {
  auto [idata, idesc] = ctx.inputs[0];
  auto [odata, odesc] = ctx.outputs[0];
  int  B              = static_cast<int>(idesc.shape.at(0));
  int  H              = static_cast<int>(idesc.shape.at(1));
  int  W              = static_cast<int>(idesc.shape.at(2));
  int  ax             = settings_.axis;
  int  dtype_idx      = dtype_index(idesc.dtype);
  auto f              = DISPATCH[ax][dtype_idx];
  f(stream_, idata, odata, B, H, W, settings_.start, settings_.end);
  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
AverageFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[AverageFactory::infer] error: {}", msg);
      throw std::invalid_argument("AverageFactory inference error: " + msg);
    }
  };
  auto settings = jsettings.get<AverageSettings>();

  const auto &idesc = input_descs[0];

  const std::set<holoflow::core::DType> supported_dtypes = {
      holoflow::core::DType::U8,
      holoflow::core::DType::U16,
      holoflow::core::DType::F32,
      holoflow::core::DType::CF32,
  };

  // Validate
  check(input_descs.size() == 1, "expected exactly one input");
  check(idesc.shape.size() == 3, "expected 3D input");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be on Device");
  check(supported_dtypes.contains(idesc.dtype), "unsupported input dtype");
  check(settings.axis >= 0 && settings.axis <= 2, "axis must be in [0, 2]");
  check(settings.start >= 0, "start must be non-negative");
  check(settings.end <= static_cast<int>(idesc.shape.at(settings.axis)), "end out of bounds");
  check(settings.start < settings.end, "start must be less than end");

  // Success
  auto odesc                    = idesc;
  odesc.shape.at(settings.axis) = 1;
  logger()->debug("[AverageFactory::infer] input shape z,y,x: {}, {}, {} / {}, {}, {}, axis = {}",
                  idesc.shape[0], idesc.shape[1], idesc.shape[2], odesc.shape[0], odesc.shape[1],
                  odesc.shape[2], settings.axis);
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
AverageFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto infer = this->infer(input_descs, jsettings);
  (void)infer; // unused
  auto settings = jsettings.get<AverageSettings>();

  // Success
  auto *task = new Average(settings, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs