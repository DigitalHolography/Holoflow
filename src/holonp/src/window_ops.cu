// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#include "holonp/window_ops.hh"

#include <array>
#include <limits>
#include <stdexcept>

#include "curaii/cuda.hh"

namespace holonp {
namespace {
using namespace holoflow::core;
constexpr size_t max_rank = 16;
using Width               = std::array<int64_t, 2>;

void check(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(message);
}

struct PadLayout {
  size_t rank;
  size_t shape[max_rank];
  size_t strides[max_rank];
  size_t padded[max_rank];
  size_t before[max_rank];
};

__global__ void pad_edge(const float *input, float *output, size_t count, PadLayout layout) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  size_t remaining = index;
  size_t source    = 0;
  for (size_t i = layout.rank; i-- > 0;) {
    const size_t coord = remaining % layout.padded[i];
    remaining /= layout.padded[i];
    const auto shifted = coord < layout.before[i] ? 0 : coord - layout.before[i];
    source += (shifted < layout.shape[i] ? shifted : layout.shape[i] - 1) * layout.strides[i];
  }
  output[index] = input[source];
}

class Pad final : public ISyncTask {
public:
  Pad(PadLayout layout, cudaStream_t stream) : layout_(layout), stream_(stream) {}
  bool supports_cuda_graph() const noexcept override { return true; }
  void record_cuda_graph(CudaGraphCtx &ctx) override {
    SyncCtx call{ctx.inputs, ctx.outputs, nullptr, nullptr, nullptr};
    (void)execute(call);
  }
  OpResult execute(SyncCtx &ctx) override {
    const size_t count = ctx.outputs[0].desc.num_elements();
    pad_edge<<<static_cast<unsigned>((count - 1) / 256 + 1), 256, 0, stream_>>>(
        reinterpret_cast<const float *>(ctx.inputs[0].data()),
        reinterpret_cast<float *>(ctx.outputs[0].data()), count, layout_);
    CUDA_CHECK(cudaGetLastError());
    return OpResult::Ok;
  }

private:
  PadLayout    layout_;
  cudaStream_t stream_;
};

class WindowView final : public ISyncTask {
public:
  bool     supports_cuda_graph() const noexcept override { return true; }
  void     record_cuda_graph(CudaGraphCtx &) override {}
  OpResult execute(SyncCtx &) override { return OpResult::Ok; }
};
} // namespace

holoflow::core::InferResult PadFactory::infer(std::span<const TDesc> inputs,
                                              const nlohmann::json  &settings) const {
  check(inputs.size() == 1, "Pad requires one input");
  const auto &x      = inputs[0];
  const auto  widths = settings.at("pad_width").get<std::vector<Width>>();
  check(settings.at("mode") == "edge" && widths.size() == x.rank() && x.rank() <= max_rank &&
            x.rank() > 0 && x.num_elements() && x.dtype == DType::F32 &&
            x.mem_loc == MemLoc::Device,
        "Pad requires nonempty device F32 input and edge widths for every axis");
  auto shape = x.shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    for (const auto width : widths[i]) {
      check(width >= 0 &&
                static_cast<uint64_t>(width) <= std::numeric_limits<size_t>::max() - shape[i],
            "invalid padding width");
      shape[i] += static_cast<size_t>(width);
    }
    check(x.strides[i] % sizeof(float) == 0, "Pad requires element-aligned strides");
  }
  TDesc out(shape, x.dtype, x.mem_loc);
  check((out.num_elements() - 1) / 256 < std::numeric_limits<int>::max(),
        "padded output exceeds CUDA grid limits");
  return {{x}, {out}, {}, {false}, {false}, TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask> PadFactory::create(std::span<const TDesc> inputs,
                                                              const nlohmann::json  &settings,
                                                              const SyncCreateCtx   &ctx) const {
  const auto  out    = infer(inputs, settings).output_descs[0];
  const auto  widths = settings.at("pad_width").get<std::vector<Width>>();
  const auto &x      = inputs[0];
  PadLayout   layout{};
  layout.rank = x.rank();
  for (size_t i = 0; i < x.rank(); ++i) {
    layout.shape[i]   = x.shape[i];
    layout.strides[i] = x.strides[i] / sizeof(float);
    layout.padded[i]  = out.shape[i];
    layout.before[i]  = static_cast<size_t>(widths[i][0]);
  }
  return std::make_unique<Pad>(layout, ctx.stream);
}

holoflow::core::InferResult SlidingWindowViewFactory::infer(std::span<const TDesc> inputs,
                                                            const nlohmann::json  &settings) const {
  check(inputs.size() == 1, "SlidingWindowView requires one input");
  const auto &x    = inputs[0];
  auto        axis = settings.at("axis").get<int64_t>();
  const auto  size = settings.at("window_shape").get<int64_t>();
  if (axis < 0)
    axis += static_cast<int64_t>(x.rank());
  check(axis >= 0 && static_cast<size_t>(axis) < x.rank(), "window axis out of range");
  check(size > 0 && static_cast<size_t>(size) <= x.shape[axis], "invalid window size");
  auto out = x;
  out.shape[axis] -= static_cast<size_t>(size) - 1;
  out.shape.push_back(static_cast<size_t>(size));
  out.strides.push_back(x.strides[axis]);
  return {{x}, {out}, {{0, 0, true}}, {false}, {false}, TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
SlidingWindowViewFactory::create(std::span<const TDesc> inputs, const nlohmann::json &settings,
                                 const SyncCreateCtx &) const {
  (void)infer(inputs, settings);
  return std::make_unique<WindowView>();
}
} // namespace holonp
