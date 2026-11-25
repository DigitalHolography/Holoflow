#include "holotask/syncs/resize.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "resize_cuda.cuh"
#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {
namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("ResizeFactory inference error: " + message);
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.dtype == b.dtype && a.mem_loc == b.mem_loc &&
         a.strides == b.strides && a.offset == b.offset;
}

template <typename T>
void resize_bilinear(const std::uint8_t *source, std::uint8_t *destination, std::size_t batch,
                     std::size_t source_height, std::size_t source_width,
                     std::size_t source_frame_stride, std::size_t source_row_stride,
                     std::size_t destination_height, std::size_t destination_width) {
  auto *out = reinterpret_cast<T *>(destination);
  const auto *in = reinterpret_cast<const std::uint8_t *>(source);
  for (std::size_t b = 0; b < batch; ++b) {
    const auto *frame = in + b * source_frame_stride;
    auto *out_frame = out + b * destination_height * destination_width;
    for (std::size_t y = 0; y < destination_height; ++y) {
      const double fy = destination_height == 1
                            ? 0.0
                            : static_cast<double>(y) * (source_height - 1) /
                                  static_cast<double>(destination_height - 1);
      const auto y0 = static_cast<std::size_t>(fy);
      const auto y1 = std::min(y0 + 1, source_height - 1);
      const double wy = fy - static_cast<double>(y0);
      const auto *row0 = reinterpret_cast<const T *>(frame + y0 * source_row_stride);
      const auto *row1 = reinterpret_cast<const T *>(frame + y1 * source_row_stride);
      for (std::size_t x = 0; x < destination_width; ++x) {
        const double fx = destination_width == 1
                              ? 0.0
                              : static_cast<double>(x) * (source_width - 1) /
                                    static_cast<double>(destination_width - 1);
        const auto x0 = static_cast<std::size_t>(fx);
        const auto x1 = std::min(x0 + 1, source_width - 1);
        const double wx = fx - static_cast<double>(x0);
        const double top = static_cast<double>(row0[x0]) * (1.0 - wx) + row0[x1] * wx;
        const double bottom = static_cast<double>(row1[x0]) * (1.0 - wx) + row1[x1] * wx;
        out_frame[y * destination_width + x] = static_cast<T>(std::lround(top * (1.0 - wy) + bottom * wy));
      }
    }
  }
}

class Resize final : public holoflow::core::ISyncTask {
public:
  Resize(ResizeSettings settings, holoflow::core::TDesc input_desc, cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto &input = ctx.inputs[0].desc;
    const auto &output = ctx.outputs[0].desc;
    const auto bytes = holoflow::core::size_of(input.dtype);
    if (settings_.algorithm == ResizeAlgorithm::CudaBilinear) {
      detail::launch_resize_bilinear_cuda(
          ctx.inputs[0].data(), ctx.outputs[0].data(), input.shape[0], input.shape[1], input.shape[2],
          input.strides[0], input.strides[1], output.shape[1], output.shape[2], bytes, stream_);
    } else if (bytes == 1)
      resize_bilinear<std::uint8_t>(reinterpret_cast<const std::uint8_t *>(ctx.inputs[0].data()),
                                    reinterpret_cast<std::uint8_t *>(ctx.outputs[0].data()), input.shape[0],
                                    input.shape[1], input.shape[2], input.strides[0], input.strides[1],
                                    output.shape[1], output.shape[2]);
    else
      resize_bilinear<std::uint16_t>(reinterpret_cast<const std::uint8_t *>(ctx.inputs[0].data()),
                                     reinterpret_cast<std::uint8_t *>(ctx.outputs[0].data()), input.shape[0],
                                     input.shape[1], input.shape[2], input.strides[0], input.strides[1],
                                     output.shape[1], output.shape[2]);
    return holoflow::core::OpResult::Ok;
  }

  const ResizeSettings &settings() const { return settings_; }
  const holoflow::core::TDesc &input_desc() const { return input_desc_; }
  void update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ResizeSettings settings_;
  holoflow::core::TDesc input_desc_;
  cudaStream_t          stream_;
};

} // namespace

void to_json(nlohmann::json &j, const ResizeInterpolation &) { j = "Bilinear"; }
void from_json(const nlohmann::json &j, ResizeInterpolation &interpolation) {
  if (j.get<std::string>() != "Bilinear")
    throw std::invalid_argument("Invalid resize interpolation");
  interpolation = ResizeInterpolation::Bilinear;
}
void to_json(nlohmann::json &j, const ResizeAlgorithm &algorithm) {
  j = algorithm == ResizeAlgorithm::CpuBilinear ? "CpuBilinear" : "CudaBilinear";
}
void from_json(const nlohmann::json &j, ResizeAlgorithm &algorithm) {
  const auto value = j.get<std::string>();
  if (value == "CpuBilinear") {
    algorithm = ResizeAlgorithm::CpuBilinear;
  } else if (value == "CudaBilinear") {
    algorithm = ResizeAlgorithm::CudaBilinear;
  } else {
    throw std::invalid_argument("Invalid resize algorithm: " + value);
  }
}
void to_json(nlohmann::json &j, const ResizeSettings &settings) {
  j = {{"width", settings.width},
       {"height", settings.height},
       {"interpolation", settings.interpolation},
       {"algorithm", settings.algorithm}};
}
void from_json(const nlohmann::json &j, ResizeSettings &settings) {
  j.at("width").get_to(settings.width);
  j.at("height").get_to(settings.height);
  if (j.contains("interpolation"))
    j.at("interpolation").get_to(settings.interpolation);
  if (j.contains("algorithm"))
    j.at("algorithm").get_to(settings.algorithm);
}

holoflow::core::InferResult ResizeFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                  const nlohmann::json &json) const {
  const auto settings = json.get<ResizeSettings>();
  check(inputs.size() == 1, "expected exactly one input");
  const auto &input = inputs[0];
  const auto expected_mem_loc = settings.algorithm == ResizeAlgorithm::CpuBilinear
                                   ? holoflow::core::MemLoc::Host
                                   : holoflow::core::MemLoc::Device;
  check(input.mem_loc == expected_mem_loc,
        settings.algorithm == ResizeAlgorithm::CpuBilinear ? "CPU resize input must be in Host memory"
                                                            : "CUDA resize input must be in Device memory");
  check(input.shape.size() == 3, "input must have rank 3 (batch, height, width)");
  check(input.dtype == holoflow::core::DType::U8 || input.dtype == holoflow::core::DType::U16,
        "only U8 and U16 inputs are supported");
  check(settings.width > 0 && settings.height > 0, "output dimensions must be positive");
  check(settings.interpolation == ResizeInterpolation::Bilinear, "unsupported interpolation");
  return {.input_descs = {input},
          .output_descs = {holoflow::core::TDesc({input.shape[0], static_cast<std::size_t>(settings.height),
                                                  static_cast<std::size_t>(settings.width)},
                                                  input.dtype, input.mem_loc)},
          .in_place = {}, .owned_inputs = {false}, .owned_outputs = {false},
          .kind = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
ResizeFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                       const holoflow::core::SyncCreateCtx &ctx) const {
  infer(inputs, json);
  return std::make_unique<Resize>(json.get<ResizeSettings>(), inputs[0], ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ResizeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                       const holoflow::core::SyncCreateCtx &ctx) const {
  infer(inputs, json);
  auto *old = dynamic_cast<Resize *>(old_task.get());
  const auto settings = json.get<ResizeSettings>();
  if (old != nullptr && old->settings() == settings && same_desc(old->input_desc(), inputs[0])) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  return create(inputs, json, ctx);
}

} // namespace holotask::syncs
