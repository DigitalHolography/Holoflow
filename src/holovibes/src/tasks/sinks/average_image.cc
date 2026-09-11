// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0.

#include "average_image.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <QImage>

#include "holoflow_event/router.hh"

namespace holovibes::tasks::sinks {
namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("AverageImageFactory error: " + message);
}

void validate(const AverageImageSettings &settings, const holoflow::core::TDesc &input) {
  check(settings.count > 0, "count must be positive");
  check(settings.format == "png" || settings.format == "jpg", "format must be png or jpg");
  check(!settings.path.empty(), "path must not be empty");
  check(input.mem_loc == holoflow::core::MemLoc::Host, "input tensor must be in Host memory");
  check(input.shape.size() == 3, "input tensor must have rank 3 (batch, height, width)");
  check(input.dtype == holoflow::core::DType::U8 || input.dtype == holoflow::core::DType::U16 ||
            input.dtype == holoflow::core::DType::F32,
        "unsupported input dtype");
  check(input.shape[0] > 0 && input.shape[1] > 0 && input.shape[2] > 0,
        "input dimensions must be positive");
  check(input.shape[1] <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            input.shape[2] <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "input dimensions are too large for Qt");
  check(input.strides.size() == 3, "input tensor must provide rank-3 strides");
  check(input.strides[1] >= input.shape[2] * holoflow::core::size_of(input.dtype),
        "input row stride is too small");
}

class AverageImageTask final : public holoflow::core::ISyncTask {
public:
  AverageImageTask(AverageImageSettings settings, holoflow::core::TDesc input)
      : settings_(std::move(settings)), input_(std::move(input)),
        sum_(input_.shape[1] * input_.shape[2], 0.0) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_events(ctx);
    if (!recording_)
      return holoflow::core::OpResult::Ok;

    auto       &input      = ctx.inputs[0];
    const auto  batch      = static_cast<int>(input.desc.shape[0]);
    const auto  frames     = std::min(settings_.count - frames_seen_, batch);
    const auto  height     = input.desc.shape[1];
    const auto  width      = input.desc.shape[2];
    const auto  pixel_size = holoflow::core::size_of(input.desc.dtype);
    const auto *base       = static_cast<const std::byte *>(input.data());

    for (int frame = 0; frame < frames; ++frame) {
      const auto *image = base + static_cast<std::size_t>(frame) * input.desc.strides[0];
      for (std::size_t y = 0; y < height; ++y) {
        const auto *row = image + y * input.desc.strides[1];
        for (std::size_t x = 0; x < width; ++x) {
          const auto *pixel = row + x * pixel_size;
          sum_[y * width + x] +=
              input.desc.dtype == holoflow::core::DType::U8
                  ? static_cast<double>(*reinterpret_cast<const std::uint8_t *>(pixel))
              : input.desc.dtype == holoflow::core::DType::U16
                  ? static_cast<double>(*reinterpret_cast<const std::uint16_t *>(pixel))
                  : static_cast<double>(*reinterpret_cast<const float *>(pixel));
        }
      }
      ++frames_seen_;
    }

    if (frames_seen_ == settings_.count) {
      try {
        write_image();
        emit_event(ctx, "recording_finished", {});
      } catch (const std::exception &error) {
        (void)std::remove(settings_.path.c_str());
        emit_event(ctx, "recording_failed", {{"message", error.what()}});
      }
      reset();
    }
    return holoflow::core::OpResult::Ok;
  }

private:
  void handle_events(holoflow::core::SyncCtx &ctx) {
    if (!ctx.event_reader)
      return;
    while (auto event = ctx.event_reader->try_pop()) {
      const auto type = event->data.at("type").get<std::string>();
      if (type == "start_recording") {
        settings_.path = event->data.value("record_path", settings_.path);
        check(!settings_.path.empty(), "recording path must not be empty");
        reset();
        recording_ = true;
      } else if (type == "stop_recording") {
        recording_ = false;
        reset();
      } else {
        throw std::invalid_argument("Unknown recording event: " + type);
      }
    }
  }

  void write_image() const {
    const auto width  = static_cast<int>(input_.shape[2]);
    const auto height = static_cast<int>(input_.shape[1]);
    const bool jpg    = settings_.format == "jpg";
    QImage     image(
        width, height,
        jpg ? QImage::Format_Grayscale8
            : (settings_.output_16bit ? QImage::Format_Grayscale16 : QImage::Format_Grayscale8));
    const double divisor = static_cast<double>(settings_.count);
    double       min_value = 0.0;
    double       max_value = 255.0;
    if (input_.dtype == holoflow::core::DType::F32 && !settings_.output_16bit) {
      min_value = std::numeric_limits<double>::max();
      max_value = std::numeric_limits<double>::lowest();
      for (const auto value : sum_) {
        const auto average = value / divisor;
        min_value           = std::min(min_value, average);
        max_value           = std::max(max_value, average);
      }
    }
    const double scale = max_value > min_value ? 255.0 / (max_value - min_value) : 0.0;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        auto average = sum_[static_cast<std::size_t>(y) * width + x] / divisor;
        if (input_.dtype == holoflow::core::DType::F32 && !settings_.output_16bit)
          average = (average - min_value) * scale;
        if (image.format() == QImage::Format_Grayscale8) {
          const auto value     = jpg && settings_.output_16bit ? average / 257.0 : average;
          image.scanLine(y)[x] = static_cast<std::uint8_t>(std::lround(value));
        } else {
          reinterpret_cast<std::uint16_t *>(image.scanLine(y))[x] =
              static_cast<std::uint16_t>(std::lround(average));
        }
      }
    }
    const auto square_size = std::max(width, height);
    image = image.scaled(square_size, square_size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    if (!image.save(QString::fromStdString(settings_.path),
                    settings_.format == "png" ? "PNG" : "JPG"))
      throw std::runtime_error("failed to save image: " + settings_.path);
  }

  void emit_event(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data) {
    if (!ctx.event_writer)
      return;
    data["type"] = type;
    data["path"] = settings_.path;
    (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                      .node_id   = "",
                                      .data      = std::move(data),
                                      .ts        = std::chrono::steady_clock::now()});
  }

  void reset() {
    std::fill(sum_.begin(), sum_.end(), 0.0);
    frames_seen_ = 0;
  }

  AverageImageSettings  settings_;
  holoflow::core::TDesc input_;
  std::vector<double>   sum_;
  int                   frames_seen_ = 0;
  bool                  recording_   = false;
};

} // namespace

void to_json(nlohmann::json &j, const AverageImageSettings &settings) {
  j = {{"path", settings.path},
       {"count", settings.count},
       {"format", settings.format},
       {"output_16bit", settings.output_16bit}};
}

void from_json(const nlohmann::json &j, AverageImageSettings &settings) {
  j.at("path").get_to(settings.path);
  j.at("count").get_to(settings.count);
  j.at("format").get_to(settings.format);
  settings.output_16bit = j.value("output_16bit", settings.output_16bit);
}

holoflow::core::InferResult
AverageImageFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                           const nlohmann::json                  &jsettings) const {
  check(inputs.size() == 1, "expected exactly one input tensor");
  const auto settings = jsettings.get<AverageImageSettings>();
  validate(settings, inputs[0]);
  check(settings.count % inputs[0].shape[0] == 0, "count must be divisible by batch size");
  check(!settings.output_16bit || inputs[0].dtype == holoflow::core::DType::F32 ||
            inputs[0].dtype == holoflow::core::DType::U16,
        "16-bit output requires U16 or F32 input");
  return {.input_descs   = {inputs[0]},
          .output_descs  = {},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
AverageImageFactory::create(std::span<const holoflow::core::TDesc> inputs,
                            const nlohmann::json                  &jsettings,
                            const holoflow::core::SyncCreateCtx &) const {
  infer(inputs, jsettings);
  return std::make_unique<AverageImageTask>(jsettings.get<AverageImageSettings>(), inputs[0]);
}

std::unique_ptr<holoflow::core::ISyncTask>
AverageImageFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                            std::span<const holoflow::core::TDesc>     inputs,
                            const nlohmann::json                      &jsettings,
                            const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)old_task;
  return create(inputs, jsettings, ctx);
}

} // namespace holovibes::tasks::sinks
