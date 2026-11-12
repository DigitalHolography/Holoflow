// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "display_spectrum.hh"

#include <QMetaObject>
#include <QPointer>

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "spectrum_power.hh"
#include "ui/widgets/spectrum_widget.hh"

namespace holovibes::tasks::sinks {
namespace {

class DisplaySpectrumTask : public holoflow::core::ISyncTask {
public:
  DisplaySpectrumTask(DisplaySpectrumSettings settings, ui::SpectrumWidget *widget,
                      cudaStream_t stream)
      : settings_(settings), widget_(widget), stream_(stream),
        scratch_(curaii::make_unique_device_ptr<float>(
            (settings.full_fft ? settings.window_size : settings.window_size / 2 + 1) *
            kSpectrumPowerPartials)),
        power_(curaii::make_unique_device_ptr<float>(
            settings.full_fft ? settings.window_size : settings.window_size / 2 + 1)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto now = std::chrono::steady_clock::now();
    if (widget_.isNull() || now < next_refresh_ || pending_->load()) {
      return holoflow::core::OpResult::Ok;
    }
    next_refresh_ = now + std::chrono::milliseconds(10);

    auto              &input = ctx.inputs[0];
    std::vector<float> power(input.desc.shape[1]);
    const auto         bytes = power.size() * sizeof(float);
    compute_spectrum_power(reinterpret_cast<const cuFloatComplex *>(input.data()), scratch_.get(),
                           power_.get(), input.desc.shape[0], input.desc.shape[1],
                           input.desc.shape[2] * input.desc.shape[3], stream_);
    CUDA_CHECK(cudaMemcpyAsync(power.data(), power_.get(), bytes, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto samples = ui::make_spectrum(power, settings_.window_size, settings_.sampling_frequency_hz,
                                     settings_.full_fft);
    pending_->store(true);
    const bool queued = QMetaObject::invokeMethod(
        widget_.data(),
        [widget = widget_, pending = pending_, settings = settings_,
         samples = std::move(samples)]() mutable {
          if (widget) {
            widget->present_spectrum(std::move(samples), settings.sampling_frequency_hz,
                                     settings.full_fft);
          }
          pending->store(false);
        },
        Qt::QueuedConnection);
    if (!queued) {
      pending_->store(false);
    }
    return holoflow::core::OpResult::Ok;
  }

private:
  DisplaySpectrumSettings               settings_;
  QPointer<ui::SpectrumWidget>          widget_;
  cudaStream_t                          stream_;
  curaii::unique_device_ptr<float>      scratch_;
  curaii::unique_device_ptr<float>      power_;
  std::chrono::steady_clock::time_point next_refresh_{};
  std::shared_ptr<std::atomic_bool>     pending_ = std::make_shared<std::atomic_bool>(false);
};

} // namespace

void to_json(nlohmann::json &j, const DisplaySpectrumSettings &settings) {
  j = {{"window_size", settings.window_size},
       {"sampling_frequency_hz", settings.sampling_frequency_hz},
       {"full_fft", settings.full_fft}};
}

void from_json(const nlohmann::json &j, DisplaySpectrumSettings &settings) {
  j.at("window_size").get_to(settings.window_size);
  j.at("sampling_frequency_hz").get_to(settings.sampling_frequency_hz);
  j.at("full_fft").get_to(settings.full_fft);
}

DisplaySpectrumFactory::DisplaySpectrumFactory(ui::SpectrumWidget *widget) : widget_(widget) {
  if (widget == nullptr) {
    throw std::invalid_argument("DisplaySpectrumFactory requires a widget");
  }
}

holoflow::core::InferResult
DisplaySpectrumFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<DisplaySpectrumSettings>();
  if (settings.window_size == 0 || !std::isfinite(settings.sampling_frequency_hz) ||
      settings.sampling_frequency_hz <= 0.0) {
    throw std::invalid_argument("DisplaySpectrum: invalid window size or sampling frequency");
  }
  const auto bins = settings.full_fft ? settings.window_size : settings.window_size / 2 + 1;
  if (bins > std::numeric_limits<unsigned int>::max() || input_descs.size() != 1 ||
      input_descs[0].rank() != 4 || input_descs[0].shape[1] != bins ||
      input_descs[0].dtype != holoflow::core::DType::CF32 ||
      input_descs[0].mem_loc != holoflow::core::MemLoc::Device) {
    throw std::invalid_argument(
        "DisplaySpectrum: expected device CF32 [batch, frequency, height, width]");
  }
  const auto &desc            = input_descs[0];
  size_t      expected_stride = sizeof(cuFloatComplex);
  for (size_t axis = desc.rank(); axis-- > 0;) {
    if (desc.shape[axis] == 0 || desc.strides[axis] != expected_stride) {
      throw std::invalid_argument("DisplaySpectrum: expected nonempty contiguous input");
    }
    expected_stride *= desc.shape[axis];
  }
  return {.input_descs   = {input_descs[0]},
          .output_descs  = {},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
DisplaySpectrumFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                               const nlohmann::json                  &jsettings,
                               const holoflow::core::SyncCreateCtx   &ctx) const {
  infer(input_descs, jsettings);
  return std::make_unique<DisplaySpectrumTask>(jsettings.get<DisplaySpectrumSettings>(), widget_,
                                               ctx.stream);
}

} // namespace holovibes::tasks::sinks
