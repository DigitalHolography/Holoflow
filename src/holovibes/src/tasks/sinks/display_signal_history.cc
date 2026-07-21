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

#include "display_signal_history.hh"

#include <QMetaObject>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bug.hh"
#include "cuda_runtime_api.h"
#include "logger.hh"
#include "ui/widgets/signal_history.hh"
#include "ui/widgets/zernike_history_widget.hh"

namespace holovibes::tasks::sinks {

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[DisplaySignalHistoryFactory] {}", msg);
    throw std::invalid_argument("DisplaySignalHistoryFactory: " + msg);
  }
}

class SignalHistoryDispatcher : public std::enable_shared_from_this<SignalHistoryDispatcher> {
public:
  explicit SignalHistoryDispatcher(QPointer<holovibes::ui::ZernikeHistoryWidget> widget)
      : widget_(std::move(widget)) {}

  void enqueue(std::vector<holovibes::ui::ZernikeHistorySample> samples) {
    bool should_schedule = false;
    {
      std::lock_guard lock(mutex_);
      pending_.insert(pending_.end(), std::make_move_iterator(samples.begin()),
                      std::make_move_iterator(samples.end()));
      if (!scheduled_) {
        scheduled_      = true;
        should_schedule = true;
      }
    }

    if (!should_schedule || widget_.isNull()) {
      return;
    }

    auto self = shared_from_this();
    if (!QMetaObject::invokeMethod(
            widget_.data(), [self]() { self->drain(); }, Qt::QueuedConnection)) {
      std::lock_guard lock(mutex_);
      scheduled_ = false;
    }
  }

  void configure(std::vector<int> indexes, double time_window_seconds) {
    if (widget_.isNull()) {
      return;
    }

    QPointer<holovibes::ui::ZernikeHistoryWidget> safe_widget = widget_;
    QMetaObject::invokeMethod(
        widget_.data(),
        [safe_widget, indexes = std::move(indexes), time_window_seconds]() {
          if (!safe_widget.isNull()) {
            safe_widget->set_series(indexes);
            safe_widget->set_time_window_seconds(time_window_seconds);
          }
        },
        Qt::QueuedConnection);
  }

private:
  void drain() {
    std::vector<holovibes::ui::ZernikeHistorySample> samples;
    {
      std::lock_guard lock(mutex_);
      samples.swap(pending_);
      scheduled_ = false;
    }

    if (!widget_.isNull() && !samples.empty()) {
      widget_->append_samples(std::move(samples));
    }
  }

  QPointer<holovibes::ui::ZernikeHistoryWidget>    widget_;
  std::mutex                                       mutex_;
  std::vector<holovibes::ui::ZernikeHistorySample> pending_;
  bool                                             scheduled_ = false;
};

// -------------------------------------------------------------------------------------------------
// DisplaySignalHistoryTask
// -------------------------------------------------------------------------------------------------

class DisplaySignalHistoryTask : public holoflow::core::ISyncTask {
public:
  DisplaySignalHistoryTask(DisplaySignalHistorySettings settings, holoflow::core::TDesc idesc,
                           QPointer<holovibes::ui::ZernikeHistoryWidget> widget,
                           cudaStream_t                                  stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)),
        dispatcher_(std::make_shared<SignalHistoryDispatcher>(std::move(widget))), stream_(stream) {
    dispatcher_->configure(settings_.indexes, settings_.time_window_seconds);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto              &input = ctx.inputs[0];
    const auto        &desc  = input.desc;
    const auto         count = settings_.indexes.size();
    std::vector<float> values(count);
    auto              *src        = reinterpret_cast<const float *>(input.data());
    const auto         byte_count = count * sizeof(float);

    switch (desc.mem_loc) {
    case holoflow::core::MemLoc::Host:
      std::memcpy(values.data(), src, byte_count);
      break;
    case holoflow::core::MemLoc::Device:
      CUDA_CHECK(cudaMemcpyAsync(values.data(), src, byte_count, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      break;
    default:
      throw std::logic_error("Unsupported memory location for Zernike signal history");
    }

    if (std::ranges::none_of(values, [](float value) { return std::isfinite(value); })) {
      return holoflow::core::OpResult::Ok;
    }

    // Logical sample time prevents variable processing latency and GUI refresh cadence from
    // distorting the acquisition/pipeline timeline represented by consecutive coefficient sets.
    const double sample_time =
        static_cast<double>(valid_sample_index_++) * settings_.sample_time_seconds;
    std::vector<holovibes::ui::ZernikeHistorySample> samples;
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (std::isfinite(values[i])) {
        samples.push_back({settings_.indexes[i], {sample_time, static_cast<double>(values[i])}});
      }
    }
    dispatcher_->enqueue(std::move(samples));
    return holoflow::core::OpResult::Ok;
  }

private:
  DisplaySignalHistorySettings             settings_;
  holoflow::core::TDesc                    idesc_;
  std::shared_ptr<SignalHistoryDispatcher> dispatcher_;
  cudaStream_t                             stream_;
  uint64_t                                 valid_sample_index_ = 0;
};

} // namespace

void to_json(nlohmann::json &j, const DisplaySignalHistorySettings &settings) {
  j = nlohmann::json{
      {"indexes", settings.indexes},
      {"time_window_seconds", settings.time_window_seconds},
      {"sample_time_seconds", settings.sample_time_seconds},
  };
}

void from_json(const nlohmann::json &j, DisplaySignalHistorySettings &settings) {
  j.at("indexes").get_to(settings.indexes);
  j.at("time_window_seconds").get_to(settings.time_window_seconds);
  j.at("sample_time_seconds").get_to(settings.sample_time_seconds);
}

DisplaySignalHistoryFactory::DisplaySignalHistoryFactory(
    holovibes::ui::ZernikeHistoryWidget *widget)
    : widget_(widget) {
  HOLOVIBES_CHECK(widget_ != nullptr,
                  "DisplaySignalHistoryFactory requires a valid widget pointer");
}

holoflow::core::InferResult
DisplaySignalHistoryFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<DisplaySignalHistorySettings>();
  check(std::isfinite(settings.time_window_seconds) && settings.time_window_seconds > 0.0,
        "time_window_seconds must be positive and finite");
  check(std::isfinite(settings.sample_time_seconds) && settings.sample_time_seconds > 0.0,
        "sample_time_seconds must be positive and finite");
  check(!settings.indexes.empty(), "at least one Zernike index is required");
  check(input_descs.size() == 1, "expected exactly one input descriptor");

  const auto &desc = input_descs[0];
  check(desc.rank() == 1, "supports only rank-1 tensors");
  check(desc.shape[0] == settings.indexes.size(),
        "input coefficient count must match configured indexes size");
  check(desc.dtype == holoflow::core::DType::F32, "supports only F32 tensors");

  return holoflow::core::InferResult{
      .input_descs   = {desc},
      .output_descs  = {},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
DisplaySignalHistoryFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings,
                                    const holoflow::core::SyncCreateCtx   &ctx) const {
  auto settings = jsettings.get<DisplaySignalHistorySettings>();
  infer(input_descs, jsettings);
  return std::make_unique<DisplaySignalHistoryTask>(std::move(settings), input_descs[0], widget_,
                                                    ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask> DisplaySignalHistoryFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &jsettings, const holoflow::core::SyncCreateCtx &ctx) const {
  // A rebuilt graph is a new computation run, so a fresh task resets logical sample index to zero.
  return create(input_descs, jsettings, ctx);
}

} // namespace holovibes::tasks::sinks
