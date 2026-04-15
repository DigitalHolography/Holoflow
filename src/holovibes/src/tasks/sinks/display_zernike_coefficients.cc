// Copyright 2025 Digital Holography Foundation
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

#include "display_zernike_coefficients.hh"

#include <QMetaObject>
#include <QPointer>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "bug.hh"
#include "cuda_runtime_api.h"
#include "logger.hh"
#include "ui/widgets/auto_focus_widget.hh"

namespace holovibes::tasks::sinks {

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[DisplayZernikeCoefficientsFactory] {}", msg);
    throw std::invalid_argument("DisplayZernikeCoefficientsFactory: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  size_t expected = size_of(desc.dtype);
  for (size_t i = desc.rank(); i-- > 0;) {
    if (desc.strides[i] != expected)
      return false;
    expected *= desc.shape[i];
  }
  return true;
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc;
}

// -------------------------------------------------------------------------------------------------
// DisplayZernikeCoefficientsTask
// -------------------------------------------------------------------------------------------------

class DisplayZernikeCoefficientsTask : public holoflow::core::ISyncTask {
public:
  DisplayZernikeCoefficientsTask(DisplayZernikeCoefficientsSettings       settings,
                                 holoflow::core::TDesc                    idesc,
                                 QPointer<holovibes::ui::AutoFocusWidget> widget,
                                 cudaStream_t                             stream);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc              &idesc() const;
  const DisplayZernikeCoefficientsSettings &settings() const;
  void                                      update_stream(cudaStream_t stream);

private:
  void dispatch_to_ui(std::vector<float> values);

  DisplayZernikeCoefficientsSettings       settings_;
  holoflow::core::TDesc                    idesc_;
  std::chrono::steady_clock::time_point    next_refresh_;
  QPointer<holovibes::ui::AutoFocusWidget> widget_;
  cudaStream_t                             stream_;
};

} // namespace

void to_json(nlohmann::json &j, const DisplayZernikeCoefficientsSettings &settings) {
  j = nlohmann::json{
      {"indexes", settings.indexes},
      {"refresh_rate_hz", settings.refresh_rate_hz},
  };
}

void from_json(const nlohmann::json &j, DisplayZernikeCoefficientsSettings &settings) {
  j.at("indexes").get_to(settings.indexes);
  j.at("refresh_rate_hz").get_to(settings.refresh_rate_hz);
}

DisplayZernikeCoefficientsTask::DisplayZernikeCoefficientsTask(
    DisplayZernikeCoefficientsSettings settings, holoflow::core::TDesc idesc,
    QPointer<holovibes::ui::AutoFocusWidget> widget, cudaStream_t stream)
    : settings_(std::move(settings)), idesc_(std::move(idesc)),
      next_refresh_(std::chrono::steady_clock::now()), widget_(std::move(widget)), stream_(stream) {
}

const holoflow::core::TDesc &DisplayZernikeCoefficientsTask::idesc() const { return idesc_; }
const DisplayZernikeCoefficientsSettings &DisplayZernikeCoefficientsTask::settings() const {
  return settings_;
}
void DisplayZernikeCoefficientsTask::update_stream(cudaStream_t stream) { stream_ = stream; }

holoflow::core::OpResult DisplayZernikeCoefficientsTask::execute(holoflow::core::SyncCtx &ctx) {
  if (widget_.isNull()) {
    logger()->warn("[DisplayZernikeCoefficientsTask::execute] target widget is not available");
    return holoflow::core::OpResult::Ok;
  }

  auto now = std::chrono::steady_clock::now();
  if (now < next_refresh_) {
    return holoflow::core::OpResult::Ok;
  }
  auto delta_ms = static_cast<int>(1000 / settings_.refresh_rate_hz);
  next_refresh_ = now + std::chrono::milliseconds(delta_ms);

  auto       &input = ctx.inputs[0];
  const auto &desc  = input.desc;
  const auto  count = static_cast<size_t>(desc.shape[0]);

  std::vector<float> values(count);
  auto              *dst        = values.data();
  auto              *src        = reinterpret_cast<float *>(input.data());
  const auto         byte_count = count * sizeof(float);

  switch (desc.mem_loc) {
  case holoflow::core::MemLoc::Host:
    std::memcpy(dst, src, byte_count);
    break;
  case holoflow::core::MemLoc::Device:
    CUDA_CHECK(cudaMemcpyAsync(dst, src, byte_count, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    break;
  default:
    throw std::logic_error("Unsupported memory location for zernike coefficient display");
  }

  dispatch_to_ui(std::move(values));
  return holoflow::core::OpResult::Ok;
}

void DisplayZernikeCoefficientsTask::dispatch_to_ui(std::vector<float> values) {
  if (widget_.isNull()) {
    logger()->warn("[DisplayZernikeCoefficientsTask::dispatchToUi] target widget is not available");
    return;
  }

  QPointer<holovibes::ui::AutoFocusWidget> safe_widget = widget_;
  auto                                     indexes     = settings_.indexes;
  QMetaObject::invokeMethod(
      widget_.data(),
      [safe_widget, indexes = std::move(indexes), values = std::move(values)]() mutable {
        if (safe_widget.isNull()) {
          return;
        }
        safe_widget->set_zernike_values(indexes, values);
      },
      Qt::QueuedConnection);
}

DisplayZernikeCoefficientsFactory::DisplayZernikeCoefficientsFactory(
    holovibes::ui::AutoFocusWidget *widget)
    : widget_(widget) {
  HOLOVIBES_CHECK(widget_ != nullptr,
                  "DisplayZernikeCoefficientsFactory requires a valid widget pointer");
}

holoflow::core::InferResult
DisplayZernikeCoefficientsFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                         const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<DisplayZernikeCoefficientsSettings>();
  check(settings.refresh_rate_hz > 0.0f, "refresh_rate_hz must be positive");

  check(input_descs.size() == 1, "expected exactly one input descriptor");

  const auto &desc = input_descs[0];
  check(desc.rank() == 1, "supports only rank-1 tensors");
  check(desc.dtype == holoflow::core::DType::F32, "supports only F32 tensors");
  check(is_c_contiguous(desc), "input must be C-contiguous");
  check(desc.shape[0] == settings.indexes.size(),
        "Input coefficient count must match configured indexes size");

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
DisplayZernikeCoefficientsFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                          const nlohmann::json                  &jsettings,
                                          const holoflow::core::SyncCreateCtx   &ctx) const {
  auto settings = jsettings.get<DisplayZernikeCoefficientsSettings>();
  infer(input_descs, jsettings);
  return std::make_unique<DisplayZernikeCoefficientsTask>(std::move(settings), input_descs[0],
                                                          widget_, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
DisplayZernikeCoefficientsFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                          std::span<const holoflow::core::TDesc>     input_descs,
                                          const nlohmann::json                      &jsettings,
                                          const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old = dynamic_cast<DisplayZernikeCoefficientsTask *>(old_task.get());
  if (old == nullptr)
    return create(input_descs, jsettings, ctx);

  infer(input_descs, jsettings);
  auto settings = jsettings.get<DisplayZernikeCoefficientsSettings>();
  if (settings == old->settings() && same_desc(input_descs[0], old->idesc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holovibes::tasks::sinks
