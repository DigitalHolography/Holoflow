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

#include "display_tensor.hh"

#include <QMetaObject>
#include <QtGlobal>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "bug.hh"
#include "cuda_runtime_api.h"
#include "curaii/cuda.hh"
#include "driver_types.h"
#include "holoflow/core/tensor.hh"
#include "logger.hh"
#include "ui/widgets/tensor_display_widget.hh"

#include <cuda_runtime.h>

namespace holovibes::tasks::sinks {

void to_json(nlohmann::json &j, const DisplayTensorSettings &ds) {
  j = nlohmann::json{
      {"refresh_rate_hz", ds.refresh_rate_hz},
  };
}

void from_json(const nlohmann::json &j, DisplayTensorSettings &ds) {
  j.at("refresh_rate_hz").get_to(ds.refresh_rate_hz);
}

DisplayTensorTask::DisplayTensorTask(DisplayTensorSettings                        settings,
                                     QPointer<holovibes::ui::TensorDisplayWidget> widget,
                                     cudaStream_t                                 stream)
    : settings_(std::move(settings)), next_refresh_(std::chrono::steady_clock::now()),
      widget_(std::move(widget)), stream_(stream) {}

holoflow::core::OpResult DisplayTensorTask::execute(holoflow::core::SyncCtx &ctx) {
  if (widget_.isNull()) {
    logger()->warn("[DisplayTensorTask::execute] target widget is not available");
    return holoflow::core::OpResult::Ok;
  }

  auto now = std::chrono::steady_clock::now();
  if (now < next_refresh_) {
    return holoflow::core::OpResult::Ok;
  }
  auto delta_ms = static_cast<int>(1000 / settings_.refresh_rate_hz);
  next_refresh_ = now + std::chrono::milliseconds(delta_ms);

  const auto  &input     = ctx.inputs[0];
  const auto  &desc      = input.desc;
  const size_t height    = desc.rank() == 2 ? desc.shape[0] : desc.shape[1];
  const size_t width     = desc.rank() == 2 ? desc.shape[1] : desc.shape[2];
  const size_t elt_count = width * height;
  size_t       byte_count;

  switch (desc.dtype) {
  case holoflow::core::DType::U16: {
    byte_count = 2;
  } break;

  default:
    byte_count = 1;
  }

  QByteArray payload;
  payload.resize(static_cast<qsizetype>(byte_count * elt_count));
  void *dst = payload.data();

  switch (desc.mem_loc) {
  case holoflow::core::MemLoc::Host: {
    const void *src = static_cast<const void *>(input.data);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, byte_count * elt_count, cudaMemcpyHostToHost, stream_));
  } break;

  case holoflow::core::MemLoc::Device: {
    const void *src = static_cast<const void *>(input.data);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, byte_count * elt_count, cudaMemcpyDeviceToHost, stream_));
  } break;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream_));
  dispatchToUi(std::move(payload), static_cast<int>(width), static_cast<int>(height), desc.dtype);
  return holoflow::core::OpResult::Ok;
}

void DisplayTensorTask::dispatchToUi(QByteArray payload, int width, int height,
                                     holoflow::core::DType dtype) {
  if (widget_.isNull()) {
    logger()->warn("[DisplayTensorTask::dispatchToUi] target widget is not available");
    return;
  }

  QPointer<holovibes::ui::TensorDisplayWidget> safe_widget = widget_;
  QMetaObject::invokeMethod(
      widget_.data(),
      [this, safe_widget, payload = std::move(payload), width, height, dtype]() mutable {
        if (safe_widget.isNull()) {
          logger()->warn("[DisplayTensorTask::dispatchToUi] target widget is not available");
          return;
        }
        safe_widget->presentTensor(payload, width, height, dtype);
      },
      Qt::QueuedConnection);
}

DisplayTensorFactory::DisplayTensorFactory(holovibes::ui::TensorDisplayWidget *widget)
    : widget_(widget) {
  HOLOVIBES_CHECK(widget_ != nullptr, "DisplayTensorFactory requires a valid widget pointer");
}

holoflow::core::InferResult
DisplayTensorFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json &) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[DisplayTensorFactory::infer] error: {}", msg);
      throw std::invalid_argument("DisplayTensorFactory inference error: " + msg);
    }
  };

  // Validate
  check(input_descs.size() == 1, "DisplayTensorFactory expects exactly one input descriptor");
  const auto &desc         = input_descs[0];
  const bool  is_2d_tensor = desc.rank() == 2 || (desc.rank() == 3 && desc.shape[0] == 1);
  check(is_2d_tensor, "DisplayTensorFactory supports only 2D tensors");
  check(desc.dtype == holoflow::core::DType::U8 || desc.dtype == holoflow::core::DType::U16,
        "DisplayTensorFactory supports only u8 or u16 tensors");

  // Success
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
DisplayTensorFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                             const nlohmann::json                  &jsettings,
                             const holoflow::core::SyncCreateCtx   &ctx) const {
  auto settings = jsettings.get<DisplayTensorSettings>();
  infer(input_descs, jsettings);
  auto *task = new DisplayTensorTask(settings, widget_, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::sinks
