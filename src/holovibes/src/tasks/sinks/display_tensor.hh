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

#pragma once

#include <QByteArray>
#include <QPointer>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holovibes::ui {
class TensorDisplayWidget;
}

namespace holovibes::tasks::sinks {

/// @brief Settings for the DisplayTensor task.
/// @details
/// @code{.json}
/// {}
/// @endcode
struct DisplayTensorSettings {
  float refresh_rate_hz = 30.0f;
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref DisplayTensorSettings.
/// @{
void to_json(nlohmann::json &j, const DisplayTensorSettings &dts);
void from_json(const nlohmann::json &j, DisplayTensorSettings &dts);
/// @}

/// @brief Synchronous task that displays an input tensor in a widget.
/// @details
/// Expects a single input tensor of shape (H, W) with dtype uint8.
///
/// @par CUDA
/// Uses the stream provided by the runtime in @ref SyncCtx to perform D2H copies
/// if the input tensor lives on the GPU. The stream is not owned.
///
/// @par Errors
/// - CUDA runtime errors from transfers operations.
class DisplayTensorTask : public holoflow::core::ISyncTask {
public:
  explicit DisplayTensorTask(DisplayTensorSettings                        settings,
                             QPointer<holovibes::ui::TensorDisplayWidget> widget,
                             cudaStream_t                                 stream);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  void dispatchToUi(QByteArray payload, int width, int height, holoflow::core::DType dtype);

  DisplayTensorSettings                        settings_;
  std::chrono::steady_clock::time_point        next_refresh_;
  QPointer<holovibes::ui::TensorDisplayWidget> widget_;
  cudaStream_t                                 stream_;
};

/// Factory that builds DisplayTensorTask instances bound to a widget instance.
class DisplayTensorFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit DisplayTensorFactory(holovibes::ui::TensorDisplayWidget *widget);

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

private:
  holovibes::ui::TensorDisplayWidget *widget_;
};

} // namespace holovibes::tasks::sinks
