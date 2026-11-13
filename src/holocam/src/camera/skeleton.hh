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

#pragma once

#define CAMERA_DECL(class_name, camera_name, camera_short_name, camera_model_name,                 \
                    camera_grabber_count)                                                          \
  namespace holocam {                                                                              \
  class Camera##class_name final : public CameraBase {                                             \
  public:                                                                                          \
    Camera##class_name(const nlohmann::json               &config,                                 \
                       const std::optional<RecordOptions> &record_options)                         \
        : CameraBase(config, record_options, camera_short_name, camera_model_name,                 \
                     camera_grabber_count, &configure) {}                                          \
    const char *name() const noexcept override { return camera_name; }                             \
    const char *short_name() const noexcept override { return camera_short_name; }                 \
    const char *model_name() const noexcept override { return camera_model_name; }                 \
    size_t      grabber_count() const noexcept override { return camera_grabber_count; }           \
                                                                                                   \
  protected:                                                                                       \
    static void configure(Impl *impl, const nlohmann::json &cfg);                                      \
  };                                                                                               \
  } // namespace holocam