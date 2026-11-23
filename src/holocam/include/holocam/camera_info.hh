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

#include <map>
#include <string_view>

namespace holocam {

struct camera_info {
  const char *name;
  const char *short_name;
  const char *model_name;
  float       pixel_size; // in micrometters
};

static constexpr camera_info s710_info = {.name       = "AmetekS710EuresysCoaxlinkOcto",
                                          .short_name = "s710",
                                          .model_name = "Phantom S710",
                                          .pixel_size = 20.0f};

static constexpr camera_info s711_info = {.name       = "AmetekS711EuresysCoaxlinkQSFP+",
                                          .short_name = "s711",
                                          .model_name = "Phantom S711",
                                          .pixel_size = 20.0f};

static constexpr camera_info s991_info = {.name       = "AmetekS991EuresysCoaxlinkQSFP+",
                                          .short_name = "s991",
                                          .model_name = "Phantom S991",
                                          .pixel_size = 6.75f};

#define REGISTER_CAMERA_INFO(info) {info.name, info}
inline const std::map<std::string_view, camera_info> name_to_info = {
    REGISTER_CAMERA_INFO(s710_info), REGISTER_CAMERA_INFO(s711_info),
    REGISTER_CAMERA_INFO(s991_info)};

#undef REGISTER_CAMERA_INFO
} // namespace holocam