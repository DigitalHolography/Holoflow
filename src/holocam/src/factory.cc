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

#include "holocam/factory.hh"

#include <format>
#include <map>
#include <spdlog/fmt/ranges.h>
#include <stdexcept>

#include "logger.hh"

#include "camera/s710.hh"
#include "camera/s711.hh"
#include "camera/s991.hh"

namespace holocam {

std::unique_ptr<CameraBase>
CameraFactory::create(std::string_view name, const nlohmann::json &config,
                      const std::optional<RecordOptions> &record_options) const {

#define CTOR_LAMBDA(T)                                                                             \
  [](const nlohmann::json &c, const std::optional<RecordOptions> &ro) {                            \
    return std::make_unique<T>(c, ro);                                                             \
  }

  using Ctor = std::function<std::unique_ptr<CameraBase>(const nlohmann::json &,
                                                         const std::optional<RecordOptions> &)>;

  static const std::map<std::string_view, Ctor> constructor_map = {
      {"AmetekS710EuresysCoaxlinkOcto", CTOR_LAMBDA(CameraS710)},
      {"AmetekS711EuresysCoaxlinkQSFP+", CTOR_LAMBDA(CameraS711)},
      {"AmetekS991EuresysCoaxlinkQSFP+", CTOR_LAMBDA(CameraS991)}};

#undef CTOR_LAMBDA

  if (!constructor_map.contains(name)) {
    auto msg = std::format("Camera named {} is not supported", name);
    logger()->error(msg);
    std::vector<std::string_view> keys;
    keys.reserve(constructor_map.size());

    for (const auto &[key, ctor] : constructor_map)
      keys.push_back(key);

    logger()->info("supported cameras: {{ {} }}", fmt::join(keys, ", "));

    throw std::runtime_error(msg);
  }

  return constructor_map.at(name)(config, record_options);
}

} // namespace holocam