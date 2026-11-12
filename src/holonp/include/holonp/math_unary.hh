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

#include "holoflow/core/tasks.hh"
#include <nlohmann/json.hpp>
#include <span>

namespace holonp {
struct UnaryMathSettings {
  bool operator==(const UnaryMathSettings &) const = default;
};

void to_json(nlohmann::json &, const UnaryMathSettings &);

void from_json(const nlohmann::json &, UnaryMathSettings &);

#define HOLONP_UNARY_FACTORY(Name)                                                                 \
  class Name##Factory : public holoflow::core::ISyncTaskFactory {                                  \
  public:                                                                                          \
    holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,                      \
                                      const nlohmann::json &) const override;                      \
    std::unique_ptr<holoflow::core::ISyncTask>                                                     \
    create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,                         \
           const holoflow::core::SyncCreateCtx &) const override;                                  \
    std::unique_ptr<holoflow::core::ISyncTask>                                                     \
    update(std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc>,     \
           const nlohmann::json &, const holoflow::core::SyncCreateCtx &) const override;          \
  };

using SqrtSettings     = UnaryMathSettings;
using RealSettings     = UnaryMathSettings;
using ImagSettings     = UnaryMathSettings;
using AngleSettings    = UnaryMathSettings;
using LogSettings      = UnaryMathSettings;
using IsfiniteSettings = UnaryMathSettings;

HOLONP_UNARY_FACTORY(Sqrt)
HOLONP_UNARY_FACTORY(Real)
HOLONP_UNARY_FACTORY(Imag)
HOLONP_UNARY_FACTORY(Angle)
HOLONP_UNARY_FACTORY(Log)
HOLONP_UNARY_FACTORY(Isfinite)

} // namespace holonp
