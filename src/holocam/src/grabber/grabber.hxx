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

#ifdef HOLOCAM_HAS_EGRABBER

#include "grabber.hh"

#include <concepts>
#include <stdexcept>

#include "logger.hh"

namespace holocam {
template <typename> inline constexpr bool always_false_v = false;

#define SET(name, value, Type, Catch)                                                              \
  try {                                                                                            \
    auto n = std::string(name);                                                                    \
    ctrl()->set##Type<Module>(n, value);                                                            \
    if constexpr (std::same_as<Module, Euresys::StreamModule>) {                                   \
      if (grabber_b_)                                                                              \
        grabber_b_->set##Type<Module>(n, value);                                                   \
    }                                                                                              \
    logger()->info("set {}={}", name, value);                                                      \
  } catch (const Euresys::genapi_error &e) {                                                       \
    (void)e;                                                                                       \
    Catch                                                                                          \
  }

template <typename Module, typename T>
void Grabber::set_required(const char *name, const T &value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_integral_v<U>) {
    SET(name, value, Integer,
        throw std::runtime_error(
            std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
  } else if constexpr (std::is_floating_point_v<U>) {
    SET(name, value, Float,
        throw std::runtime_error(
            std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
  } else if constexpr (std::is_convertible_v<const T &, std::string>) {
    const std::string v(value);
    SET(name, v, String,
        throw std::runtime_error(
            std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
  } else {
    static_assert(always_false_v<T>, "unsupported type for set_required");
  }
}

template <typename Module, typename T> void Grabber::set_opt(const char *name, const T &value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_integral_v<U>) {
    SET(name, value, Integer, logger()->warn("could not set {}={}", name, value);)
  } else if constexpr (std::is_floating_point_v<U>) {
    SET(name, value, Float, logger()->warn("could not set {}={}", name, value);)
  } else if constexpr (std::is_convertible_v<const T &, std::string>) {
    const std::string v(value);
    SET(name, v, String, logger()->warn("could not set {}={}", name, value);)
  } else {
    static_assert(always_false_v<T>, "unsupported type for set_required");
  }
}

#undef SET
} // namespace holocam
#endif