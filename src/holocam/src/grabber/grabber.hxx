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

#include <stdexcept>

#include "logger.hh"

namespace holocam {

#define SET(name, value, Type, Catch)                                                              \
  try {                                                                                            \
    auto n = std::string(name);                                                                    \
    ctrl_->set##Type<Module>(n, value);                                                            \
    if constexpr (std::same_as<Module, Euresys::StreamModule>) {                                   \
      if (grabber_b_)                                                                              \
        grabber_b_->setString<Module>(n, value);                                                   \
    }                                                                                              \
    logger()->info("set {}={}", name, value);                                                      \
  } catch (const Euresys::genapi_error &e) {                                                       \
    Catch                                                                                          \
  }

template <typename Module>
void Grabber::set_required<Module>(const char *name, const std::string &value) {
  SET(name, value, String,
      throw std::runtime_error(
          std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
}

template <typename Module> void Grabber::set_required<Module>(const char *name, float value) {
  SET(name, value, Float,
      throw std::runtime_error(
          std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
}

template <typename Module> void Grabber::set_required<Module>(const char *name, int64_t value) {
  SET(name, value, Integer,
      throw std::runtime_error(
          std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));)
}

template <typename Module>
void Grabber::set_required<Module>(const char *name, const std::string &value) {
  SET(name, value, String, logger()->warn("could not set {}={}", name, value));
}

template <typename Module> void Grabber::set_required<Module>(const char *name, float value) {
  SET(name, value, Float, logger()->warn("could not set {}={}", name, value));
}

template <typename Module> void Grabber::set_required<Module>(const char *name, int64_t value) {
  SET(name, value, Integer, logger()->warn("could not set {}={}", name, value));
}

#undef SET
} // namespace holocam
#endif