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

template <typename Module>
void Grabber::set_required<Module>(const char *name, const std::string &value) {
  try {
    ctrl_->setString<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));
  }
}

template <typename Module> void Grabber::set_required<Module>(const char *name, float value) {
  try {
    ctrl_->setFloat<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));
  }
}

template <typename Module> void Grabber::set_required<Module>(const char *name, int value) {
  try {
    ctrl_->setInt<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("failed to set {}={}: {}", name, value, format_genapi_error(e)));
  }
}

template <typename Module> void Grabber::set_required<Module>(const char *name, const std::string& value) {
  try {
    ctrl_->setString<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    logger()->warn("could not set {}={}", name, value);
  }
}

template <typename Module> void Grabber::set_required<Module>(const char *name, float value) {
  try {
    ctrl_->setFloat<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    logger()->warn("could not set {}={}", name, value);
  }
}

template <typename Module> void Grabber::set_required<Module>(const char *name, int value) {
  try {
    ctrl_->setInt<Module>(std::string(name), value);
    logger()->info("set {}={}", name, value);
  } catch (const Euresys::genapi_error &e) {
    logger()->warn("could not set {}={}", name, value);
  }
}

} // namespace holocam
#endif