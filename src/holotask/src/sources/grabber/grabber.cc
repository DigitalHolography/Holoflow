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

#include "grabber.hh"

#include <EuresysGenapiErrorFormats.h>

#include "logger.hh"

namespace {

using Clock = std::chrono::steady_clock;
// TODO factorise this function
bool log_due(Clock::time_point &last_log) {
  const auto now = Clock::now();
  if (last_log != Clock::time_point{} && now - last_log < std::chrono::seconds(1)) {
    return false;
  }
  last_log = now;
  return true;
}

} // namespace

namespace holotask::sources {

void Grabber::requeue_buffer(const Euresys::NewBufferData &data) noexcept {
  try {
    Euresys::Buffer(data).push(*this);
  } catch (const std::exception &e) {
    if (log_due(last_error_log_)) {
      logger()->error(
          "[Grabber::requeue_buffer_noexcept] failed to requeue buffer on grabber {} while handling an error: {}",
          name_ == EGrabberName::A ? "A" : "B", e.what());
    }
  }
}
} // namespace holotask::sources