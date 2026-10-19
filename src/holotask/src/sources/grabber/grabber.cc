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

namespace holotask::sources {

void Grabber::requeue_buffer(const Euresys::NewBufferData &data) noexcept {
  try {
    Euresys::Buffer(data).push(*this);
  } catch (const std::exception &e) {
    if (log_due(last_error_log_)) {
      logger()->error("[Grabber::requeue_buffer_noexcept] failed to requeue buffer on grabber {} "
                      "while handling an error: {}",
                      name_ == EGrabberName::A ? "A" : "B", e.what());
    }
  }
}
  void Grabber::start_if_not_running() {
    if (!is_running_) {
      start();
      logger()->info("[Grabber::start_if_not_running] started {}", name());
      is_running_ = true;
    }
  }

  void Grabber::stop_if_running() {
    if (is_running_) {
      try {
        stop();
        logger()->info("[Grabber::stop_if_running] stopped {}", name());
      } catch (const std::exception &e) {
        logger()->error("[Grabber::stop_if_running] failed to stop {}: {}", name(), e.what());
      }
      is_running_ = false;
    }
  }
} // namespace holotask::sources