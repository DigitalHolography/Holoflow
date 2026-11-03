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

#ifdef HOLOTASK_HAS_EGRABBER

#include <EGrabber.h>
#include <chrono>

namespace holotask::sources {

class Grabber final : public Euresys::EGrabber<Euresys::CallbackOnDemand> {
public:
  using parent = Euresys::EGrabber<Euresys::CallbackOnDemand>;

  using Clock = std::chrono::steady_clock;
  enum class EGrabberName : uint8_t { A = 0, B = 1 };

  Grabber(Euresys::EGrabberInfo info, EGrabberName name) : parent(info), name_{name} {
    enableEvent<Euresys::NewBufferData>();
  }

  const char *name() const noexcept { return name_ == EGrabberName::A ? "A" : "B"; }

  void requeue_buffer(const Euresys::NewBufferData &data) noexcept;

  void start_if_not_running();

  void stop_if_running();

  bool running() const noexcept { return is_running_; }

private:
  Clock::time_point last_error_log_;
  EGrabberName      name_;
  bool              is_running_ = false;
};

} // namespace holotask::sources

#endif // HOLOTASK_HAS_EGRABBER
