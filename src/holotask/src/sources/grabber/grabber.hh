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

#include <EGrabber.h>
#include <chrono>

namespace holotask::sources {

class Grabber : public Euresys::EGrabber<Euresys::CallbackOnDemand> {
public:
  using Clock = std::chrono::steady_clock;
  enum class EGrabberName : uint8_t { A = 0, B = 1 };

  Grabber(Euresys::EGrabberInfo info, EGrabberName name)
      : EGrabber<Euresys::CallbackOnDemand>(info), name_{name} {
    enableEvent<Euresys::NewBufferData>();
  }

  void requeue_buffer(const Euresys::NewBufferData &data) noexcept;

private:
  Clock::time_point last_error_log_;
  EGrabberName      name_;
};

} // namespace holotask::sources