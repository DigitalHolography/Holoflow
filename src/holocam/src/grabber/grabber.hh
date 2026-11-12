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
#include <EGrabber.h>

#include <chrono>
#include <memory>
#include <optional>

namespace holocam {

struct FrameBufferPair {
  Euresys::NewBufferData                data_a;
  std::optional<Euresys::NewBufferData> data_b;
};

std::string format_genapi_error(const Euresys::genapi_error &err);

class Grabber {
public:
  Grabber(uint8_t *buffer, size_t buffer_size, size_t buffer_part_count, Euresys::EGrabberInfo info_a,
          std::optional<Euresys::EGrabberInfo> info_b = std::nullopt);
  ~Grabber();

  void requeue(const FrameBufferPair &buffers);

  void start_if_not_running();

  void stop_if_running();

  bool running() const noexcept { return is_running_; }

  void reset_config();

  template <typename Module, typename T> void set_required(const char *name, const T &value);
  template <typename Module, typename T> void set_opt(const char *name, const T &value);

  void set_offsets(int64_t a, int64_t b);

  size_t count() const noexcept {
    return static_cast<bool>(grabber_a_) + static_cast<bool>(grabber_b_);
  }

  FrameBufferPair pop(uint64_t timeout);

  // validate the buffers data and returns its base ptr (or nullptr if not validated)
  std::byte *validate_buffer_data(FrameBufferPair& frame);

private:
  using GrabberType = Euresys::EGrabber<Euresys::CallbackOnDemand>;
  union {
    std::unique_ptr<GrabberType> grabber_a_;
    std::unique_ptr<GrabberType> ctrl_;
  };

  std::unique_ptr<GrabberType> grabber_b_;
  bool                         is_running_ = false;

  using Clock = std::chrono::steady_clock;
  Clock::time_point last_error_log_;
  Clock::time_point last_rejected_log_;
  Clock::time_point last_accepted_log_;
  
  size_t max_ts_delta_since_log_{};
  size_t accepted_since_log_{};

  size_t buffer_part_count_;

};

} // namespace holocam

#endif

#include "grabber.hxx"