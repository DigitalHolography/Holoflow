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

#include "signal_history.hh"

#include <cmath>
#include <stdexcept>

namespace holovibes::ui {

SignalHistory::SignalHistory(double time_window_seconds)
    : time_window_seconds_(time_window_seconds) {
  if (!std::isfinite(time_window_seconds_) || time_window_seconds_ <= 0.0) {
    throw std::invalid_argument("signal history time window must be positive and finite");
  }
}

void SignalHistory::set_time_window_seconds(double time_window_seconds) {
  if (!std::isfinite(time_window_seconds) || time_window_seconds <= 0.0) {
    throw std::invalid_argument("signal history time window must be positive and finite");
  }

  time_window_seconds_ = time_window_seconds;
  trim();
}

void SignalHistory::clear() { samples_.clear(); }

bool SignalHistory::append(SignalSample sample) {
  if (!std::isfinite(sample.time_seconds) || !std::isfinite(sample.value)) {
    return false;
  }
  if (!samples_.empty() && sample.time_seconds < samples_.back().time_seconds) {
    return false;
  }

  samples_.push_back(sample);
  trim();
  return true;
}

double SignalHistory::time_window_seconds() const { return time_window_seconds_; }

const std::deque<SignalSample> &SignalHistory::samples() const { return samples_; }

void SignalHistory::trim() {
  if (samples_.empty()) {
    return;
  }

  const double oldest_time = samples_.back().time_seconds - time_window_seconds_;
  while (!samples_.empty() && samples_.front().time_seconds < oldest_time) {
    samples_.pop_front();
  }
}

} // namespace holovibes::ui
