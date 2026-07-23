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

#include <cstddef>
#include <deque>
#include <optional>

namespace holovibes::ui {

struct SignalSample {
  double time_seconds;
  double value;
};

struct SignalStatistics {
  std::size_t sample_count;
  double      mean;
  double      standard_deviation;
};

class SignalHistory {
public:
  explicit SignalHistory(double time_window_seconds = 8.0);

  void set_time_window_seconds(double time_window_seconds);
  void clear();
  bool append(SignalSample sample);

  [[nodiscard]] double                          time_window_seconds() const;
  [[nodiscard]] const std::deque<SignalSample> &samples() const;
  [[nodiscard]] std::optional<SignalStatistics> statistics(double minimum_time_seconds) const;

private:
  void trim();

  double                   time_window_seconds_;
  std::deque<SignalSample> samples_;
};

} // namespace holovibes::ui
