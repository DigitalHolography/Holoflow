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

#include <gtest/gtest.h>

#include <limits>

#include "ui/widgets/signal_history.hh"

namespace holovibes::ui {

TEST(SignalHistoryTest, KeepsOnlySamplesInsideLatestTimeWindow) {
  SignalHistory history(2.0);

  EXPECT_TRUE(history.append({0.0, 1.0}));
  EXPECT_TRUE(history.append({1.0, 2.0}));
  EXPECT_TRUE(history.append({2.0, 3.0}));
  EXPECT_TRUE(history.append({3.0, 4.0}));

  ASSERT_EQ(history.samples().size(), 3);
  EXPECT_DOUBLE_EQ(history.samples().front().time_seconds, 1.0);
  EXPECT_DOUBLE_EQ(history.samples().back().time_seconds, 3.0);
}

TEST(SignalHistoryTest, TimeWindowChangeImmediatelyTrimsExistingSamples) {
  SignalHistory history(8.0);
  history.append({0.0, 1.0});
  history.append({2.0, 2.0});
  history.append({4.0, 3.0});

  history.set_time_window_seconds(1.5);

  ASSERT_EQ(history.samples().size(), 1);
  EXPECT_DOUBLE_EQ(history.samples().front().time_seconds, 4.0);
}

TEST(SignalHistoryTest, RejectsInvalidSamplesWithoutChangingHistory) {
  SignalHistory history;
  EXPECT_TRUE(history.append({1.0, 2.0}));
  EXPECT_FALSE(history.append({0.5, 3.0}));
  EXPECT_FALSE(history.append({2.0, std::numeric_limits<double>::quiet_NaN()}));
  EXPECT_FALSE(history.append({std::numeric_limits<double>::infinity(), 4.0}));
  EXPECT_EQ(history.samples().size(), 1);
}

TEST(SignalHistoryTest, RejectsNonPositiveOrNonFiniteTimeWindows) {
  EXPECT_THROW((void)SignalHistory{0.0}, std::invalid_argument);
  EXPECT_THROW((void)SignalHistory{-1.0}, std::invalid_argument);
  EXPECT_THROW((void)SignalHistory{std::numeric_limits<double>::infinity()}, std::invalid_argument);
}

} // namespace holovibes::ui
