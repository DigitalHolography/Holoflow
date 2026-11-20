// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "../../../../src/python/latency.hh"
#include <gtest/gtest.h>

using holoflow::python::LatencyMeasurement;
using namespace std::chrono_literals;

TEST(LatencyMeasurementTest, CorrelatesOffsetAndOverlappingWindowsAcrossBatches) {
  LatencyMeasurement                          measurement("end", 4, 8, 2, 4, 2, 500);
  const LatencyMeasurement::Clock::time_point origin{};
  measurement.start(origin);
  measurement.input(origin + 10ms);
  measurement.input(origin + 20ms);
  measurement.output(origin + 25ms); // Frames 2–5.
  measurement.output(origin + 30ms); // Frames 4–7.
  const auto stats = measurement.sample(origin + 500ms);
  EXPECT_EQ(stats.at("matched_outputs"), 2);
  EXPECT_EQ(stats.at("oldest_ms").at("mean"), 12.5);
  EXPECT_EQ(stats.at("newest_ms").at("mean"), 7.5);
  EXPECT_EQ(stats.at("oldest_ms").at("max"), 15);
  EXPECT_EQ(stats.at("newest_ms").at("min"), 5);
  EXPECT_EQ(measurement.sample(origin + 1s).at("matched_outputs"), 0);
  EXPECT_EQ(measurement.snapshot().at("matched_outputs"), 2);
}

TEST(LatencyMeasurementTest, CountsExpiredAndUnavailableSourceTimestamps) {
  LatencyMeasurement                          measurement("end", 4, 1, 0, 4, 4, 500);
  const LatencyMeasurement::Clock::time_point origin{};
  measurement.input(origin + 1ms);
  measurement.input(origin + 2ms);
  measurement.output(origin + 3ms); // Frames 0–3 have expired.
  measurement.output(origin + 4ms); // Frames 4–7 are retained.
  measurement.output(origin + 5ms); // Frames 8–11 have not arrived.
  const auto stats = measurement.snapshot();
  EXPECT_EQ(stats.at("matched_outputs"), 1);
  EXPECT_EQ(stats.at("unmatched_outputs"), 2);
  EXPECT_EQ(stats.at("oldest_ms").at("mean"), 2);
  EXPECT_EQ(stats.at("oldest_ms"), stats.at("newest_ms"));
}
