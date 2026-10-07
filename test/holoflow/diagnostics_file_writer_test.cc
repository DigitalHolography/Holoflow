// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <vector>

#include "../../src/holoflow/src/runtime/diagnostics_file_writer.hh"

namespace {
using holoflow::runtime::DiagnosticsFileWriter;
using namespace std::chrono_literals;

TEST(DiagnosticsFileWriterTest, SubmissionDoesNotWaitAndNewestPendingReportWins) {
  std::promise<void>    started, release;
  auto                  started_future = started.get_future();
  auto                  released       = release.get_future();
  std::vector<int>      written;
  DiagnosticsFileWriter writer([&](const auto &, const auto &report) {
    const int sequence = report.at("sequence").template get<int>();
    if (sequence == 0) {
      started.set_value();
      EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
    }
    written.push_back(sequence);
  });
  writer.submit("diagnostics.json", {{"sequence", 0}});
  ASSERT_EQ(started_future.wait_for(3s), std::future_status::ready);
  // The worker remains blocked throughout these submissions. Equivalent paths coalesce.
  writer.submit("./diagnostics.json", {{"sequence", 1}});
  writer.submit(std::filesystem::absolute("diagnostics.json"), {{"sequence", 2}});
  writer.submit("diagnostics.json", {{"sequence", 3}});
  release.set_value();
  writer.flush();
  EXPECT_EQ(written, (std::vector<int>{0, 3}));
}

TEST(DiagnosticsFileWriterTest, RetainsIndependentFilesAndOwnedSnapshotsUntilShutdown) {
  std::promise<void>         started, release;
  auto                       started_future = started.get_future();
  auto                       released       = release.get_future();
  std::map<std::string, int> written;
  {
    DiagnosticsFileWriter writer([&](const auto &path, const auto &report) {
      const std::string name = path.filename().string();
      if (name == "block.json") {
        started.set_value();
        EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
      }
      written[name] = report.at("value").template get<int>();
    });
    writer.submit("block.json", {{"value", 0}});
    ASSERT_EQ(started_future.wait_for(3s), std::future_status::ready);
    {
      nlohmann::json snapshot{{"value", 42}};
      writer.submit("first.json", snapshot);
      snapshot["value"] = 100;
      writer.submit("second.json", {{"value", 9}});
    } // Snapshot sources are gone before the worker is allowed to write.
    release.set_value();
  } // Destruction must drain both pending files without an explicit flush.
  EXPECT_EQ(written.at("first.json"), 42);
  EXPECT_EQ(written.at("second.json"), 9);
}

TEST(DiagnosticsFileWriterTest, BackgroundFailureDoesNotPreventSubsequentWrites) {
  int                   attempts = 0;
  std::vector<int>      written;
  DiagnosticsFileWriter writer([&](const auto &, const auto &report) {
    ++attempts;
    const int value = report.template get<int>();
    if (value == 1)
      throw std::runtime_error("injected serialization/write failure");
    written.push_back(value);
  });
  writer.submit("report.json", 1);
  writer.flush();
  writer.submit("report.json", 2);
  writer.flush();
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(written, (std::vector<int>{2}));
}

TEST(DiagnosticsFileWriterTest, RealIoFailureStillAllowsAReadableReport) {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("holoflow-diagnostics-writer-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(directory);
  DiagnosticsFileWriter writer;
  writer.submit(directory, {"cannot overwrite a directory"});
  writer.flush();
  const nlohmann::json expected{{"status", "ready"}, {"variants", 3}};
  writer.submit(directory / "report.json", expected);
  writer.flush();
  {
    std::ifstream  file(directory / "report.json");
    nlohmann::json actual;
    file >> actual;
    EXPECT_EQ(actual, expected);
  }
  std::filesystem::remove_all(directory);
}

} // namespace
