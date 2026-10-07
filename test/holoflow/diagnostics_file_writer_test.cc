// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/diagnostics.hh"
#include "support/native_trace.hh"
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

TEST(DiagnosticsFileWriterTest, TextRenderingRunsInBackgroundAndCoalescesPendingFiles) {
  const auto            directory = trace_test::temporary_path("holoflow-text-writer-");
  DiagnosticsFileWriter writer;
  std::promise<void>    started, release;
  auto                  starting = started.get_future();
  auto                  released = release.get_future();
  writer.submit_text(directory / "block.txt", [&] {
    started.set_value();
    EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
    return std::string{"block"};
  });
  ASSERT_EQ(starting.wait_for(3s), std::future_status::ready);
  std::vector<int> rendered;
  for (int i = 1; i <= 3; ++i)
    writer.submit_text(directory / "report.txt", [&, i] {
      rendered.push_back(i);
      return std::to_string(i);
    });
  EXPECT_TRUE(rendered.empty());
  release.set_value();
  writer.flush();
  EXPECT_EQ(rendered, (std::vector<int>{3}));
  writer.submit_text(directory / "bad.txt",
                     []() -> std::string { throw std::runtime_error("render"); });
  writer.submit_text(directory / "good.txt", [] { return std::string{"ready"}; });
  writer.flush();
  std::ifstream file(directory / "good.txt");
  std::string   text;
  file >> text;
  EXPECT_EQ(text, "ready");
}

TEST(DiagnosticsFileWriterTest, GraphDumpsOwnSnapshotsAndPreserveSerialization) {
  auto &writer = holoflow::runtime::section_diagnostics_file_writer();
  writer.flush();
  const auto         directory = trace_test::temporary_path("holoflow-graph-snapshots-");
  std::promise<void> started, release;
  auto               starting = started.get_future();
  auto               released = release.get_future();
  writer.submit_text(directory / "block.txt", [&] {
    started.set_value();
    EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
    return std::string{};
  });
  ASSERT_EQ(starting.wait_for(3s), std::future_status::ready);
  std::string expected_json, expected_dot;
  {
    holoflow::core::GraphSpec graph;
    auto a = add_vertex(holoflow::core::NodeSpec{"source", "Source", {{"value", 42}}}, graph);
    auto b = add_vertex(holoflow::core::NodeSpec{"sink", "Sink", {}}, graph);
    add_edge(a, b, holoflow::core::EdgeSpec{0, 0}, graph);
    holoflow::core::GraphSpecDumpPreferences preferences;
    preferences.dump_node_settings = false;
    expected_json                  = holoflow::core::to_json(graph).dump(2);
    expected_dot                   = holoflow::core::to_dot(graph, preferences);
    holoflow::runtime::dump_pipeline_graph_async(directory, graph, preferences);
    holoflow::runtime::dump_graph_spec_async(directory / "graph_spec.json", graph);
    graph[a].settings["value"]     = 99;
    preferences.dump_node_settings = true;
  }
  EXPECT_FALSE(std::filesystem::exists(directory / "pipeline.json"));
  release.set_value();
  writer.flush();
  auto read = [&](const char *name) {
    std::ifstream file(directory / name);
    return std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  };
  EXPECT_EQ(read("pipeline.json"), expected_json);
  EXPECT_EQ(read("graph_spec.json"), expected_json);
  EXPECT_EQ(read("pipeline.dot"), expected_dot);
}

TEST(DiagnosticsFileWriterTest, CompiledGraphDumpOwnsMetadataAndDefersRendering) {
  auto &writer = holoflow::runtime::section_diagnostics_file_writer();
  writer.flush();
  const auto         directory = trace_test::temporary_path("holoflow-compiled-snapshot-");
  std::promise<void> started, release;
  auto               starting = started.get_future();
  auto               released = release.get_future();
  writer.submit_text(directory / "block.txt", [&] {
    started.set_value();
    EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
    return std::string{};
  });
  EXPECT_EQ(starting.wait_for(3s), std::future_status::ready);
  std::string expected;
  {
    holoflow::runtime::CompilerOutput output;
    holoflow::runtime::NodePlan       source{
        .spec     = {"source", "snapshot_source", {{"value", 1.234567}}},
        .infer    = {{}, {}, {}, {}, {}, holoflow::core::TaskKind::Sync},
        .out_tids = {0},
    };
    auto node = add_vertex(source, output.graph);
    output.resources.tid_to_sid.emplace(0, 7);
    output.resources.tasks.emplace("snapshot_task", nullptr);
    output.sections.push_back({.id        = 2,
                               .name      = "snapshot_section",
                               .stream    = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234}),
                               .sync_topo = {node}});
    holoflow::runtime::GraphCompiledDumpPreferences preferences;
    preferences.layout = holoflow::runtime::GraphCompiledDumpPreferences::Layout::Snake;
    preferences.floating_point_precision = 3;
    expected = holoflow::runtime::to_dot(output, preferences, "owned_compiled");
    holoflow::runtime::dump_compiled_graph_async(directory / "compiled.dot", output, preferences,
                                                 "owned_compiled");
    output.graph[node].spec.name = "changed_source";
    output.sections.clear();
    preferences.dump_resource_info = false;
  }
  EXPECT_FALSE(std::filesystem::exists(directory / "compiled.dot"));
  release.set_value();
  writer.flush();
  std::ifstream     file(directory / "compiled.dot");
  const std::string actual{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_EQ(actual, expected);
  EXPECT_NE(actual.find("snapshot_task"), std::string::npos);
  EXPECT_NE(actual.find("snapshot_section"), std::string::npos);
}

} // namespace
