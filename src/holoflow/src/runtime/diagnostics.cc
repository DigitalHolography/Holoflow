// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holoflow/runtime/diagnostics.hh"
#include "../logger.hh"
#include "diagnostics_file_writer.hh"
#include "holoflow/runtime/tracing.hh"
#include <memory>

namespace holoflow::runtime {
namespace {
void submit_json(const std::filesystem::path           &path,
                 std::shared_ptr<const core::GraphSpec> snapshot) {
  section_diagnostics_file_writer().submit_text(path, [snapshot = std::move(snapshot)] {
    tracing::ScopedTrace format("Format Graph Spec JSON", "detail");
    return core::to_json(*snapshot).dump(2);
  });
}
} // namespace

void dump_graph_spec_async(const std::filesystem::path &json_path, const core::GraphSpec &graph) {
  tracing::ScopedTrace snapshot_trace("Copy Graph Spec Snapshot", "detail");
  submit_json(json_path, std::make_shared<const core::GraphSpec>(graph));
}

void dump_pipeline_graph_async(const std::filesystem::path &log_dir, const core::GraphSpec &graph,
                               const core::GraphSpecDumpPreferences &preferences) {
  tracing::ScopedTrace snapshot_trace("Copy Pipeline Graph Snapshot", "detail");
  auto                 snapshot = std::make_shared<const core::GraphSpec>(graph);
  submit_json(log_dir / "pipeline.json", snapshot);
  section_diagnostics_file_writer().submit_text(log_dir / "pipeline.dot", [snapshot, preferences] {
    tracing::ScopedTrace format("Format Pipeline Graph DOT", "detail");
    return core::to_dot(*snapshot, preferences);
  });
}

void dump_compiled_graph_async(const std::filesystem::path &dot_path, const CompilerOutput &output,
                               const GraphCompiledDumpPreferences &preferences,
                               std::string                         graph_name) noexcept {
  try {
    std::shared_ptr<const GraphCompiledDumpSnapshot> snapshot;
    {
      tracing::ScopedTrace copy("Copy Compiled Graph Snapshot", "detail");
      snapshot = make_graph_compiled_dump_snapshot(output);
    }
    tracing::ScopedTrace submit("Submit Compiled Graph Dump", "detail");
    section_diagnostics_file_writer().submit_text(
        dot_path,
        [snapshot = std::move(snapshot), preferences, graph_name = std::move(graph_name)] {
          tracing::ScopedTrace format("Format Compiled Graph DOT", "detail");
          return to_dot(*snapshot, preferences, graph_name);
        });
  } catch (...) {
    try {
      holoflow::logger()->warn("Could not submit compiled graph dump for background writing");
    } catch (...) {
    }
  }
}
} // namespace holoflow::runtime
