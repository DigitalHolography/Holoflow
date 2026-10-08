// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/graph_display.hh"
#include <filesystem>

namespace holoflow::runtime {

// Copy graph values before enqueueing; serialization and file I/O run in the background.
void dump_graph_spec_async(const std::filesystem::path &json_path, const core::GraphSpec &graph);
void dump_pipeline_graph_async(const std::filesystem::path &log_dir, const core::GraphSpec &graph,
                               const core::GraphSpecDumpPreferences &preferences);
// Best-effort capture/submission; background failures also warn without affecting compilation.
void dump_compiled_graph_async(const std::filesystem::path &dot_path, const CompilerOutput &output,
                               const GraphCompiledDumpPreferences &preferences = {},
                               std::string graph_name = "compiled") noexcept;

} // namespace holoflow::runtime
