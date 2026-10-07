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

} // namespace holoflow::runtime
