// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

// -------------------------------------------------------------------------------------------------
// Section graph refresh
// -------------------------------------------------------------------------------------------------

/// Inspect at compilation and retain a one-use plan. First startup prepares that plan;
/// later non-paused starts inspect current queue phases before preparing variants.
void refresh_section_cuda_graphs(const GraphPlan &graph, const std::vector<Section> &sections,
                                 ExecResouces &resources, bool instantiate);

// ---- Diagnostics -------------------------------------------------------------------------------

/// Queue an owned copy of current section snapshots when a diagnostics directory is configured.
void write_section_cuda_graph_diagnostics(const ExecResouces &resources);

} // namespace holoflow::runtime
