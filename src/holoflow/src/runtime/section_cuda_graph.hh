// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

// -------------------------------------------------------------------------------------------------
// Section graph refresh
// -------------------------------------------------------------------------------------------------

/// Inspect at compilation; when instantiate is true, prepare eagerly before workers start.
void refresh_section_cuda_graphs(const GraphPlan &graph, const std::vector<Section> &sections,
                                 ExecResouces &resources, bool instantiate);

// ---- Diagnostics -------------------------------------------------------------------------------

/// Queue an owned copy of current section snapshots when a diagnostics directory is configured.
void write_section_cuda_graph_diagnostics(const ExecResouces &resources);

} // namespace holoflow::runtime
