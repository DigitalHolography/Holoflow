// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

/// Inspection at compilation; eager preparation on every start, before any worker is created.
void refresh_section_cuda_graphs(const GraphPlan &graph, const std::vector<Section> &sections,
                                 ExecResouces &resources, bool instantiate);
void write_section_cuda_graph_diagnostics(const ExecResouces &resources);

} // namespace holoflow::runtime
