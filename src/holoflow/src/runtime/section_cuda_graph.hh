// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/compiler.hh"

namespace holoflow::runtime {

/// Called after task binding, with the scheduler stopped and all prior work drained.
void build_section_cuda_graphs(CompilerOutput &output, size_t limit);

} // namespace holoflow::runtime
