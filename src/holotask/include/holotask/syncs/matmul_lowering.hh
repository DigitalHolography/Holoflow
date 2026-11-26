// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/runtime/compiler.hh"

namespace holotask::syncs {

std::optional<holoflow::core::GraphSpec> lower_matmul_tasks(const holoflow::runtime::GraphPlan &,
                                                            const holoflow::core::Registry &);

} // namespace holotask::syncs
