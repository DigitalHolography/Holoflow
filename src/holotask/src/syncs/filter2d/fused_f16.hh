// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "holotask/syncs/filter2d.hh"
namespace holotask::syncs {
bool                                       supports_fused_f16_filter(const holoflow::core::TDesc &);
std::unique_ptr<holoflow::core::ISyncTask> create_fused_f16_filter(const holoflow::core::TDesc &,
                                                                   Filter2DSettings, cudaStream_t);
bool update_fused_f16_filter(holoflow::core::ISyncTask &, const holoflow::core::TDesc &,
                             Filter2DSettings, cudaStream_t);
} // namespace holotask::syncs
