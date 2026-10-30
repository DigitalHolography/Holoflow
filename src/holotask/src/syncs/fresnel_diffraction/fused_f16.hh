// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "holotask/syncs/fresnel_diffraction.hh"
namespace holotask::syncs {
bool supports_fused_f16_fresnel(const holoflow::core::TDesc &, const FresnelDiffractionSettings &);
std::unique_ptr<holoflow::core::ISyncTask>
create_fused_f16_fresnel(const holoflow::core::TDesc &, const FresnelDiffractionSettings &,
                         const holoflow::core::SyncCreateCtx &);
} // namespace holotask::syncs
