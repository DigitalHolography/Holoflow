// Copyright 2025 Digital Holography Foundation
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

#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"

#include <string>

namespace holoflow::runtime {

/// Serialize a compiled graph (CompilerOutput) to Graphviz DOT format.
/// This prints:
///  - node labels with name/kind/settings/in_tids/out_tids/infer marker
///  - edge labels with out/in indices, tid and TDesc summary
///  - clusters for Sections (sync/async grouping)
///
/// @param out       Compiled graph output (non-owning reference).
/// @param registry  Registry used to detect async node kinds.
/// @return          DOT source as std::string.
std::string to_dot(const CompilerOutput &out, core::Registry &registry);

} // namespace holoflow::runtime
