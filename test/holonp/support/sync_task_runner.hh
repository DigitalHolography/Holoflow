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

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holonp_test {

struct RunResult {
  std::vector<std::vector<std::byte>> output_bytes; ///< Dense host bytes per output
  std::vector<holoflow::core::TDesc>  output_descs; ///< Output descriptors from infer
};

// -------------------------------------------------------------------------------------------------
// run_sync_factory
// -------------------------------------------------------------------------------------------------
//
// Full path: infer → create → upload inputs → execute → stream sync → download outputs.
// input_host_data[i] must contain exactly input_descs[i].num_bytes() bytes.
//
RunResult run_sync_factory(const holoflow::core::ISyncTaskFactory &factory,
                           std::span<const holoflow::core::TDesc>  input_descs,
                           std::span<const std::vector<std::byte>> input_host_data,
                           const nlohmann::json                   &jsettings);

// -------------------------------------------------------------------------------------------------
// run_sync_factory_update
// -------------------------------------------------------------------------------------------------
//
// Same as run_sync_factory but exercises the update seam: create, then update with the
// same descriptors/settings, then execute.  Reuse for testing the update/reuse branch.
//
RunResult run_sync_factory_update(const holoflow::core::ISyncTaskFactory &factory,
                                  std::span<const holoflow::core::TDesc>  input_descs,
                                  std::span<const std::vector<std::byte>> input_host_data,
                                  const nlohmann::json                   &jsettings);

} // namespace holonp_test
