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

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"

namespace holonp_test {

// -------------------------------------------------------------------------------------------------
// Python oracle bridge
// -------------------------------------------------------------------------------------------------
//
// Interchange format:
//   JSON manifest  — op name, per-tensor descriptors (shape/dtype/strides/offset), payload names
//   Binary payloads — dense logical bytes (num_bytes() each) starting at the logical origin
//
// The oracle script writes output payloads as dense C-contiguous arrays.
// Throws std::runtime_error if the Python process exits non-zero or outputs are missing.
//

struct OracleInput {
  std::string                         op;
  size_t                              n_outputs = 1; ///< How many output files to collect
  std::vector<holoflow::core::TDesc>  input_descs;
  std::vector<std::vector<std::byte>> input_bytes; ///< Dense logical bytes per input tensor
  nlohmann::json                      settings =
      nlohmann::json::object(); ///< Op-specific settings passed verbatim to the oracle
};

struct OracleOutput {
  std::vector<std::vector<std::byte>> output_bytes; ///< Dense logical bytes per output tensor
};

// Invoke the Python oracle script at `oracle_script` with the given inputs.
[[nodiscard]] OracleOutput invoke_oracle(const OracleInput           &input,
                                         const std::filesystem::path &oracle_script);

} // namespace holonp_test
