// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"

namespace holonp_test {

// -------------------------------------------------------------------------------------------------
// In-process C++ reference operations
// -------------------------------------------------------------------------------------------------

struct ReferenceInput {
  std::string                        op;
  size_t                             n_outputs = 1;
  std::vector<holoflow::core::TDesc> input_descs;
  std::vector<std::vector<std::byte>> input_bytes;
  nlohmann::json                     settings = nlohmann::json::object();
};

struct ReferenceOutput {
  std::vector<std::vector<std::byte>> output_bytes;
};

// Computes expected outputs in-process on the host CPU. Input payloads may be dense logical
// payloads or backing stores described by strides and offsets. Outputs are dense C-contiguous
// payloads, matching the previous Python reference contract.
[[nodiscard]] ReferenceOutput invoke_reference(const ReferenceInput &input);

} // namespace holonp_test
