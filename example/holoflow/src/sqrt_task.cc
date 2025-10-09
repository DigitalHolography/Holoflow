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

#include "holoflow/examples/sqrt_task.hh"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace holoflow_examples {

holoflow::core::OpResult SqrtTask::execute(holoflow::core::SyncCtx &ctx) {
  const auto  &input  = ctx.inputs[0];
  auto        &output = ctx.outputs[0];
  const float *idata  = reinterpret_cast<const float *>(input.data);
  float       *odata  = reinterpret_cast<float *>(output.data);
  const size_t nelems = input.desc.num_elements();

  for (size_t i = 0; i < nelems; ++i) {
    odata[i] = std::sqrt(idata[i]);
  }

  return holoflow::core::OpResult::Ok;
}

// holoflow::core::InferResult
// SqrtTaskFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
//                        const nlohmann::json                  &jsettings) const {}

// std::unique_ptr<holoflow::core::ISyncTask>
// SqrtTaskFactory::create(std::span<const holoflow::core::TDesc> input_descs,
//                         const nlohmann::json                  &jsettings,
//                         const holoflow::core::SyncCreateCtx   &ctx) const {
//   return std::make_unique<SqrtTask>();
// }

} // namespace holoflow_examples
