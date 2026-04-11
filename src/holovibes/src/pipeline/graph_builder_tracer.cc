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

#include "graph_builder_tracer.hh"

namespace holovibes::pipeline {

GraphBuilderTracer::GraphBuilderTracer(holoflow::core::Registry &registry) : reg_(registry) {}

holoflow::core::TDesc GraphBuilderTracer::TDesc::as_core() const {
  holoflow::core::TDesc t{};
  t.shape   = shape;
  t.dtype   = dtype;
  t.mem_loc = mem_loc;
  t.strides = strides;
  t.offset  = offset;
  return t;
}

GraphBuilderTracer::TDesc GraphBuilderTracer::TDesc::from_core(const holoflow::core::TDesc &base) {
  TDesc t;
  static_cast<holoflow::core::TDesc &>(t) = base;
  return t;
}

std::vector<holoflow::core::TDesc> GraphBuilderTracer::to_core_descs(std::span<const TDesc> src) {
  std::vector<holoflow::core::TDesc> out;
  out.reserve(src.size());
  for (const auto &t : src) {
    out.push_back(t.as_core());
  }
  return out;
}

} // namespace holovibes::pipeline
