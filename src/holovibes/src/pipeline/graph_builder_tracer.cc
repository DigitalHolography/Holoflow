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

#include <sstream>

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

std::string GraphBuilderTracer::failure_graph_dot(
    const holoflow::core::GraphSpecDumpPreferences &prefs) const {
  if (!inference_failure_) {
    return {};
  }

  const auto &failure = *inference_failure_;
  const auto &node    = g_[failure.vertex];
  std::string label   = node.name + " (" + node.kind + ")\nInference failed: " + failure.message;
  for (size_t i = 0; i < failure.inputs.size(); ++i) {
    const auto &input = failure.inputs[i];
    const holoflow::core::TDesc contiguous{input.shape, input.dtype, input.mem_loc, input.offset};
    label += "\nInput " + std::to_string(i) + ": shape=" + nlohmann::json(input.shape).dump() +
             ", dtype=" + std::string(holoflow::core::to_string(input.dtype)) +
             ", memory=" + std::string(holoflow::core::to_string(input.mem_loc)) +
             "\n  byte strides=" + nlohmann::json(input.strides).dump() +
             ", contiguous byte strides=" + nlohmann::json(contiguous.strides).dump() +
             ", offset=" + std::to_string(input.offset);
  }

  auto dot = holoflow::core::to_dot(g_, prefs);
  const auto end = dot.rfind('}');
  if (end == std::string::npos) {
    return {};
  }

  std::ostringstream annotation;
  annotation << "  graph [label=\"Graph construction failed (partial graph)\", labelloc=t];\n"
             << "  v" << failure.vertex << " [label=" << nlohmann::json(label).dump()
             << ", style=\"filled,bold\", fillcolor=\"#ffe1e1\", color=\"#b00020\", "
                "penwidth=2];\n";
  dot.insert(end, annotation.str());
  return dot;
}

} // namespace holovibes::pipeline
