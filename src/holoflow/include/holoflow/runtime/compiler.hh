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

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

// -------------------------------------------------------------------------------------------------
// Compiler output
// -------------------------------------------------------------------------------------------------

struct CompilerOutput {
  GraphPlan            graph;
  std::vector<Section> sections;
  ExecResouces         resources;
};

// -------------------------------------------------------------------------------------------------
// Compiler
// -------------------------------------------------------------------------------------------------

class Compiler {
public:
  struct Config {
    std::filesystem::path log_dir;
    bool                  dump_dot_on_failure = true;
    bool                  verbose_tracing     = true;
    // Profiling toggles
    bool        enable_profiling = true;
    std::string trace_filename   = "trace_events.json";
  };

  explicit Compiler(core::Registry &registry, Config config = {});
  ~Compiler();

  [[nodiscard]] std::unique_ptr<CompilerOutput>
  compile(const core::GraphSpec &gspec, std::unique_ptr<CompilerOutput> prev = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holoflow::runtime