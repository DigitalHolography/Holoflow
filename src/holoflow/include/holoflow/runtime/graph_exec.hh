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

#include <boost/graph/adjacency_list.hpp>
#include <cstddef>
#include <cuda_runtime.h>
#include <variant>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holoflow::runtime {

struct SyncMetrics {
  size_t num_execs     = 0;
  double total_time_ms = 0.0;
};

struct SyncRT {
  SyncMetrics      metrics = {};
  core::ISyncTask *task    = nullptr;
  core::SyncCtx    ctx     = {};
};

struct AsyncMetrics {
  size_t num_pushes    = 0;
  size_t num_pops      = 0;
  double total_push_ms = 0.0;
  double total_pop_ms  = 0.0;
};

struct AsyncRT {
  AsyncMetrics       metrics  = {};
  core::IAsyncTask  *task     = nullptr;
  core::AsyncPushCtx push_ctx = {};
  core::AsyncPopCtx  pop_ctx  = {};
};

using NodeRT = std::variant<SyncRT, AsyncRT>;

struct NodeExec {
  core::NodeSpec    spec  = {};
  core::InferResult infer = {};
  NodeRT            rt    = {};
  std::vector<int>  inputs_ids;
  std::vector<int>  outputs_ids;
};

struct PES {
  cudaStream_t stream;
};

} // namespace holoflow::runtime