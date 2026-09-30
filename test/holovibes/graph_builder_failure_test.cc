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

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>

#include "holotask/asyncs/batch_queue.hh"
#include "pipeline/graph_builder_tracer.hh"

namespace holovibes::pipeline {
namespace {

class TensorFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit TensorFactory(holoflow::core::TDesc output) : output_(std::move(output)) {}

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                   const nlohmann::json &) const override {
    using namespace holoflow::core;
    return {.input_descs   = std::vector<TDesc>(inputs.begin(), inputs.end()),
            .output_descs  = {output_},
            .in_place      = {},
            .owned_inputs  = std::vector<bool>(inputs.size(), false),
            .owned_outputs = {false},
            .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return nullptr;
  }

private:
  holoflow::core::TDesc output_;
};

class TestGraphBuilder : public GraphBuilderTracer {
public:
  using GraphBuilderTracer::GraphBuilderTracer;
  using GraphBuilderTracer::failure_graph_dot;

  void build() {
    const auto source = make_source_sync_node("coefficients", "Coefficients", "Coefficients",
                                              nlohmann::json::object())
                            .at(0);
    const auto reshaped =
        make_unary_sync_node("reshape", "Reshape", "Reshape", source, nlohmann::json::object())
            .at(0);
    (void)make_unary_async_node("batch_queue", "BatchQueue", "BatchQueue", reshaped,
                                holotask::asyncs::BatchQueueSettings{2, 1, 1});
  }
};

TEST(GraphBuilderFailureTest, ShowsRejectedTensorAtTheFailedNode) {
  holoflow::core::Registry registry;
  using holoflow::core::DType;
  using holoflow::core::MemLoc;
  using holoflow::core::TDesc;
  registry.register_sync("Coefficients",
                         std::make_unique<TensorFactory>(TDesc({1}, DType::F32, MemLoc::Host)));
  registry.register_sync("Reshape", std::make_unique<TensorFactory>(
                                        TDesc({1, 1}, DType::F32, MemLoc::Host, {4, 1})));
  registry.register_async("BatchQueue", std::make_unique<holotask::asyncs::BatchQueueFactory>());

  TestGraphBuilder builder(registry);
  try {
    builder.build();
    FAIL() << "BatchQueue should reject the malformed tensor";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("must be contiguous"), std::string::npos);
  }

  const auto dot = builder.failure_graph_dot({});
  EXPECT_NE(dot.find("Graph construction failed (partial graph)"), std::string::npos);
  EXPECT_NE(dot.find("v0 -> v1"), std::string::npos);
  EXPECT_NE(dot.find("v1 -> v2"), std::string::npos);
  EXPECT_NE(dot.find("batch_queue_2 (BatchQueue)"), std::string::npos);
  EXPECT_NE(dot.find("shape=[1,1], dtype=F32"), std::string::npos);
  EXPECT_NE(dot.find("byte strides=[4,1], contiguous byte strides=[4,4]"), std::string::npos);
  EXPECT_NE(dot.find("fillcolor=\"#ffe1e1\""), std::string::npos);
}

} // namespace
} // namespace holovibes::pipeline
