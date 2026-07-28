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

#include <boost/graph/adjacency_list.hpp>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::GraphSpec;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::TDesc;

class NoopTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    return holoflow::core::OpResult::Ok;
  }
};

class NoopFactory final : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const TDesc> input_descs,
                                    const nlohmann::json &) const override {
    return {
        .input_descs   = {input_descs.begin(), input_descs.end()},
        .output_descs  = {},
        .in_place      = {},
        .owned_inputs  = std::vector<bool>(input_descs.size(), false),
        .owned_outputs = {},
        .kind          = holoflow::core::TaskKind::Sync,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

GraphSpec sample_graph() {
  GraphSpec  graph;
  const auto source =
      add_vertex(NodeSpec{.name = "source", .kind = "source", .settings = {{"frames", 4}}}, graph);
  const auto sink = add_vertex(
      NodeSpec{
          .name = "sink", .kind = "sink", .settings = nlohmann::json::object(), .debug = false},
      graph);
  add_edge(source, sink, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, graph);
  return graph;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Tensor descriptors
// -------------------------------------------------------------------------------------------------

TEST(TensorDescriptorTest, CreatesContiguousStridesAndSizes) {
  const TDesc desc({2, 3, 4}, DType::F32, MemLoc::Host);

  EXPECT_EQ(desc.rank(), 3);
  EXPECT_EQ(desc.strides, (std::vector<size_t>{48, 16, 4}));
  EXPECT_EQ(desc.num_elements(), 24);
  EXPECT_EQ(desc.num_bytes(), 96);
}

TEST(TensorDescriptorTest, HandlesEmptyAndZeroElementShapes) {
  EXPECT_EQ(TDesc({}, DType::U8, MemLoc::Host).num_bytes(), 0);
  EXPECT_EQ(TDesc({3, 0, 2}, DType::U16, MemLoc::Host).num_elements(), 0);
}

TEST(TensorDescriptorTest, DetectsElementCountOverflow) {
  const TDesc desc({std::numeric_limits<size_t>::max(), 2}, DType::U8, MemLoc::Host);
  EXPECT_THROW((void)desc.num_elements(), std::overflow_error);
}

TEST(TensorDescriptorTest, SerializesDtypeMemoryAndStrides) {
  const TDesc original({2, 3}, DType::CF32, MemLoc::Device, std::vector<size_t>{32, 8});
  const auto  encoded = nlohmann::json(original);
  const auto  decoded = encoded.get<TDesc>();

  EXPECT_EQ(decoded.shape, original.shape);
  EXPECT_EQ(decoded.strides, original.strides);
  EXPECT_EQ(decoded.dtype, original.dtype);
  EXPECT_EQ(decoded.mem_loc, original.mem_loc);
  EXPECT_THROW((void)nlohmann::json("bad").get<DType>(), std::invalid_argument);
  EXPECT_THROW((void)nlohmann::json("bad").get<MemLoc>(), std::invalid_argument);
}

// -------------------------------------------------------------------------------------------------
// Registry
// -------------------------------------------------------------------------------------------------

TEST(RegistryTest, RegistersAndLooksUpSyncFactory) {
  holoflow::core::Registry registry;
  registry.register_sync("noop", std::make_unique<NoopFactory>());

  EXPECT_TRUE(registry.is_registered("noop"));
  EXPECT_TRUE(registry.is_sync_registered("noop"));
  EXPECT_FALSE(registry.is_async_registered("noop"));
  EXPECT_EQ(&registry.get("noop"), &registry.get_sync("noop"));
}

TEST(RegistryTest, RejectsDuplicateAndMissingFactories) {
  holoflow::core::Registry registry;
  registry.register_sync("noop", std::make_unique<NoopFactory>());

  EXPECT_THROW(registry.register_sync("noop", std::make_unique<NoopFactory>()),
               std::invalid_argument);
  EXPECT_THROW((void)registry.get("missing"), std::out_of_range);
  EXPECT_THROW((void)registry.get_async("missing"), std::out_of_range);
}

// -------------------------------------------------------------------------------------------------
// Graph serialization
// -------------------------------------------------------------------------------------------------

TEST(GraphSpecTest, RoundTripsNodesEdgesSettingsAndDebugFlag) {
  const auto encoded   = holoflow::core::to_json(sample_graph());
  const auto decoded   = holoflow::core::from_json(encoded);
  const auto roundtrip = holoflow::core::to_json(decoded);

  EXPECT_EQ(roundtrip, encoded);
  EXPECT_EQ(num_vertices(decoded), 2);
  EXPECT_EQ(num_edges(decoded), 1);
}

TEST(GraphSpecTest, ProducesDeterministicVertexOrdering) {
  const auto decoded = holoflow::core::from_json({
      {"nodes",
       {
           {"z", {{"type", "sink"}, {"params", nlohmann::json::object()}}},
           {"a", {{"type", "source"}, {"params", nlohmann::json::object()}}},
       }},
      {"edges", nlohmann::json::array()},
  });

  EXPECT_EQ(decoded[0].name, "a");
  EXPECT_EQ(decoded[1].name, "z");
}

TEST(GraphSpecTest, RejectsMalformedDocumentsAndUnknownNodes) {
  EXPECT_THROW((void)holoflow::core::from_json(nlohmann::json::array()), std::runtime_error);
  EXPECT_THROW((void)holoflow::core::from_json({{"nodes", nlohmann::json::array()}}),
               std::runtime_error);
  EXPECT_THROW((void)holoflow::core::from_json({
                   {"nodes", {{"a", {{"type", "source"}, {"params", nlohmann::json::object()}}}}},
                   {"edges", {{{"from", "a"}, {"to", "missing"}, {"out", 0}, {"in", 0}}}},
               }),
               std::runtime_error);
}

TEST(GraphSpecTest, DotOutputContainsEscapedLabelsAndEdgePorts) {
  auto graph        = sample_graph();
  graph[0].settings = {{"label", "line\n\"quoted\""}};

  const auto dot = holoflow::core::to_dot(graph);

  EXPECT_NE(dot.find("source"), std::string::npos);
  EXPECT_NE(dot.find("quoted"), std::string::npos);
  EXPECT_NE(dot.find("taillabel=\"0\" headlabel=\"0\""), std::string::npos);
}
