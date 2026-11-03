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

#include "holoflow/runtime/graph_display.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::GraphSpec;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::TDesc;

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

// -------------------------------------------------------------------------------------------------
// GraphSpec
// -------------------------------------------------------------------------------------------------

TEST(GraphSpecTest, AppliesDefaultsAndAcceptsPrimitiveSettings) {
  const auto graph = holoflow::core::from_json({
      {"nodes",
       {
           {"a", {{"type", "source"}, {"params", nlohmann::json::object()}}},
           {"b", {{"type", "sink"}, {"params", 42}, {"debug", false}}},
       }},
  });

  ASSERT_EQ(num_vertices(graph), 2);
  EXPECT_TRUE(graph[0].settings.is_object());
  EXPECT_TRUE(graph[0].debug);
  EXPECT_EQ(graph[1].settings, 42);
  EXPECT_FALSE(graph[1].debug);
  EXPECT_TRUE(holoflow::core::to_json(graph).at("edges").empty());
}

TEST(GraphSpecTest, RejectsInvalidNodeAndEdgeFields) {
  const std::vector<nlohmann::json> invalid_documents{
      {{"nodes", nullptr}},
      {{"nodes", {{"", {{"type", "x"}, {"params", {}}}}}}},
      {{"nodes", {{"a", 1}}}},
      {{"nodes", {{"a", {{"params", {}}}}}}},
      {{"nodes", {{"a", {{"type", 1}, {"params", {}}}}}}},
      {{"nodes", {{"a", {{"type", "x"}}}}}},
      {{"nodes", {{"a", {{"type", "x"}, {"params", {}}, {"debug", 1}}}}}},
      {{"nodes", nlohmann::json::object()}, {"edges", nlohmann::json::object()}},
      {{"nodes", {{"a", {{"type", "x"}, {"params", {}}}}}}, {"edges", {1}}},
      {{"nodes", {{"a", {{"type", "x"}, {"params", {}}}}}},
       {"edges", {{{"to", "a"}, {"out", 0}, {"in", 0}}}}},
      {{"nodes", {{"a", {{"type", "x"}, {"params", {}}}}}},
       {"edges", {{{"from", "a"}, {"to", "a"}, {"out", "bad"}, {"in", 0}}}}},
  };
  for (const auto &document : invalid_documents) {
    EXPECT_THROW((void)holoflow::core::from_json(document), std::runtime_error) << document.dump();
  }
}

TEST(GraphSpecTest, RejectsUnnamedNodesWhenSerializing) {
  holoflow::core::GraphSpec graph;
  add_vertex(holoflow::core::NodeSpec{"", "source", {}}, graph);
  EXPECT_THROW((void)holoflow::core::to_json(graph), std::runtime_error);
}

TEST(GraphSpecTest, DotHandlesUnnamedNodesKindsAndCarriageReturns) {
  holoflow::core::GraphSpec graph;
  add_vertex(holoflow::core::NodeSpec{"", "", "line\r\nvalue"}, graph);
  const auto dot = holoflow::core::to_dot(graph);
  EXPECT_NE(dot.find("(unnamed)"), std::string::npos);
  EXPECT_NE(dot.find("line"), std::string::npos);
  EXPECT_NE(dot.find("value"), std::string::npos);
  EXPECT_EQ(dot.find('\r'), std::string::npos);
}

TEST(GraphSpecTest, NullParamsAreNormalizedToAnObject) {
  const auto graph = holoflow::core::from_json({
      {"nodes", {{"a", {{"type", "source"}, {"params", nullptr}}}}},
  });
  ASSERT_EQ(num_vertices(graph), 1);
  EXPECT_TRUE(graph[0].settings.is_object());
}
