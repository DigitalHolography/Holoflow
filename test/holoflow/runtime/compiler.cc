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

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

#include "factory_maker.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"


TEST(CompilerTest, EmptyGraph) {
  holoflow::core::Registry    registry;
  holoflow::core::GraphSpec   gspec;
  holoflow::runtime::Compiler compiler(registry);
  EXPECT_THROW(compiler.compile(gspec), std::logic_error);
}

TEST(DummyFactory, SyncOk) {
  holoflow::test::DummyTaskFactory<holoflow::core::ISyncTaskFactory> fac(
      {.num_inputs = 1, .num_outputs = 1});
  auto ir = fac.infer({}, {});
  EXPECT_EQ(ir.kind, holoflow::core::TaskKind::Sync);

  holoflow::core::SyncCreateCtx ctx;
  auto                          task = fac.create({}, {}, ctx);
  holoflow::core::SyncCtx       run_ctx{{}, {}, nullptr};
  EXPECT_EQ(task->execute(run_ctx), holoflow::core::OpResult::Ok);
}

TEST(CompilerTestValid, SingleNode) {
  holoflow::core::Registry                                           registry;
  holoflow::test::DummyTaskFactory<holoflow::core::ISyncTaskFactory> fac({0, 0});

  registry.register_sync("noop", std::unique_ptr<holoflow::core::ISyncTaskFactory>(&fac));
  holoflow::core::GraphSpec gspec;
  holoflow::core::NodeSpec  node;
  node.name     = "node1";
  node.kind     = "noop";
  node.settings = nlohmann::json::object();
  boost::add_vertex(node, gspec);
  holoflow::runtime::Compiler compiler(registry);

  EXPECT_NO_THROW({
    try {
      auto out = compiler.compile(gspec);
    } catch (const std::logic_error &e) {
      std::cout << e.what() << std::endl;
      throw;
    }
  });
}

auto make_node = [](const std::string &name, const std::string &kind) {
  holoflow::core::NodeSpec node;
  node.name     = name;
  node.kind     = kind;
  node.settings = nlohmann::json::object();
  return node;
};

TEST(CompilerTestValid, OneInputOneOutput) {
  holoflow::core::Registry                                           registry;
  holoflow::test::DummyTaskFactory<holoflow::core::ISyncTaskFactory> source({0, 1});
  holoflow::test::DummyTaskFactory<holoflow::core::ISyncTaskFactory> sink({1, 0});

  registry.register_sync("source", std::unique_ptr<holoflow::core::ISyncTaskFactory>(&source));
  registry.register_sync("sink", std::unique_ptr<holoflow::core::ISyncTaskFactory>(&sink));

  holoflow::core::GraphSpec gspec;

  auto node1 = make_node("node1", "source");
  auto node2 = make_node("node2", "sink");
  auto v1 = boost::add_vertex(node1, gspec);
  auto v2 = boost::add_vertex(node2, gspec);
  boost::add_edge(v1, v2, {0, 0}, gspec);
  holoflow::runtime::Compiler compiler(registry);

  EXPECT_NO_THROW({
    try {
      auto out = compiler.compile(gspec);
    } catch (const std::logic_error &e) {
      std::cout << e.what() << std::endl;
      throw;
    }
  });
}