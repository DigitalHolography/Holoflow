// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <utility>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "support/math_tasks.hh"

namespace {

class ExposedTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    return holoflow::core::OpResult::Ok;
  }

  spdlog::logger                  *bound_logger() { return logger(); }
  holoflow::core::IOStorageAccess &bound_storage() { return storage_access(); }
};

class StorageAccess final : public holoflow::core::IOStorageAccess {
public:
  holoflow::core::Storage &owned_input_storage(size_t) override { return storage; }
  holoflow::core::Storage &owned_output_storage(size_t) override { return storage; }

  holoflow::core::Storage storage{holoflow::core::MemLoc::Host, 0, nullptr};
};

} // namespace

TEST(TensorDescriptorTest, CoversEveryEnumAndConstructor) {
  using holoflow::core::DType;
  using holoflow::core::MemLoc;

  EXPECT_EQ(holoflow::core::size_of(DType::U8), 1);
  EXPECT_EQ(holoflow::core::size_of(DType::U16), 2);
  EXPECT_EQ(holoflow::core::size_of(DType::F32), 4);
  EXPECT_EQ(holoflow::core::size_of(DType::CF32), 8);
  EXPECT_EQ(holoflow::core::to_string(DType::U8), "U8");
  EXPECT_EQ(holoflow::core::to_string(DType::U16), "U16");
  EXPECT_EQ(holoflow::core::to_string(DType::F32), "F32");
  EXPECT_EQ(holoflow::core::to_string(DType::CF32), "CF32");
  EXPECT_EQ(holoflow::core::to_string(MemLoc::Host), "Host");
  EXPECT_EQ(holoflow::core::to_string(MemLoc::Device), "Device");

  for (const auto dtype : {DType::U8, DType::U16, DType::F32, DType::CF32}) {
    EXPECT_EQ(nlohmann::json(dtype).get<DType>(), dtype);
  }
  for (const auto location : {MemLoc::Host, MemLoc::Device}) {
    EXPECT_EQ(nlohmann::json(location).get<MemLoc>(), location);
  }

  const holoflow::core::TDesc offset_desc({2, 3}, DType::U16, MemLoc::Host, 7);
  EXPECT_EQ(offset_desc.strides, (std::vector<size_t>{6, 2}));
  EXPECT_EQ(offset_desc.offset, 7);
  const holoflow::core::TDesc custom_desc({2, 3}, DType::U16, MemLoc::Host,
                                          std::vector<size_t>{16, 4}, 9);
  EXPECT_EQ(custom_desc.strides, (std::vector<size_t>{16, 4}));
  EXPECT_EQ(custom_desc.offset, 9);
  EXPECT_EQ(custom_desc.num_bytes(), 32);
}

TEST(TensorTest, AllocatesHostStorageAndExposesOffsetView) {
  holoflow::core::Tensor tensor(
      holoflow::core::TDesc({4}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host));
  ASSERT_NE(tensor.data(), nullptr);
  EXPECT_EQ(std::as_const(tensor).data(), tensor.data());
  EXPECT_EQ(tensor.desc().num_bytes(), 16);
  auto view = tensor.view();
  EXPECT_FALSE(view.is_nullptr());
  EXPECT_EQ(view.data(), tensor.data());

  std::array<std::byte, 16> bytes{};
  holoflow::core::Storage   storage{holoflow::core::MemLoc::Host, bytes.size(), bytes.data()};
  holoflow::core::TView     offset_view{
      holoflow::core::TDesc({2}, holoflow::core::DType::U8, holoflow::core::MemLoc::Host, 3),
      &storage};
  EXPECT_EQ(offset_view.data(), bytes.data() + 3);
  EXPECT_TRUE(holoflow::core::TView{}.is_nullptr());
}

TEST(RegistryTest, SupportsAsyncFactoriesAndRejectsCrossKindDuplicates) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_async("async", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));

  EXPECT_TRUE(registry.is_async_registered("async"));
  EXPECT_TRUE(registry.is_registered("async"));
  EXPECT_FALSE(registry.is_sync_registered("async"));
  EXPECT_EQ(&registry.get("async"), &registry.get_async("async"));
  EXPECT_THROW(
      registry.register_sync("async", std::make_unique<holoflow::test::ScaleFactory>(1.F, state)),
      std::invalid_argument);
  EXPECT_THROW(registry.register_sync("null", nullptr), std::invalid_argument);
  EXPECT_THROW(registry.register_async("null", nullptr), std::invalid_argument);
  EXPECT_THROW(
      registry.register_async("async", std::make_unique<holoflow::test::AsyncBridgeFactory>(state)),
      std::invalid_argument);
  EXPECT_THROW((void)registry.get_sync("missing"), std::out_of_range);
}

TEST(TaskTest, DefaultOwnershipHooksThrowAndServicesCanBeBound) {
  ExposedTask task;
  EXPECT_THROW((void)task.acquire_input(0), std::out_of_range);
  EXPECT_THROW(task.release_output(0), std::out_of_range);

  auto          sink   = std::make_shared<spdlog::sinks::null_sink_mt>();
  auto          logger = std::make_shared<spdlog::logger>("holoflow-test-task", sink);
  StorageAccess storage;
  task.bind_logger(logger);
  task.bind_storage_access(&storage);

  EXPECT_EQ(task.bound_logger(), logger.get());
  EXPECT_EQ(&task.bound_storage(), &storage);
}

TEST(TaskFactoryTest, DefaultUpdatesRecreateSyncAndAsyncTasks) {
  auto                               state = std::make_shared<holoflow::test::MathState>();
  holoflow::test::ScaleFactory       sync_factory(2.F, state);
  holoflow::test::AsyncBridgeFactory async_factory(state);

  EXPECT_NE(sync_factory.update(nullptr, {}, nlohmann::json::object(), {}).get(), nullptr);
  EXPECT_NE(async_factory.update(nullptr, {}, nlohmann::json::object(), {}).get(), nullptr);
}

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
