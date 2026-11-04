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

#include <array>
#include <cstddef>
#include <memory>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <utility>

#include "holoflow/core/tasks.hh"

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
