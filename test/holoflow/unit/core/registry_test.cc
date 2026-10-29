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
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

#include "support/math_tasks.hh"

using TDesc  = holoflow::core::TDesc;
using DType  = holoflow::core::DType;
using MemLoc = holoflow::core::MemLoc;

namespace {

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

} // namespace
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