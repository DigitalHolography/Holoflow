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

#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"

using holoflow::core::Registry;

namespace {

// Minimal test doubles. Extend if your interfaces have pure virtuals.
struct DummySyncFactory final : public holoflow::core::ISyncTaskFactory {
  ~DummySyncFactory() override = default;

  virtual holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                            const nlohmann::json &) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    throw std::runtime_error("Not implemented");
  }
};

struct DummyAsyncFactory final : public holoflow::core::IAsyncTaskFactory {
  ~DummyAsyncFactory() override = default;

  virtual holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                            const nlohmann::json &) const override {
    throw std::runtime_error("Not implemented");
  }

  virtual std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann ::json &,
         const holoflow::core::AsyncCreateCtx &) const override {
    throw std::runtime_error("Not implemented");
  }
};

} // namespace

TEST(RegistryTest, RegisterAndGetSync) {
  Registry reg;
  auto     f   = std::make_unique<DummySyncFactory>();
  auto    *raw = f.get();

  reg.register_sync("add", std::move(f));
  const auto &got = reg.get_sync("add");

  // Same instance back.
  EXPECT_EQ(&got, static_cast<const holoflow::core::ISyncTaskFactory *>(raw));
}

TEST(RegistryTest, RegisterAndGetAsync) {
  Registry reg;
  auto     f   = std::make_unique<DummyAsyncFactory>();
  auto    *raw = f.get();

  reg.register_async("queue", std::move(f));
  const auto &got = reg.get_async("queue");

  EXPECT_EQ(&got, static_cast<const holoflow::core::IAsyncTaskFactory *>(raw));
}

TEST(RegistryTest, DuplicateSyncKeyThrows) {
  Registry reg;
  reg.register_sync("blur", std::make_unique<DummySyncFactory>());

  EXPECT_THROW(reg.register_sync("blur", std::make_unique<DummySyncFactory>()),
               std::invalid_argument);
}

TEST(RegistryTest, DuplicateAsyncKeyThrows) {
  Registry reg;
  reg.register_async("source", std::make_unique<DummyAsyncFactory>());

  EXPECT_THROW(reg.register_async("source", std::make_unique<DummyAsyncFactory>()),
               std::invalid_argument);
}

TEST(RegistryTest, GetMissingSyncThrows) {
  Registry reg;
  EXPECT_THROW((void)reg.get_sync("missing"), std::out_of_range);
}

TEST(RegistryTest, GetMissingAsyncThrows) {
  Registry reg;
  EXPECT_THROW((void)reg.get_async("missing"), std::out_of_range);
}

TEST(RegistryTest, CrossKindDuplicateThrows) {
  Registry reg;
  reg.register_sync("x", std::make_unique<DummySyncFactory>());
  EXPECT_THROW(reg.register_async("x", std::make_unique<DummyAsyncFactory>()),
               std::invalid_argument);
}

TEST(RegistryTest, MultipleDistinctKeysWork) {
  Registry reg;
  auto    *s1 = (reg.register_sync("a", std::make_unique<DummySyncFactory>()), &reg.get_sync("a"));
  auto    *s2 = (reg.register_sync("b", std::make_unique<DummySyncFactory>()), &reg.get_sync("b"));
  EXPECT_NE(s1, s2);

  auto *a1 = (reg.register_async("c", std::make_unique<DummyAsyncFactory>()), &reg.get_async("c"));
  auto *a2 = (reg.register_async("d", std::make_unique<DummyAsyncFactory>()), &reg.get_async("d"));
  EXPECT_NE(a1, a2);
}