// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <thread>
#include <type_traits>

#include "holoflow/runtime/tracing.hh"
#include "support/native_trace.hh"

namespace {
using namespace holoflow::runtime::tracing;
static_assert(!std::is_copy_constructible_v<ScopedTrace>);
static_assert(!std::is_move_constructible_v<ScopedTrace>);

TEST(TracingTest, NativeCaptureIncludesNestedScopesExceptionsAndWorkerThreads) {
  if (!Session::available())
    GTEST_SKIP() << "SDK disabled";
  const auto path    = trace_test::temporary_path("holoflow-native-");
  auto       session = Session::start();
  ASSERT_NE(session, nullptr);
  set_thread_name("Trace Test Main");
  {
    ScopedTrace outer("operation");
    try {
      ScopedTrace inner("quoted \"node\"\n\t\\", "detail");
      throw std::runtime_error("failure");
    } catch (const std::runtime_error &) {
    }
    std::thread worker([] {
      set_thread_name("Trace Test Worker");
      ScopedTrace scope("worker", "scheduler");
    });
    worker.join();
    outer.set_outcome(Outcome::Success);
  }
  session->stop_and_save(path);
  EXPECT_FALSE(Session::active());
  const auto trace = trace_test::read(path);
  ASSERT_EQ(trace.slices.size(), 3);
  EXPECT_EQ(trace.slices[0].name, "operation");
  EXPECT_EQ(trace.slices[0].outcome, "success");
  EXPECT_EQ(trace.slices[1].name, "quoted \"node\"\n\t\\");
  EXPECT_EQ(trace.slices[1].outcome, "failure");
  EXPECT_EQ(trace.slices[0].track, trace.slices[1].track);
  EXPECT_NE(trace.slices[0].track, trace.slices[2].track);
  EXPECT_EQ(trace.threads.at(trace.slices[2].track), "Trace Test Worker");
  for (const auto &slice : trace.slices)
    EXPECT_TRUE(slice.complete);
}

TEST(TracingTest, AutomaticCaptureJoinsExplicitSessionAndDoesNotExportIt) {
  if (!Session::available())
    GTEST_SKIP() << "SDK disabled";
  const auto path      = trace_test::temporary_path("holoflow-explicit-");
  const auto automatic = path.string() + ".automatic";
  auto       session   = Session::start();
  ASSERT_NE(session, nullptr);
  EXPECT_EQ(Session::start(), nullptr);
  {
    Capture     joined(automatic);
    ScopedTrace scope("joined");
  }
  EXPECT_TRUE(Session::active());
  EXPECT_FALSE(std::filesystem::exists(automatic));
  session->stop_and_save(path);
  const auto trace = trace_test::read(path);
  ASSERT_EQ(trace.slices.size(), 1);
  EXPECT_EQ(trace.slices[0].name, "joined");
  EXPECT_TRUE(trace.slices[0].complete);
}

TEST(TracingTest, SequentialSessionsAreIsolatedAndCategoryFilteringWorks) {
  if (!Session::available())
    GTEST_SKIP() << "SDK disabled";
  const auto path = trace_test::temporary_path("holoflow-sequential-");
  {
    Capture     capture(path);
    ScopedTrace scope("first");
  }
  {
    auto session = Session::start({.include_details = false});
    ASSERT_NE(session, nullptr);
    {
      ScopedTrace detail("filtered", "detail");
    }
    {
      ScopedTrace scope("second");
    }
    session->stop_and_save(path);
  }
  const auto trace = trace_test::read(path);
  ASSERT_EQ(trace.slices.size(), 1);
  EXPECT_EQ(trace.slices[0].name, "second");
  EXPECT_TRUE(trace.slices[0].complete);
}

TEST(TracingTest, OldScopeDoesNotCloseASliceInANewSession) {
  if (!Session::available())
    GTEST_SKIP() << "SDK disabled";
  const auto                 path  = trace_test::temporary_path("holoflow-generation-");
  auto                       first = Session::start();
  std::optional<ScopedTrace> old;
  old.emplace("old");
  first.reset();
  auto second = Session::start();
  ASSERT_NE(second, nullptr);
  old.reset();
  {
    ScopedTrace scope("new");
  }
  second->stop_and_save(path);
  const auto trace = trace_test::read(path);
  ASSERT_EQ(trace.slices.size(), 1);
  EXPECT_EQ(trace.slices[0].name, "new");
  EXPECT_TRUE(trace.slices[0].complete);
}

TEST(TracingTest, ExportFailureReleasesSessionAndAutomaticCaptureIsBestEffort) {
  if (!Session::available()) {
    EXPECT_EQ(Session::start(), nullptr);
    return;
  }
  const auto directory = trace_test::temporary_path("holoflow-export-failure-");
  std::filesystem::create_directory(directory);
  auto session = Session::start();
  ASSERT_NE(session, nullptr);
  {
    ScopedTrace scope("event");
  }
  EXPECT_THROW(session->stop_and_save(directory), std::ios_base::failure);
  EXPECT_FALSE(Session::active());
  EXPECT_NO_THROW({
    Capture     capture(directory);
    ScopedTrace scope("automatic");
  });
  EXPECT_FALSE(Session::active());
  EXPECT_THROW((void)Session::start({.buffer_size_kb = 0}), std::invalid_argument);
  EXPECT_THROW((void)Session::start({.buffer_size_kb = (std::numeric_limits<uint32_t>::max)()}),
               std::invalid_argument);
  auto next = Session::start();
  EXPECT_NE(next, nullptr);
}
} // namespace
