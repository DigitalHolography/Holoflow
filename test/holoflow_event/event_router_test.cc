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

#include <nlohmann/json.hpp>

#include "holoflow_event/router.hh"

TEST(EventRouterTest, RoutesUiEventsToBoundNode) {
  holoflow_event::Router router;
  auto                   handles = router.bind_node("node");

  ASSERT_TRUE(router.ui_try_send("node", {{"command", "start"}}));
  router.tick();
  const auto event = handles.in.try_pop();

  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->node_id, "node");
  EXPECT_EQ(event->data.at("command"), "start");
  EXPECT_EQ(event->direction, holoflow_event::EventDirection::ToNode);
}

TEST(EventRouterTest, RoutesNodeEventsToUi) {
  holoflow_event::Router router;
  auto                   handles = router.bind_node("node");
  holoflow_event::Event  event{
      .direction = holoflow_event::EventDirection::ToUi,
      .node_id   = "node",
      .data      = {{"state", "ready"}},
  };

  ASSERT_TRUE(handles.out.try_push(std::move(event)));
  router.tick();
  const auto received = router.ui_try_receive();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->data.at("state"), "ready");
}

TEST(EventRouterTest, HonorsTickBudget) {
  holoflow_event::Router router;
  auto                   handles = router.bind_node("node");

  ASSERT_TRUE(router.ui_try_send("node", 1));
  ASSERT_TRUE(router.ui_try_send("node", 2));
  router.tick(1);

  ASSERT_TRUE(handles.in.try_pop().has_value());
  EXPECT_FALSE(handles.in.try_pop().has_value());
  router.tick(1);
  EXPECT_TRUE(handles.in.try_pop().has_value());
}

TEST(EventRouterTest, DropsUnknownAndFullMailboxEventsWithCounters) {
  holoflow_event::Router::Config config;
  config.router_to_node_capacity = 1;
  holoflow_event::Router router(config);
  auto                   handles = router.bind_node("node");

  ASSERT_TRUE(router.ui_try_send("missing", nullptr));
  router.tick();
  EXPECT_EQ(router.ui_to_router_counters().dropped_events, 1);

  ASSERT_TRUE(router.ui_try_send("node", 1));
  ASSERT_TRUE(router.ui_try_send("node", 2));
  router.tick();
  EXPECT_EQ(handles.in.counters().sent_events, 1);
  EXPECT_EQ(handles.in.counters().dropped_events, 1);
}

TEST(EventRouterTest, DropAllUpdatesReaderCounters) {
  holoflow_event::Router router;
  auto                   handles = router.bind_node("node");

  ASSERT_TRUE(router.ui_try_send("node", 1));
  ASSERT_TRUE(router.ui_try_send("node", 2));
  router.tick();

  EXPECT_EQ(handles.in.drop_all(), 2);
  EXPECT_EQ(handles.in.counters().dropped_events, 2);
}
