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

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "holoflow_event/bounded_mpmc.hh"

namespace holoflow_event {

using NodeId     = std::string;
using time_point = std::chrono::steady_clock::time_point;

enum class EventDirection {
  ToNode,
  ToUi,
};

struct Event {
  EventDirection direction;
  NodeId         node_id;
  nlohmann::json data;
  time_point     ts;
};

struct MailboxCounters {
  std::atomic<uint64_t> received_events{0};
  std::atomic<uint64_t> dropped_events{0};
  std::atomic<uint64_t> sent_events{0};

  struct Snapshot {
    uint64_t received_events;
    uint64_t dropped_events;
    uint64_t sent_events;
  };

  [[nodiscard]] Snapshot snapshot() const noexcept;
};

class EventReader {
public:
  EventReader(BoundedMPMC<Event> *queue, MailboxCounters *counters);

  [[nodiscard]] std::optional<Event> try_pop() noexcept;

  [[nodiscard]] size_t drop_all() noexcept;

  [[nodiscard]] MailboxCounters::Snapshot counters() const noexcept;

private:
  BoundedMPMC<Event> *queue_;
  MailboxCounters    *counters_;
};

class EventWriter {
public:
  EventWriter(BoundedMPMC<Event> *queue, MailboxCounters *counters);

  [[nodiscard]] bool try_push(Event &&event) noexcept;

  [[nodiscard]] MailboxCounters::Snapshot counters() const noexcept;

private:
  BoundedMPMC<Event> *queue_;
  MailboxCounters    *counters_;
};

class Router {
public:
  struct Config {
    size_t ui_to_router_capacity    = 1024;
    size_t router_to_ui_capacity    = 1024;
    size_t nodes_to_router_capacity = 256;
    size_t router_to_node_capacity  = 1024;
  };

  struct NodeHandles {
    EventReader in;
    EventWriter out;
  };

  explicit Router(const Config &config = {});

  [[nodiscard]] NodeHandles bind_node(const NodeId &node_id);

  [[nodiscard]] bool ui_try_send(const NodeId &node_id, nlohmann::json &&data,
                                 time_point ts = std::chrono::steady_clock::now()) noexcept;

  [[nodiscard]] std::optional<Event> ui_try_receive() noexcept;

  void tick(size_t budget = 1024);

  [[nodiscard]] MailboxCounters::Snapshot ui_to_router_counters() const noexcept;

  [[nodiscard]] MailboxCounters::Snapshot router_to_ui_counters() const noexcept;

  [[nodiscard]] MailboxCounters::Snapshot nodes_to_router_counters() const noexcept;

  [[nodiscard]] MailboxCounters::Snapshot
  router_to_node_counters(const NodeId &node_id) const noexcept;

private:
  Config config_;

  BoundedMPMC<Event>                                              ui_to_router_;
  BoundedMPMC<Event>                                              router_to_ui_;
  BoundedMPMC<Event>                                              nodes_to_router_;
  std::unordered_map<NodeId, std::unique_ptr<BoundedMPMC<Event>>> router_to_nodes_;

  MailboxCounters                             ui_to_router_counters_;
  MailboxCounters                             router_to_ui_counters_;
  MailboxCounters                             nodes_to_router_counters_;
  std::unordered_map<NodeId, MailboxCounters> router_to_node_counters_;
};

} // namespace holoflow_event