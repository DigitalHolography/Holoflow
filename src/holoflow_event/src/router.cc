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

#include "holoflow_event/router.hh"

#include <iostream>

namespace holoflow_event {

MailboxCounters::Snapshot MailboxCounters::snapshot() const noexcept {
  return Snapshot{received_events.load(std::memory_order_relaxed),
                  dropped_events.load(std::memory_order_relaxed),
                  sent_events.load(std::memory_order_relaxed)};
}

EventReader::EventReader(BoundedMPMC<Event> *queue, MailboxCounters *counters)
    : queue_(queue), counters_(counters) {}

std::optional<Event> EventReader::try_pop() noexcept {
  auto event = queue_->try_pop();
  if (event) {
    counters_->received_events.fetch_add(1, std::memory_order_relaxed);
  }
  return event;
}

size_t EventReader::drop_all() noexcept {
  size_t dropped = 0;
  while (true) {
    auto event = queue_->try_pop();
    if (!event) {
      break;
    }
    dropped++;
  }
  counters_->dropped_events.fetch_add(dropped, std::memory_order_relaxed);
  return dropped;
}

MailboxCounters::Snapshot EventReader::counters() const noexcept { return counters_->snapshot(); }

EventWriter::EventWriter(BoundedMPMC<Event> *queue, MailboxCounters *counters)
    : queue_(queue), counters_(counters) {}

bool EventWriter::try_push(Event &&event) noexcept {
  bool ok = queue_->try_push(std::move(event));
  counters_->sent_events.fetch_add(ok, std::memory_order_relaxed);
  counters_->dropped_events.fetch_add(!ok, std::memory_order_relaxed);
  return ok;
}

MailboxCounters::Snapshot EventWriter::counters() const noexcept { return counters_->snapshot(); }

Router::Router(const Config &config)
    : config_(config), ui_to_router_(config.ui_to_router_capacity),
      router_to_ui_(config.router_to_ui_capacity),
      nodes_to_router_(config.nodes_to_router_capacity) {}

Router::NodeHandles Router::bind_node(const NodeId &node_id) {
  auto queue          = std::make_unique<BoundedMPMC<Event>>(config_.router_to_node_capacity);
  auto [it, inserted] = router_to_nodes_.try_emplace(node_id, std::move(queue));
  auto &counters      = router_to_node_counters_[node_id];

  if (!inserted) {
    std::cerr << "[holoflow_event::Router] Error: Node '" << node_id
              << "' is already bound to the router." << std::endl;
    std::abort();
  }

  EventWriter w{&nodes_to_router_, &nodes_to_router_counters_};
  EventReader r{it->second.get(), &counters};
  return NodeHandles{r, w};
}

bool Router::ui_try_send(const NodeId &node_id, nlohmann::json &&data, time_point ts) noexcept {
  Event event{EventDirection::ToNode, node_id, std::move(data), ts};
  bool  ok = ui_to_router_.try_push(std::move(event));
  ui_to_router_counters_.received_events.fetch_add(ok, std::memory_order_relaxed);
  ui_to_router_counters_.dropped_events.fetch_add(!ok, std::memory_order_relaxed);
  return ok;
}

std::optional<Event> Router::ui_try_receive() noexcept {
  auto event = router_to_ui_.try_pop();
  if (event) {
    router_to_ui_counters_.sent_events.fetch_add(1, std::memory_order_relaxed);
  }
  return event;
}

void Router::tick(size_t budget) {
  // Nodes->Router to Router->UI
  for (size_t i = 0; i < budget; i++) {
    auto event = nodes_to_router_.try_pop();
    if (!event) {
      break;
    }
    nodes_to_router_counters_.received_events.fetch_add(1, std::memory_order_relaxed);

    bool ok = router_to_ui_.try_push(std::move(*event));
    router_to_ui_counters_.sent_events.fetch_add(ok, std::memory_order_relaxed);
    router_to_ui_counters_.dropped_events.fetch_add(!ok, std::memory_order_relaxed);
  }

  // UI->Router to Router->Nodes
  for (size_t i = 0; i < budget; i++) {
    auto event = ui_to_router_.try_pop();
    if (!event) {
      break;
    }
    ui_to_router_counters_.received_events.fetch_add(1, std::memory_order_relaxed);

    auto it = router_to_nodes_.find(event->node_id);
    if (it == router_to_nodes_.end()) {
      std::cerr << "[holoflow_event::Router] Warning: Dropping event for unknown node '"
                << event->node_id << "'." << std::endl;
      ui_to_router_counters_.dropped_events.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    auto &queue    = it->second;
    auto &counters = router_to_node_counters_[event->node_id];
    bool  ok       = queue->try_push(std::move(*event));
    counters.sent_events.fetch_add(ok, std::memory_order_relaxed);
    counters.dropped_events.fetch_add(!ok, std::memory_order_relaxed);
  }
}

MailboxCounters::Snapshot Router::ui_to_router_counters() const noexcept {
  return ui_to_router_counters_.snapshot();
}

MailboxCounters::Snapshot Router::router_to_ui_counters() const noexcept {
  return router_to_ui_counters_.snapshot();
}

MailboxCounters::Snapshot Router::nodes_to_router_counters() const noexcept {
  return nodes_to_router_counters_.snapshot();
}

MailboxCounters::Snapshot Router::router_to_node_counters(const NodeId &node_id) const noexcept {
  auto it = router_to_node_counters_.find(node_id);
  if (it == router_to_node_counters_.end()) {
    std::cerr << "[holoflow_event::Router] Warning: No counters for unknown node '" << node_id
              << "'." << std::endl;
    return MailboxCounters::Snapshot{0, 0, 0};
  }
  return it->second.snapshot();
}

} // namespace holoflow_event