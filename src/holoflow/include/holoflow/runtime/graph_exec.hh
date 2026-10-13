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

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/fmt/fmt.h>
#include <string>
#include <vector>

#include "holoflow/runtime/graph_exec_types.hh"
#include "holoflow_event/router.hh"

namespace holoflow::runtime {

struct NodeMetrics {
  double   average_duration_ms                = 0.0;
  double   runs_per_second                    = 0.0;
  double   host_throughput_bytes_per_second   = 0.0;
  double   device_throughput_bytes_per_second = 0.0;
  uint64_t sample_count                       = 0;
  bool     individual_timing_available        = true;
};

class Scheduler {
public:
  Scheduler(const GraphPlan &graph, const std::vector<Section> &sections, ExecResouces &res,
            std::chrono::milliseconds metrics_interval = std::chrono::milliseconds{1000});

  ~Scheduler();

  void set_metrics_interval(std::chrono::milliseconds interval);
  [[nodiscard]] std::map<std::string, NodeMetrics> metrics() const;
  /// Host submission timing/counts for section graph launches, not GPU execution duration.
  [[nodiscard]] std::map<std::string, NodeMetrics> section_graph_metrics() const;
  [[nodiscard]] nlohmann::json                     section_graph_diagnostics() const;

  void start();
  /// Suspend with acquired buffers and completed operations retained for start().
  void request_pause();
  void request_stop();
  void wait();

  bool is_running() const;
  bool stop_requested() const;

  [[nodiscard]] bool ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept;

  [[nodiscard]] std::optional<holoflow_event::Event> ui_try_receive() noexcept;

  Scheduler(const Scheduler &)            = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  Scheduler(Scheduler &&)                 = delete;
  Scheduler &operator=(Scheduler &&)      = delete;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holoflow::runtime

template <> struct fmt::formatter<holoflow::runtime::NodeMetrics> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const holoflow::runtime::NodeMetrics &m, FormatContext &ctx) const {
    if (!m.individual_timing_available) {
      return fmt::format_to(ctx.out(),
                            "{{avg: n/a (section graph), rps: {:.3f}, host: {:.3f} B/s, device: "
                            "{:.3f} B/s, samples: {}}}",
                            m.runs_per_second, m.host_throughput_bytes_per_second,
                            m.device_throughput_bytes_per_second, m.sample_count);
    }
    return fmt::format_to(
        ctx.out(),
        "{{avg: {:.3f} ms, rps: {:.3f}, host: {:.3f} B/s, device: {:.3f} B/s, samples: {}}}",
        m.average_duration_ms, m.runs_per_second, m.host_throughput_bytes_per_second,
        m.device_throughput_bytes_per_second, m.sample_count);
  }
};
