// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/tasks.hh"
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <vector>

namespace holoflow::python {

class LatencyMarker final : public core::ISyncTask {
public:
  core::OpResult execute(core::SyncCtx &) override { return core::OpResult::Ok; }
  bool           supports_cuda_graph() const noexcept override { return true; }
  void           record_cuda_graph(core::CudaGraphCtx &) override {}
};

class LatencyMarkerFactory final : public core::ISyncTaskFactory {
public:
  core::InferResult infer(std::span<const core::TDesc> inputs,
                          const nlohmann::json &) const override {
    if (inputs.size() != 1)
      throw std::invalid_argument("Latency marker expects one tensor");
    return {.input_descs   = {inputs[0]},
            .output_descs  = {inputs[0]},
            .in_place      = {{0, 0}},
            .owned_inputs  = {false},
            .owned_outputs = {false},
            .kind          = core::TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &) const override {
    return std::make_unique<LatencyMarker>();
  }
};

class LatencyMeasurement {
public:
  using Clock = std::chrono::steady_clock;
  using Json  = nlohmann::json;

  LatencyMeasurement(std::string id, size_t batch_size, size_t history_batches, uint64_t first,
                     uint64_t window, uint64_t stride, int interval_ms)
      : id(std::move(id)), batch_size_(batch_size), history_(history_batches), next_frame_(first),
        window_(window), stride_(stride), interval_(interval_ms) {}

  const std::string id;

  void input(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    history_[batches_ % history_.size()] = now;
    ++batches_;
  }

  void output(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    const auto      first       = next_frame_;
    const auto      max         = std::numeric_limits<uint64_t>::max();
    const bool      valid_range = !sequence_overflow_ && window_ - 1 <= max - first;
    sequence_overflow_ |= stride_ > max - first;
    if (!sequence_overflow_)
      next_frame_ += stride_;
    if (valid_range) {
      const auto oldest = first / batch_size_;
      const auto newest = (first + (window_ - 1)) / batch_size_;
      if (newest < batches_ && batches_ - oldest <= history_.size()) {
        using Milliseconds     = std::chrono::duration<double, std::milli>;
        const double oldest_ms = Milliseconds(now - history_[oldest % history_.size()]).count();
        const double newest_ms = Milliseconds(now - history_[newest % history_.size()]).count();
        if (newest_ms >= 0) {
          total_.add(oldest_ms, newest_ms);
          period_.add(oldest_ms, newest_ms);
          return;
        }
      }
    }
    ++total_.unmatched;
    ++period_.unmatched;
  }

  // Deadline state belongs to the reporting worker after start.
  void              start(Clock::time_point now) { previous_ = now; }
  Clock::time_point deadline() const { return previous_ + interval_; }

  Json sample(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    auto            result     = period_.json();
    result["type"]             = "latency";
    result["interval_seconds"] = std::chrono::duration<double>(now - previous_).count();
    previous_                  = now;
    period_                    = {};
    return result;
  }

  Json snapshot() const {
    std::lock_guard lock(mutex_);
    return total_.json();
  }

private:
  struct Stats {
    double latest = 0, minimum = 0, maximum = 0, sum = 0;
    void   add(double value, uint64_t count) {
      latest  = value;
      minimum = count ? std::min(minimum, value) : value;
      maximum = count ? std::max(maximum, value) : value;
      sum += value;
    }
    Json json(uint64_t count) const {
      if (!count)
        return nullptr;
      return {{"latest", latest}, {"min", minimum}, {"mean", sum / count}, {"max", maximum}};
    }
  };
  struct Summary {
    uint64_t matched = 0, unmatched = 0;
    Stats    oldest, newest;
    void     add(double oldest_ms, double newest_ms) {
      oldest.add(oldest_ms, matched);
      newest.add(newest_ms, matched);
      ++matched;
    }
    Json json() const {
      return {{"matched_outputs", matched},
              {"unmatched_outputs", unmatched},
              {"oldest_ms", oldest.json(matched)},
              {"newest_ms", newest.json(matched)}};
    }
  };

  size_t                         batch_size_;
  std::vector<Clock::time_point> history_;
  uint64_t                       next_frame_, window_, stride_;
  bool                           sequence_overflow_ = false;
  uint64_t                       batches_           = 0;
  std::chrono::milliseconds      interval_;
  Clock::time_point              previous_;
  mutable std::mutex             mutex_;
  Summary                        total_, period_;
};

} // namespace holoflow::python
