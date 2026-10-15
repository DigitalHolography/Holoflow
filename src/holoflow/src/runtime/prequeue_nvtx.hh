// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <string>

namespace holoflow::runtime::detail {

// Process ranges may overlap and survive worker replacement during pause/resume. Their duration
// describes reservation ownership, including acquisition and paused time, rather than GPU time.
class PrequeueNvtxSlot {
public:
  PrequeueNvtxSlot(const std::string &section, size_t slot)
      : lifetime_name_(std::format("{} / Slot {}", section, slot)),
        submit_name_(std::format("Slot {}: Submit", slot)),
        retire_name_(std::format("Slot {}: Retire", slot)),
        color_(slot == 0 ? nvtx3::rgb{78, 121, 167} : nvtx3::rgb{242, 142, 43}) {}
  ~PrequeueNvtxSlot() { finish(); }
  PrequeueNvtxSlot(const PrequeueNvtxSlot &)            = delete;
  PrequeueNvtxSlot &operator=(const PrequeueNvtxSlot &) = delete;

  bool active() const { return lifetime_.has_value(); }
  bool submitted() const { return submitted_; }
  void begin(uint64_t iteration) {
    iteration_ = iteration;
    lifetime_  = nvtx3::start_range(attributes(lifetime_name_));
  }
  void mark_submitted() { submitted_ = true; }
  void finish() noexcept {
    if (lifetime_) {
      nvtx3::end_range(*lifetime_);
      lifetime_.reset();
    }
    submitted_ = false;
  }
  nvtx3::event_attributes attributes(const nvtx3::registered_string &name) const {
    return nvtx3::event_attributes{name, color_, nvtx3::payload{iteration_}};
  }
  const nvtx3::registered_string &operation(bool retirement) const {
    return retirement ? retire_name_ : submit_name_;
  }

private:
  nvtx3::registered_string           lifetime_name_, submit_name_, retire_name_;
  nvtx3::rgb                         color_;
  uint64_t                           iteration_ = 0;
  std::optional<nvtx3::range_handle> lifetime_;
  bool                               submitted_ = false;
};

enum class PrequeueNvtxPhase { Acquire, Compute, Push, Wait, Commit, Release };

// All nested scopes are thread-bound and must be destroyed before leaving a worker. Keep the
// acquisition/wait scopes alive across retries; no NVTX calls are made for unchanged phases.
class PrequeueNvtxRanges {
public:
  explicit PrequeueNvtxRanges(const nvtx3::registered_string *section = nullptr) {
    if (section)
      section_.emplace(*section);
  }
  void operation(const PrequeueNvtxSlot &slot, bool retirement = false) {
    if (slot_ == &slot && retirement_ == retirement)
      return;
    end_operation();
    slot_       = &slot;
    retirement_ = retirement;
    operation_.emplace(slot.attributes(slot.operation(retirement)));
  }
  void phase(PrequeueNvtxPhase phase) {
    if (phase_ && phase_id_ == phase)
      return;
    end_wait();
    phase_.reset();
    phase_id_ = phase;
    static const std::array<nvtx3::registered_string, 6> names{
        nvtx3::registered_string{"Acquire buffers"}, nvtx3::registered_string{"Compute"},
        nvtx3::registered_string{"Async push"},      nvtx3::registered_string{"Wait"},
        nvtx3::registered_string{"Commit outputs"},  nvtx3::registered_string{"Release outputs"}};
    phase_.emplace(names.at(static_cast<size_t>(phase)));
  }
  void wait(const nvtx3::registered_string &name) {
    if (waiting_for_ == &name)
      return;
    end_wait();
    waiting_.emplace(name);
    waiting_for_ = &name;
  }
  void end_wait() {
    waiting_.reset();
    waiting_for_ = nullptr;
  }
  void end_operation() {
    end_wait();
    phase_.reset();
    operation_.reset();
    slot_ = nullptr;
  }

private:
  std::optional<nvtx3::scoped_range> section_, operation_, phase_, waiting_;
  const PrequeueNvtxSlot            *slot_        = nullptr;
  const nvtx3::registered_string    *waiting_for_ = nullptr;
  PrequeueNvtxPhase                  phase_id_    = PrequeueNvtxPhase::Acquire;
  bool                               retirement_  = false;
};

// Ends the process lifetime on both successful retirement and exception unwinding. Declare before
// the CPU ranges so those close first. Other slots are closed by the section's exception handler.
struct FinishPrequeueNvtxSlot {
  PrequeueNvtxSlot &slot;
  ~FinishPrequeueNvtxSlot() { slot.finish(); }
};

} // namespace holoflow::runtime::detail
