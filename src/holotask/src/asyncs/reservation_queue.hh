// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <stdexcept>

#include "holoflow/core/tasks.hh"

namespace holotask::asyncs::detail {

// Per-side SPSC bookkeeping. Only that side's worker accesses these private reservations;
// cross-thread visibility and capacity still use the queue's committed atomic cursors.
template <class T> class ReservationQueue {
public:
  bool     empty() const { return size_ == 0; }
  bool     full() const { return size_ == slots_.size(); }
  const T &back() const { return slots_[(head_ + size_ - 1) % slots_.size()].data; }
  holoflow::core::AsyncTicket push(T data) {
    if (full())
      throw std::logic_error("Too many queue reservations");
    auto &slot = slots_[(head_ + size_++) % slots_.size()];
    slot       = {++sequence_, std::move(data)};
    return slot.ticket;
  }
  T &get(holoflow::core::AsyncTicket ticket) {
    for (size_t i = 0; i < size_; ++i) {
      auto &slot = slots_[(head_ + i) % slots_.size()];
      if (slot.ticket == ticket)
        return slot.data;
    }
    throw std::logic_error("Unknown queue reservation");
  }
  T pop(holoflow::core::AsyncTicket ticket) {
    if (empty() || slots_[head_].ticket != ticket)
      throw std::logic_error("Queue reservations must retire in FIFO order");
    auto data = slots_[head_].data;
    head_     = (head_ + 1) % slots_.size();
    --size_;
    return data;
  }
  void cancel(holoflow::core::AsyncTicket ticket) {
    if (empty() || slots_[(head_ + size_ - 1) % slots_.size()].ticket != ticket)
      throw std::logic_error("Queue reservations must cancel in reverse order");
    --size_;
  }

private:
  struct Slot {
    holoflow::core::AsyncTicket ticket = 0;
    T                           data{};
  };
  std::array<Slot, 2>         slots_{};
  size_t                      head_ = 0, size_ = 0;
  holoflow::core::AsyncTicket sequence_ = 0;
};

} // namespace holotask::asyncs::detail
