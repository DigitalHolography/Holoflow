// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "signal_history_dispatcher.hh"

#include <QMetaObject>
#include <iterator>
#include <utility>

namespace holovibes::ui {

SignalHistoryDispatcher::SignalHistoryDispatcher(QObject *context, std::function<void()> reset,
                                                 std::function<void(Samples)> append)
    : context_(context), reset_(std::move(reset)), append_(std::move(append)) {}

bool SignalHistoryDispatcher::queue_reset() {
  {
    std::lock_guard lock(mutex_);
    if (state_ == State::Cancelled || reset_queued_)
      return false;
    reset_queued_ = true;
  }
  if (context_.isNull() ||
      !QMetaObject::invokeMethod(
          context_.data(), [self = shared_from_this()] { self->reset(); }, Qt::QueuedConnection)) {
    cancel();
    return false;
  }
  return true;
}

void SignalHistoryDispatcher::enqueue(Samples samples) {
  bool schedule = false;
  {
    std::lock_guard lock(mutex_);
    if (state_ == State::Cancelled || context_.isNull() || samples.empty())
      return;
    pending_.insert(pending_.end(), std::make_move_iterator(samples.begin()),
                    std::make_move_iterator(samples.end()));
    if (state_ == State::Ready && !scheduled_) {
      scheduled_ = true;
      schedule   = true;
    }
  }
  if (schedule && !queue_drain())
    cancel();
}

void SignalHistoryDispatcher::cancel() {
  std::lock_guard lock(mutex_);
  state_ = State::Cancelled;
  pending_.clear();
}

void SignalHistoryDispatcher::reset() {
  {
    std::lock_guard lock(mutex_);
    if (state_ != State::WaitingForReset || context_.isNull())
      return;
  }
  reset_();
  {
    std::lock_guard lock(mutex_);
    // A newer update may cancel this run while the GUI action executes.
    if (state_ == State::Cancelled)
      return;
    state_     = State::Ready;
    scheduled_ = true;
  }
  drain();
}

bool SignalHistoryDispatcher::queue_drain() {
  return !context_.isNull() &&
         QMetaObject::invokeMethod(
             context_.data(), [self = shared_from_this()] { self->drain(); }, Qt::QueuedConnection);
}

void SignalHistoryDispatcher::drain() {
  Samples samples;
  {
    std::lock_guard lock(mutex_);
    if (state_ != State::Ready || context_.isNull())
      return;
    samples.swap(pending_);
    scheduled_ = false;
  }
  // An already executing delivery may finish after cancellation, but GUI serialization ensures
  // it finishes before the next run's reset. Never hold the producer mutex across widget calls.
  if (!samples.empty())
    append_(std::move(samples));
}

} // namespace holovibes::ui
