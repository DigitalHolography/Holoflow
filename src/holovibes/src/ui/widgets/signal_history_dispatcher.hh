// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <QObject>
#include <QPointer>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "signal_history.hh"

namespace holovibes::ui {

struct ZernikeHistorySample {
  int          noll_index;
  SignalSample sample;
};

// One dispatcher per computation run. Widget actions execute only on the context's GUI thread.
class SignalHistoryDispatcher : public std::enable_shared_from_this<SignalHistoryDispatcher> {
public:
  using Samples = std::vector<ZernikeHistorySample>;
  SignalHistoryDispatcher(QObject *context, std::function<void()> reset,
                          std::function<void(Samples)> append);

  bool queue_reset();
  void enqueue(Samples samples);
  void cancel();

private:
  enum class State { WaitingForReset, Ready, Cancelled };
  void reset();
  void drain();
  bool queue_drain();

  QPointer<QObject>            context_;
  std::function<void()>        reset_;
  std::function<void(Samples)> append_;
  std::mutex                   mutex_;
  Samples                      pending_;
  State                        state_        = State::WaitingForReset;
  bool                         reset_queued_ = false;
  bool                         scheduled_    = false;
};

// Selected/read sequentially on the manager thread. Tasks keep a fixed copy of current.
struct SignalHistoryDispatcherProvider {
  std::shared_ptr<SignalHistoryDispatcher> current;
};

} // namespace holovibes::ui
