// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <QCoreApplication>
#include <QEvent>
#include <QThread>
#include <gtest/gtest.h>
#include <thread>

#include "ui/widgets/signal_history_dispatcher.hh"

namespace holovibes::ui {
namespace {

class SignalHistoryDispatcherTest : public ::testing::Test {
protected:
  int                                      argc_     = 1;
  char                                     name_[16] = "history-test";
  char                                    *argv_[2]  = {name_, nullptr};
  QCoreApplication                         application_{argc_, argv_};
  QObject                                  context_;
  int                                      resets_ = 0;
  std::vector<ZernikeHistorySample>        received_;
  std::shared_ptr<SignalHistoryDispatcher> make_dispatcher() {
    return std::make_shared<SignalHistoryDispatcher>(
        &context_,
        [this] {
          ++resets_;
          received_.clear();
        },
        [this](auto samples) {
          EXPECT_EQ(QThread::currentThread(), context_.thread());
          received_.insert(received_.end(), samples.begin(), samples.end());
        });
  }
  void process() { QCoreApplication::sendPostedEvents(&context_, QEvent::MetaCall); }
};

TEST_F(SignalHistoryDispatcherTest, BuffersNewDataWhileGuiIsStalledAndResetsBeforeDelivery) {
  auto dispatcher = make_dispatcher();
  ASSERT_TRUE(dispatcher->queue_reset());
  std::thread producer([&] {
    for (int i = 0; i < 10; ++i)
      dispatcher->enqueue({{4, {double(i), double(i)}}});
  });
  producer.join(); // Producers complete even though the GUI hasn't processed the reset.
  EXPECT_EQ(resets_, 0);
  EXPECT_TRUE(received_.empty());
  process();
  EXPECT_EQ(resets_, 1);
  ASSERT_EQ(received_.size(), 10);
  EXPECT_EQ(received_.front().sample.time_seconds, 0);
  EXPECT_EQ(received_.back().sample.time_seconds, 9);
}

TEST_F(SignalHistoryDispatcherTest, CancelledOldDeliveryCannotRepopulateNewHistory) {
  auto old = make_dispatcher();
  ASSERT_TRUE(old->queue_reset());
  process();
  old->enqueue({{4, {5, 100}}}); // Queue an old-run delivery, but don't process it.
  old->cancel();
  auto next = make_dispatcher();
  ASSERT_TRUE(next->queue_reset());
  next->enqueue({{4, {0, 200}}});
  old->enqueue({{4, {6, 101}}});
  process();
  ASSERT_EQ(received_.size(), 1);
  EXPECT_EQ(received_[0].sample.value, 200);
  EXPECT_EQ(received_[0].sample.time_seconds, 0);
}

TEST_F(SignalHistoryDispatcherTest, RapidUpdatesAndFailedRunCancelPendingResets) {
  auto first = make_dispatcher();
  ASSERT_TRUE(first->queue_reset());
  first->enqueue({{4, {0, 1}}});
  first->cancel();
  auto failed = make_dispatcher();
  ASSERT_TRUE(failed->queue_reset());
  failed->cancel();
  auto latest = make_dispatcher();
  ASSERT_TRUE(latest->queue_reset());
  latest->enqueue({{4, {0, 3}}});
  process();
  EXPECT_EQ(resets_, 1);
  ASSERT_EQ(received_.size(), 1);
  EXPECT_EQ(received_[0].sample.value, 3);
}

TEST_F(SignalHistoryDispatcherTest, CancellationDuringResetCannotOpenDelivery) {
  std::shared_ptr<SignalHistoryDispatcher> dispatcher;
  dispatcher = std::make_shared<SignalHistoryDispatcher>(
      &context_,
      [&] {
        ++resets_;
        dispatcher->cancel();
      },
      [&](auto) { ADD_FAILURE() << "Cancelled reset must not deliver samples"; });
  ASSERT_TRUE(dispatcher->queue_reset());
  dispatcher->enqueue({{4, {0, 1}}});
  process();
  dispatcher->enqueue({{4, {1, 2}}});
  process();
  EXPECT_EQ(resets_, 1);
}

TEST_F(SignalHistoryDispatcherTest, ProducersDoNotWaitForExecutingGuiReset) {
  std::shared_ptr<SignalHistoryDispatcher> dispatcher;
  dispatcher = std::make_shared<SignalHistoryDispatcher>(
      &context_,
      [&] {
        std::thread producer([&] { dispatcher->enqueue({{4, {0, 42}}}); });
        producer.join(); // Would deadlock if the dispatcher held its mutex during reset.
      },
      [&](auto samples) { received_ = std::move(samples); });
  ASSERT_TRUE(dispatcher->queue_reset());
  process();
  ASSERT_EQ(received_.size(), 1);
  EXPECT_EQ(received_[0].sample.value, 42);
}

TEST_F(SignalHistoryDispatcherTest, DestroyedContextDropsResetAndDeliveryCallbacks) {
  auto context    = std::make_unique<QObject>();
  auto dispatcher = std::make_shared<SignalHistoryDispatcher>(
      context.get(), [&] { ADD_FAILURE() << "Destroyed context must not reset"; },
      [&](auto) { ADD_FAILURE() << "Destroyed context must not receive samples"; });
  ASSERT_TRUE(dispatcher->queue_reset());
  dispatcher->enqueue({{4, {0, 1}}});
  context.reset();
  QCoreApplication::sendPostedEvents(nullptr, QEvent::MetaCall);
  dispatcher->enqueue({{4, {1, 2}}});
  EXPECT_FALSE(dispatcher->queue_reset());
}

} // namespace
} // namespace holovibes::ui
