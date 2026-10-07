#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <vector>

#include "../../src/holotask/src/sources/camera_buffer_pair.hh"
#include "../../src/holotask/src/sources/camera_buffer_queue.hh"
#include "../../src/holotask/src/sources/camera_recording_session.hh"

namespace {
using Queue   = holotask::sources::detail::CameraBufferQueue<int>;
using Session = holotask::sources::detail::CameraRecordingSession;
using Result  = Queue::PushResult;

TEST(CameraBufferPair, TimeoutRetainsFirstBankAndSuccessTransfersBoth) {
  using Pair = holotask::sources::detail::CameraBufferPair<int>;
  std::vector<int> released;
  int              a_calls = 0;
  {
    Pair       pair{[&](size_t, int buffer) { released.push_back(buffer); }};
    const auto pop_a = [&] {
      ++a_calls;
      return 1;
    };
    EXPECT_THROW(pair.poll(pop_a, []() -> int { throw std::runtime_error("timeout"); }),
                 std::runtime_error);
    EXPECT_EQ(pair.a(), 1);
    EXPECT_TRUE(released.empty());
    pair.poll(pop_a, [] { return 2; });
    EXPECT_EQ(a_calls, 1);
    EXPECT_EQ(pair.b(), 2);
    pair.transfer();
  }
  EXPECT_TRUE(released.empty());
}

TEST(CameraBufferPair, ExceptionsAndRejectedPairsReturnEachBufferOnce) {
  using Pair = holotask::sources::detail::CameraBufferPair<int>;
  std::vector<int> released;
  {
    Pair pair{[&](size_t, int buffer) { released.push_back(buffer); }};
    EXPECT_THROW(
        pair.poll([] { return 1; }, []() -> int { throw std::runtime_error("fatal error"); }),
        std::runtime_error);
  }
  EXPECT_EQ(released, (std::vector<int>{1}));
  {
    Pair pair{[&](size_t, int buffer) { released.push_back(buffer); }};
    pair.poll([] { return 2; }, [] { return 3; });
    pair.release();
    pair.release();
  }
  EXPECT_EQ(released, (std::vector<int>{1, 2, 3}));
}

TEST(CameraBufferQueue, RejectsZeroCapacity) {
  EXPECT_THROW((Queue{0, [](int) {}}), std::invalid_argument);
}

TEST(CameraBufferQueue, FullPushRetainsCallerOwnershipAndWraps) {
  std::vector<int> released;
  Queue            queue{2, [&](int frame) { released.push_back(frame); }};
  EXPECT_EQ(queue.try_push(1), Result::Accepted);
  EXPECT_EQ(queue.try_push(2), Result::Accepted);
  EXPECT_EQ(queue.try_push(3), Result::Full);
  EXPECT_TRUE(released.empty());
  EXPECT_EQ(*queue.read_a(), 1);
  queue.release_a();
  EXPECT_EQ(queue.try_push(3), Result::Accepted);
  EXPECT_EQ(*queue.read_a(), 2);
  queue.release_a();
  EXPECT_EQ(*queue.read_a(), 3);
  queue.release_a();
  EXPECT_EQ(released, (std::vector<int>{1, 2, 3}));
  EXPECT_TRUE(queue.empty());
}

TEST(CameraBufferQueue, BothReadersMustReleaseBeforeReuse) {
  for (bool recorder_first : {false, true}) {
    std::vector<int> released;
    Queue            queue{1, [&](int frame) { released.push_back(frame); }};
    queue.subscribe_b();
    ASSERT_EQ(queue.try_push(7), Result::Accepted);
    EXPECT_EQ(*queue.read_a(), 7);
    EXPECT_EQ(*queue.read_b(), 7);
    if (recorder_first)
      queue.release_b();
    else
      queue.release_a();
    EXPECT_EQ(queue.size(), 1u);
    EXPECT_EQ(queue.try_push(8), Result::Full);
    EXPECT_TRUE(released.empty());
    if (recorder_first)
      queue.release_a();
    else
      queue.release_b();
    EXPECT_EQ(released, (std::vector<int>{7}));
    EXPECT_EQ(queue.try_push(8), Result::Accepted);
    queue.unsubscribe_b();
    queue.release_a();
    EXPECT_EQ(released, (std::vector<int>{7, 8}));
  }
}

TEST(CameraBufferQueue, SubscriptionStartsAtPublicationAndUnsubscribeDrains) {
  std::vector<int> released;
  Queue            queue{3, [&](int frame) { released.push_back(frame); }};
  ASSERT_EQ(queue.try_push(1), Result::Accepted);
  queue.subscribe_b();
  EXPECT_THROW(queue.subscribe_b(), std::logic_error);
  ASSERT_EQ(queue.try_push(2), Result::Accepted);
  ASSERT_EQ(queue.try_push(3), Result::Accepted);
  EXPECT_EQ(*queue.read_b(), 2);
  queue.release_a();
  queue.release_a();
  queue.release_a();
  EXPECT_EQ(released, (std::vector<int>{1}));
  queue.unsubscribe_b();
  queue.unsubscribe_b();
  EXPECT_EQ(released, (std::vector<int>{1, 2, 3}));
  queue.subscribe_b();
  ASSERT_EQ(queue.try_push(4), Result::Accepted);
  EXPECT_EQ(*queue.read_b(), 4);
  queue.release_b();
  queue.release_a();
}

TEST(CameraBufferQueue, ClosureUnblocksReadersAndRetainsRejectedOwnership) {
  std::vector<int> released;
  Queue            queue{1, [&](int frame) { released.push_back(frame); }};
  auto             reader = std::async(std::launch::async, [&] { return queue.read_a(); });
  queue.close();
  ASSERT_EQ(reader.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(reader.get(), nullptr);
  EXPECT_EQ(queue.try_push(1), Result::Closed);
  EXPECT_TRUE(released.empty());
}

TEST(CameraBufferQueue, CancellationUnblocksAnEmptyQueue) {
  Queue             queue{1, [](int) {}};
  std::atomic<bool> cancelled{false};
  auto reader = std::async(std::launch::async, [&] { return queue.read_a(&cancelled); });
  cancelled.store(true);
  EXPECT_EQ(reader.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  queue.close(); // Ensure failure cannot leave the async worker blocked.
  EXPECT_EQ(reader.get(), nullptr);
}

TEST(CameraBufferQueue, DestructionReleasesOutstandingFramesOnce) {
  std::vector<int> released;
  {
    Queue queue{2, [&](int frame) { released.push_back(frame); }};
    queue.subscribe_b();
    ASSERT_EQ(queue.try_push(1), Result::Accepted);
    ASSERT_EQ(queue.try_push(2), Result::Accepted);
    queue.release_a();
    queue.release_b();
    queue.close();
  }
  EXPECT_EQ(released, (std::vector<int>{1, 2}));
}

TEST(CameraBufferQueue, ConcurrentProducerAndReadersKeepFramesOrdered) {
  constexpr int     count = 10000;
  std::atomic<int>  released{0};
  std::atomic<bool> ordered{true};
  Queue             queue{4, [&](int) { ++released; }};
  queue.subscribe_b();
  const auto consume = [&](bool recorder) {
    for (int expected = 0; expected < count; ++expected) {
      const auto *frame = recorder ? queue.read_b() : queue.read_a();
      if (!frame || *frame != expected) {
        ordered.store(false);
        queue.close();
        return;
      }
      if (recorder)
        queue.release_b();
      else
        queue.release_a();
    }
  };
  std::jthread a([&] { consume(false); });
  std::jthread b([&] { consume(true); });
  for (int frame = 0; frame < count && ordered.load(); ++frame) {
    while (queue.try_push(frame) == Result::Full && ordered.load())
      std::this_thread::yield();
  }
  a.join();
  b.join();
  EXPECT_TRUE(ordered.load());
  EXPECT_EQ(released.load(), count);
  EXPECT_TRUE(queue.empty());
}

TEST(CameraRecordingSession, OverflowCancelsStartupAndOnlyFirstErrorWins) {
  Session session;
  EXPECT_TRUE(session.fail("queue full"));
  EXPECT_TRUE(session.token().stop_requested());
  EXPECT_FALSE(session.fail("another error"));
  const auto result = session.finish();
  EXPECT_FALSE(result.completed);
  EXPECT_TRUE(result.discard_file);
  EXPECT_EQ(result.failure, "queue full");
}

TEST(CameraRecordingSession, CompletionCannotBeCancelledByLaterOverflow) {
  Session session;
  EXPECT_TRUE(session.finish().completed);
  EXPECT_FALSE(session.fail("queue full"));
}

TEST(CameraRecordingSession, OrdinaryWriterErrorsKeepExistingFailureBehavior) {
  Session    session;
  const auto result = session.finish("write failed");
  EXPECT_FALSE(result.completed);
  EXPECT_FALSE(result.discard_file);
  EXPECT_EQ(result.failure, "write failed");
}

TEST(CameraRecordingSession, CancellationAndRestartAreIndependent) {
  Session old_session, new_session;
  old_session.request_stop();
  EXPECT_FALSE(old_session.fail("queue full"));
  EXPECT_FALSE(old_session.finish().completed);
  EXPECT_FALSE(new_session.token().stop_requested());
  EXPECT_TRUE(new_session.finish().completed);
}

TEST(CameraRecordingSession, OverflowAndCompletionHaveOneWinner) {
  for (int i = 0; i < 200; ++i) {
    Session         session;
    bool            failed = false;
    Session::Result result;
    std::jthread    failure([&] { failed = session.fail("queue full"); });
    std::jthread    completion([&] { result = session.finish(); });
    failure.join();
    completion.join();
    EXPECT_NE(failed, result.completed);
    EXPECT_EQ(result.failure.empty(), result.completed);
  }
}

TEST(CameraRecordingSession, FailedFileIsDeletedAfterClose) {
  namespace fs = std::filesystem;
  const auto path =
      fs::temp_directory_path() /
      ("holoflow-camera-recording-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".holo");
  {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.good());
    file << "incomplete";
  }
  EXPECT_EQ(
      holotask::sources::detail::remove_incomplete_camera_recording(path.string(), "queue full"),
      "queue full");
  EXPECT_FALSE(fs::exists(path));
  EXPECT_EQ(
      holotask::sources::detail::remove_incomplete_camera_recording(path.string(), "queue full"),
      "queue full");
}
} // namespace
