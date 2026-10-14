// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <mutex>
#include <spdlog/sinks/base_sink.h>
#include <string>
#include <vector>

#include "logger.hh"

namespace {
using namespace std::chrono_literals;

class BlockingSink : public spdlog::sinks::base_sink<std::mutex> {
public:
  BlockingSink() : released(release.get_future()) {}
  std::promise<void>       started, release, flushed;
  std::vector<std::string> messages;

private:
  void sink_it_(const spdlog::details::log_msg &message) override {
    if (messages.empty()) {
      started.set_value();
      EXPECT_EQ(released.wait_for(3s), std::future_status::ready);
    }
    messages.emplace_back(message.payload.data(), message.payload.size());
  }
  void              flush_() override { flushed.set_value(); }
  std::future<void> released;
};

TEST(HolovibesLoggerTest, SlowOutputDoesNotBlockSubmissionAndQueuedMessagesOwnTheirPayload) {
  auto log     = holovibes::logger()->clone("holovibes-async-test");
  auto sink    = std::make_shared<BlockingSink>();
  log->sinks() = {sink};
  log->set_level(spdlog::level::info);
  log->flush_on(spdlog::level::off);
  auto started  = sink->started.get_future();
  auto flushed  = sink->flushed.get_future();
  auto producer = std::async(std::launch::async, [&] {
    log->info("blocked output");
    std::string payload = "original payload";
    log->info("{}", payload);
    payload = "changed payload";
    log->info("last message");
    log->flush();
  });
  EXPECT_EQ(started.wait_for(3s), std::future_status::ready);
  // Release even on failure so a synchronous regression cannot hang test/process teardown.
  EXPECT_EQ(producer.wait_for(500ms), std::future_status::ready);
  sink->release.set_value();
  producer.get();
  ASSERT_EQ(flushed.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(sink->messages,
            (std::vector<std::string>{"blocked output", "original payload", "last message"}));
}
} // namespace
