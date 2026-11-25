// Copyright 2026 Digital Holography Foundation
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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "holofile/holofile.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "holotask/sinks/holofile.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::SyncCreateCtx;
using holoflow::core::TDesc;

namespace {

TDesc host_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Host);
}

nlohmann::json settings(std::string path, int count, int pipeline_version = 1) {
  return holotask::sinks::HolofileSettings{
      .path              = std::move(path),
      .count             = count,
      .pipeline_settings = nlohmann::json{{"version", pipeline_version}},
      .use_buffer        = true,
  };
}

} // namespace

class HolofileSinkUpdateTest : public ::testing::Test {
protected:
  holotask::sinks::HolofileFactory factory;
  SyncCreateCtx                    ctx{};
};

TEST_F(HolofileSinkUpdateTest, ReusesTaskWhenOnlyMetadataChanges) {
  const std::vector<TDesc> input = {host_desc({2, 4, 4}, DType::U8)};

  auto  task = factory.create(input, settings("first.holo", 4), ctx);
  auto *old  = task.get();

  task = factory.update(std::move(task), input, settings("second.holo", 4, 2), ctx);

  EXPECT_EQ(task.get(), old);
}

TEST_F(HolofileSinkUpdateTest, ReusesTaskWhenOnlyBatchSizeChanges) {
  const std::vector<TDesc> old_input = {host_desc({2, 4, 4}, DType::U8)};
  const std::vector<TDesc> new_input = {host_desc({1, 4, 4}, DType::U8)};

  auto  task = factory.create(old_input, settings("first.holo", 4), ctx);
  auto *old  = task.get();

  task = factory.update(std::move(task), new_input, settings("second.holo", 4), ctx);

  EXPECT_EQ(task.get(), old);
}

TEST_F(HolofileSinkUpdateTest, RecreatesTaskWhenBufferSizeChanges) {
  const std::vector<TDesc> input = {host_desc({2, 4, 4}, DType::U8)};

  auto  task = factory.create(input, settings("first.holo", 4), ctx);
  auto *old  = task.get();

  task = factory.update(std::move(task), input, settings("second.holo", 6), ctx);

  EXPECT_NE(task.get(), old);
}

TEST_F(HolofileSinkUpdateTest, RecordsFrames) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_holofile_sink_test.holo";
  std::filesystem::remove(path);

  const std::vector<TDesc> input_descs = {host_desc({2, 2, 3}, DType::U8)};
  auto                     task = factory.create(input_descs, settings(path.string(), 4), ctx);
  task->bind_logger(spdlog::default_logger());

  holoflow_event::Router router;
  auto                   handles = router.bind_node("holofile");

  holoflow::core::Tensor               tensor(input_descs[0]);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx              execute_ctx{inputs, {}, nullptr, &handles.out, &handles.in};

  const std::array<std::uint8_t, 24> frames{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

  ASSERT_TRUE(router.ui_try_send("holofile",
                                 {{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();

  using Clock            = std::chrono::steady_clock;
  constexpr auto timeout = std::chrono::seconds(10);

  bool   finished = false;
  size_t recorded = 0;
  bool   flushed  = false;

  size_t batch_index = 0;

  const auto execute_batch = [&]() {
    const auto *data = frames.data() + batch_index * 12;
    std::memcpy(tensor.data(), data, 12);
    ++batch_index;
    if (batch_index * 12 >= frames.size())
      batch_index = 0;
    return task->execute(execute_ctx);
  };

  const auto start = Clock::now();

  while (Clock::now() - start < timeout && !finished) {
    ASSERT_EQ(execute_batch(), holoflow::core::OpResult::Ok);

    router.tick();

    while (auto event = router.ui_try_receive()) {
      const auto &data = event->data;
      const auto &type = data.at("type");

      finished |= type == "recording_finished";
      if (type == "recording_update")
        recorded = data.value("frame_recorded", 0);
      flushed |= type == "record_flushing_finished";
    }
  }

  ASSERT_TRUE(finished) << "Recording did not finish before timeout";
  ASSERT_TRUE(std::filesystem::exists(path));

  {
    holofile::Reader reader(path.string());

    EXPECT_EQ(reader.header().frame_count, 4);
    EXPECT_EQ(reader.header().data_size_in_bytes, frames.size());

    std::array<std::uint8_t, 24> actual{};
    reader.read_frames(actual.data(), 4);

    EXPECT_EQ(actual, frames);
  }

  // Drain any remaining events.
  router.tick();
  while (auto event = router.ui_try_receive()) {
    const auto &data = event->data;
    const auto &type = data.at("type");

    finished |= type == "recording_finished";
    if (type == "recording_update")
      recorded = data.value("frame_recorded", 0);
    flushed |= type == "record_flushing_finished";
  }

  EXPECT_TRUE(finished);
  EXPECT_EQ(recorded, 4);
  EXPECT_TRUE(flushed);

  std::filesystem::remove(path);
}