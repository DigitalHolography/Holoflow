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

TEST_F(HolofileSinkUpdateTest, RecordsFramesToActivePathAcrossSettingsUpdate) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_holofile_sink_test.holo";
  const auto next_path =
      std::filesystem::temp_directory_path() / "holoflow_holofile_sink_next.holo";
  std::filesystem::remove(path);
  std::filesystem::remove(next_path);

  const std::vector<TDesc> input_descs = {host_desc({2, 2, 3}, DType::U8)};
  auto task = factory.create(input_descs, settings(path.string(), 4), ctx);
  task->bind_logger(spdlog::default_logger());

  holoflow_event::Router router;
  auto handles = router.bind_node("holofile");
  holoflow::core::Tensor tensor(input_descs[0]);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx execute_ctx{inputs, {}, nullptr, &handles.out, &handles.in};
  const std::array<std::uint8_t, 24> frames{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

  ASSERT_TRUE(router.ui_try_send("holofile",
                                 {{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();
  std::memcpy(tensor.data(), frames.data(), 12);
  ASSERT_EQ(task->execute(execute_ctx), holoflow::core::OpResult::Ok);

  task = factory.update(std::move(task), input_descs, settings(next_path.string(), 4, 2), ctx);
  std::memcpy(tensor.data(), frames.data() + 12, 12);
  ASSERT_EQ(task->execute(execute_ctx), holoflow::core::OpResult::Ok);

  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(next_path));
  {
    holofile::Reader reader(path.string());
    EXPECT_EQ(reader.header().frame_count, 4);
    EXPECT_EQ(reader.header().data_size_in_bytes, frames.size());
    std::array<std::uint8_t, 24> recorded{};
    reader.read_frames(recorded.data(), 4);
    EXPECT_EQ(recorded, frames);
  }

  router.tick();
  bool finished = false, recorded = false, flushed = false;
  while (auto event = router.ui_try_receive()) {
    finished |= event->data.at("type") == "recording_finished";
    recorded |= event->data.at("type") == "recording_update" &&
                event->data.value("frame_recorded", 0) == 4;
    flushed |= event->data.at("type") == "record_flushing_finished";
  }
  EXPECT_TRUE(finished);
  EXPECT_TRUE(recorded);
  EXPECT_TRUE(flushed);

  std::filesystem::remove(path);
}
