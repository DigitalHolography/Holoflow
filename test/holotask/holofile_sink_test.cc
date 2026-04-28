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

#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
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
