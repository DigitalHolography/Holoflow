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

#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/syncs/zernike_defocus_z_prop.hh"

#include "tensor_test_buffer.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {

class CountingSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
  [[nodiscard]] size_t count() const { return count_.load(); }

protected:
  void sink_it_(const spdlog::details::log_msg &) override { ++count_; }
  void flush_() override {}

private:
  std::atomic<size_t> count_{0};
};

TDesc input_desc() { return TDesc({1}, DType::F32, MemLoc::Host); }

holotask::syncs::ZernikeDefocusZPropSettings settings(double interval_seconds = 1.0) {
  return {
      .indexes          = {4},
      .lambda           = 532.0e-9f,
      .z_curr           = 0.1f,
      .pupil_radius     = 1.0e-3f,
      .interval_seconds = interval_seconds,
  };
}

std::vector<std::byte> as_bytes(float value) {
  std::vector<std::byte> bytes(sizeof(value));
  std::memcpy(bytes.data(), &value, sizeof(value));
  return bytes;
}

} // namespace

TEST(ZernikeDefocusZPropSettingsTest, DefaultsLegacyJsonToOneSecond) {
  const nlohmann::json legacy = {
      {"indexes", {4}},
      {"lambda", 532.0e-9f},
      {"z_curr", 0.1f},
      {"pupil_radius", 1.0e-3f},
  };

  const auto parsed = legacy.get<holotask::syncs::ZernikeDefocusZPropSettings>();

  EXPECT_DOUBLE_EQ(parsed.interval_seconds, 1.0);
}

TEST(ZernikeDefocusZPropSettingsTest, RoundTripsConfiguredInterval) {
  const auto original = settings(0.25);
  const auto json     = nlohmann::json(original);

  EXPECT_DOUBLE_EQ(json.at("interval_seconds").get<double>(), 0.25);
  EXPECT_EQ(json.get<holotask::syncs::ZernikeDefocusZPropSettings>(), original);
}

TEST(ZernikeDefocusZPropInferTest, RejectsInvalidIntervals) {
  holotask::syncs::ZernikeDefocusZPropFactory factory;
  const std::vector<TDesc>                     inputs{input_desc()};

  for (const double interval : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW((void)factory.infer(inputs, settings(interval)), std::invalid_argument);
  }
}

TEST(ZernikeDefocusZPropExecutionTest, ExecutesImmediatelyAndThenAtConfiguredInterval) {
  holotask::syncs::ZernikeDefocusZPropFactory factory;
  const auto                                  desc = input_desc();
  const std::vector<TDesc>                    input_descs{desc};
  const auto                                  task_settings = settings(0.02);
  auto task = factory.create(input_descs, task_settings, {});

  auto sink   = std::make_shared<CountingSink>();
  auto logger = std::make_shared<spdlog::logger>("zernike-defocus-z-prop-test", sink);
  task->bind_logger(logger);

  holonp_test::TensorTestBuffer input(desc);
  input.upload(as_bytes(0.25f));
  std::vector<holoflow::core::TView> input_views{input.view()};
  std::vector<holoflow::core::TView> output_views;
  std::atomic<bool>                   cancelled{false};
  holoflow::core::SyncCtx ctx{
      .inputs       = input_views,
      .outputs      = output_views,
      .cancelled    = &cancelled,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  EXPECT_EQ(sink->count(), 1u);

  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  EXPECT_EQ(sink->count(), 1u);

  std::this_thread::sleep_for(std::chrono::milliseconds(30));

  EXPECT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  EXPECT_EQ(sink->count(), 2u);
}
