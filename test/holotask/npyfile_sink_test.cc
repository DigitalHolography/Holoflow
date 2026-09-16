#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "holotask/sinks/npyfile.hh"

namespace {
using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

TDesc          desc(DType dtype = DType::U8) { return TDesc({2, 2, 3}, dtype, MemLoc::Host); }
nlohmann::json settings(const std::string &path, int count = 4) {
  return holotask::sinks::NpyfileSettings{path, count, true};
}
std::filesystem::path test_path() {
  return std::filesystem::temp_directory_path() / "holoflow_npy_sink_test.npy";
}
} // namespace

TEST(NpyfileSinkTest, AcceptsSupportedInputs) {
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc>        u8_input{desc(DType::U8)};
  const std::vector<TDesc>        u16_input{desc(DType::U16)};
  EXPECT_EQ(factory.infer(u8_input, settings("test.npy")).input_descs[0].dtype, DType::U8);
  EXPECT_EQ(factory.infer(u16_input, settings("test.npy")).input_descs[0].dtype, DType::U16);
}

TEST(NpyfileSinkTest, RejectsInvalidInputs) {
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc>        f32_input{TDesc({2, 2, 3}, DType::F32, MemLoc::Host)};
  const std::vector<TDesc>        device_input{TDesc({2, 2, 3}, DType::U8, MemLoc::Device)};
  const std::vector<TDesc>        rank2_input{TDesc({2, 3}, DType::U8, MemLoc::Host)};
  const std::vector<TDesc>        valid_input{desc()};
  EXPECT_THROW(factory.infer(f32_input, settings("test.npy")), std::invalid_argument);
  EXPECT_THROW(factory.infer(device_input, settings("test.npy")), std::invalid_argument);
  EXPECT_THROW(factory.infer(rank2_input, settings("test.npy")), std::invalid_argument);
  EXPECT_THROW(factory.infer(valid_input, settings("test.npy", 3)), std::invalid_argument);
}

TEST(NpyfileSinkTest, WritesStandardNpyArrayAndCompletionEvent) {
  const auto path = test_path();
  std::filesystem::remove(path);
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc>        input_descs{desc()};
  auto                      task = factory.create(input_descs, settings(path.string(), 4), {});
  task->bind_logger(spdlog::default_logger());
  holoflow_event::Router    router;
  auto                      handles = router.bind_node("npy");
  std::vector<std::uint8_t> batch_values(12);
  for (std::size_t i = 0; i < batch_values.size(); ++i)
    batch_values[i] = static_cast<std::uint8_t>(i);
  std::vector<std::uint8_t> values;
  values.insert(values.end(), batch_values.begin(), batch_values.end());
  values.insert(values.end(), batch_values.begin(), batch_values.end());
  TDesc                  batch_desc({2, 2, 3}, DType::U8, MemLoc::Host);
  holoflow::core::Tensor tensor(batch_desc);
  std::memcpy(tensor.data(), batch_values.data(), batch_values.size());
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx              ctx{inputs, {}, nullptr, &handles.out, &handles.in};

  ASSERT_TRUE(
      router.ui_try_send("npy", {{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  router.tick();
  const auto event = router.ui_try_receive();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->data.at("type"), "recording_finished");

  std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file.good());
  std::array<char, 10> prefix{};
  file.read(prefix.data(), prefix.size());
  EXPECT_EQ(std::string(prefix.data(), 6), "\x93NUMPY");
  const auto header_size =
      static_cast<unsigned char>(prefix[8]) | (static_cast<unsigned char>(prefix[9]) << 8);
  std::string header(header_size, '\0');
  file.read(header.data(), header.size());
  EXPECT_NE(header.find("'descr': '<u1'"), std::string::npos);
  EXPECT_NE(header.find("'fortran_order': False"), std::string::npos);
  EXPECT_NE(header.find("'shape': (4, 2, 3)"), std::string::npos);
  EXPECT_EQ((10 + header_size) % 16, 0);
  std::vector<std::uint8_t> actual(values.size());
  file.read(reinterpret_cast<char *>(actual.data()), actual.size());
  EXPECT_TRUE(std::equal(values.begin(), values.end(), actual.begin()));
  file.close();
  std::filesystem::remove(path);
}

TEST(NpyfileSinkTest, ReusesOnlyWhenGeometryAndCountMatch) {
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc>        input_descs{desc()};
  auto                            task = factory.create(input_descs, settings("first.npy"), {});
  auto                           *old  = task.get();
  task = factory.update(std::move(task), input_descs, settings("second.npy"), {});
  EXPECT_EQ(task.get(), old);
  task = factory.update(std::move(task), input_descs, settings("third.npy", 6), {});
  EXPECT_NE(task.get(), old);
}

TEST(NpyfileSinkTest, RemovesIncompleteFileAfterStopRecording) {
  const auto path = test_path();
  std::filesystem::remove(path);
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc> input_descs{desc()};
  auto task = factory.create(input_descs, settings(path.string(), 4), {});
  task->bind_logger(spdlog::default_logger());
  holoflow_event::Router router;
  auto handles = router.bind_node("npy");
  TDesc batch_desc({2, 2, 3}, DType::U8, MemLoc::Host);
  holoflow::core::Tensor tensor(batch_desc);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &handles.out, &handles.in};

  ASSERT_TRUE(
      router.ui_try_send("npy", {{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_TRUE(router.ui_try_send("npy", {{"type", "stop_recording"}}));
  router.tick();
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);

  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(NpyfileSinkTest, ReportsFailedFileCreation) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_missing_npy_dir" /
                    "recording.npy";
  std::filesystem::remove(path);
  holotask::sinks::NpyfileFactory factory;
  const std::vector<TDesc> input_descs{desc()};
  auto task = factory.create(input_descs, settings(path.string(), 2), {});
  task->bind_logger(spdlog::default_logger());
  holoflow_event::Router router;
  auto handles = router.bind_node("npy");
  holoflow::core::Tensor tensor(TDesc({2, 2, 3}, DType::U8, MemLoc::Host));
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &handles.out, &handles.in};

  ASSERT_TRUE(
      router.ui_try_send("npy", {{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  router.tick();
  const auto event = router.ui_try_receive();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->data.at("type"), "recording_failed");
  EXPECT_NE(event->data.at("message").get<std::string>().find("Failed to open NPY file"),
            std::string::npos);
}
