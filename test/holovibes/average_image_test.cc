// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <QFile>
#include <QImage>
#include <QTemporaryDir>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "tasks/sinks/average_image.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::Storage;
using holoflow::core::TDesc;
using holoflow::core::TView;

class AverageImageHarness {
public:
  AverageImageHarness(const std::string &path, int count)
      : image_desc_({1, 2, 3}, DType::U8, MemLoc::Host), valid_desc_({1}, DType::U8, MemLoc::Host),
        image_storage_{MemLoc::Host, image_.size(), reinterpret_cast<std::byte *>(image_.data())},
        valid_storage_{MemLoc::Host, sizeof(valid_), reinterpret_cast<std::byte *>(&valid_)},
        handles_(router_.bind_node("record")) {
    const holovibes::tasks::sinks::AverageImageSettings settings{
        .path = path, .count = count, .format = "png", .output_16bit = false};
    const std::array input_descs{image_desc_, valid_desc_};
    task_ = factory_.create(input_descs, settings, {});
    if (!router_.ui_try_send("record", {{"type", "start_recording"}, {"record_path", path}}))
      throw std::runtime_error("failed to enqueue start_recording event");
    router_.tick();
  }

  OpResult execute(std::uint8_t value, bool valid) {
    image_.fill(value);
    valid_ = valid ? std::uint8_t{1} : std::uint8_t{0};
    std::array inputs{TView{image_desc_, &image_storage_}, TView{valid_desc_, &valid_storage_}};
    std::array<TView, 0>    outputs{};
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, &handles_.out, &handles_.in};
    const auto              result = task_->execute(ctx);
    router_.tick();
    return result;
  }

  std::optional<holoflow_event::Event> event() { return router_.ui_try_receive(); }

private:
  TDesc                                        image_desc_;
  TDesc                                        valid_desc_;
  std::array<std::uint8_t, 6>                  image_{};
  std::uint8_t                                 valid_ = 0;
  Storage                                      image_storage_;
  Storage                                      valid_storage_;
  holoflow_event::Router                       router_;
  holoflow_event::Router::NodeHandles          handles_;
  holovibes::tasks::sinks::AverageImageFactory factory_;
  std::unique_ptr<holoflow::core::ISyncTask>   task_;
};

void write_single_image(const std::string &path, const std::string &format, bool output_16bit,
                        const TDesc &desc, void *data) {
  holovibes::tasks::sinks::AverageImageFactory        factory;
  const holovibes::tasks::sinks::AverageImageSettings settings{
      .path = path, .count = 1, .format = format, .output_16bit = output_16bit};
  const std::array     input_descs{desc};
  auto                 task = factory.create(input_descs, settings, {});
  Storage              storage{MemLoc::Host, desc.num_bytes(), reinterpret_cast<std::byte *>(data)};
  std::array           inputs{TView{desc, &storage}};
  std::array<TView, 0> outputs{};
  std::atomic<bool>    cancelled{false};
  holoflow_event::Router router;
  auto                   handles = router.bind_node("record");
  if (!router.ui_try_send("record", {{"type", "start_recording"}, {"record_path", path}}))
    throw std::runtime_error("failed to enqueue start_recording event");
  router.tick();
  holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, &handles.out, &handles.in};
  ASSERT_EQ(task->execute(ctx), OpResult::Ok);
}

TEST(AverageImageTest, ReplacesRejectedFramesUntilRequestedCountIsAccepted) {
  QTemporaryDir directory;
  ASSERT_TRUE(directory.isValid());
  const auto          path = directory.filePath("average.png");
  AverageImageHarness writer(path.toStdString(), 2);

  EXPECT_EQ(writer.execute(10, true), OpResult::Ok);
  EXPECT_EQ(writer.execute(200, false), OpResult::Ok);
  EXPECT_EQ(writer.execute(220, false), OpResult::Ok);
  EXPECT_EQ(writer.execute(30, true), OpResult::Ok);

  const auto event = writer.event();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->data.at("type"), "recording_finished");
  const QImage image(path);
  ASSERT_FALSE(image.isNull());
  EXPECT_EQ(image.constScanLine(0)[0], std::uint8_t{20});
}

TEST(AverageImageTest, FailsAfterThreeTimesTheRequestedAttempts) {
  QTemporaryDir directory;
  ASSERT_TRUE(directory.isValid());
  const auto          path = directory.filePath("rejected.png");
  AverageImageHarness writer(path.toStdString(), 2);

  for (int attempt = 0; attempt < 6; ++attempt)
    EXPECT_EQ(writer.execute(100, false), OpResult::Ok);

  const auto event = writer.event();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->data.at("type"), "recording_failed");
  EXPECT_TRUE(QImage(path).isNull());
}

TEST(AverageImageTest, PreservesRawU16PngAndWritesProcessedEightBitFormats) {
  QTemporaryDir directory;
  ASSERT_TRUE(directory.isValid());

  std::array<std::uint16_t, 6> raw{1000, 2000, 3000, 4000, 5000, 6000};
  const TDesc                  raw_desc({1, 2, 3}, DType::U16, MemLoc::Host);
  const auto                   raw_path = directory.filePath("raw.png");
  write_single_image(raw_path.toStdString(), "png", true, raw_desc, raw.data());
  QFile raw_file(raw_path);
  ASSERT_TRUE(raw_file.open(QIODevice::ReadOnly));
  const auto raw_png = raw_file.readAll();
  ASSERT_GT(raw_png.size(), 24);
  EXPECT_EQ(static_cast<unsigned char>(raw_png[24]), 16);

  std::array<float, 6> processed{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  const TDesc          processed_desc({1, 2, 3}, DType::F32, MemLoc::Host);
  const auto           png_path = directory.filePath("processed.png");
  const auto           jpg_path = directory.filePath("processed.jpg");
  write_single_image(png_path.toStdString(), "png", false, processed_desc, processed.data());
  write_single_image(jpg_path.toStdString(), "jpg", false, processed_desc, processed.data());
  QFile png_file(png_path);
  ASSERT_TRUE(png_file.open(QIODevice::ReadOnly));
  const auto png = png_file.readAll();
  ASSERT_GT(png.size(), 24);
  EXPECT_EQ(static_cast<unsigned char>(png[24]), 8);
  EXPECT_FALSE(QImage(jpg_path).isNull());
}

} // namespace
