// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

#include <spdlog/spdlog.h>

#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "holotask/sinks/ffmpeg.hh"

namespace {
using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

TDesc desc(DType dtype = DType::U8, std::size_t batch = 1) {
  return TDesc({batch, 2, 4}, dtype, MemLoc::Host);
}

nlohmann::json settings(std::string path, int count = 2, double fps = 25.0,
                        std::string format = "matroska", std::string codec = "ffv1") {
  return holotask::sinks::FfmpegSettings{std::move(path), count, fps, std::move(format),
                                         std::move(codec)};
}

std::filesystem::path test_path(const char *name) {
  return std::filesystem::temp_directory_path() / name;
}

struct RecordingContext {
  holoflow_event::Router              router;
  holoflow_event::Router::NodeHandles handles;

  RecordingContext() : handles(router.bind_node("ffmpeg")) {}
};

void start_recording(RecordingContext &recording, const std::string &path) {
  ASSERT_TRUE(
      recording.router.ui_try_send("ffmpeg", {{"type", "start_recording"}, {"record_path", path}}));
  recording.router.tick();
}

std::vector<holoflow_event::Event> pump_until_terminal(
    holoflow::core::ISyncTask &task, holoflow::core::SyncCtx &ctx,
    RecordingContext &recording, std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  std::vector<holoflow_event::Event> events;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    (void)task.execute(ctx);
    recording.router.tick();
    bool terminal = false;
    while (auto event = recording.router.ui_try_receive()) {
      const auto type = event->data.at("type");
      terminal |= type == "recording_finished" || type == "recording_failed";
      events.push_back(std::move(*event));
    }
    if (terminal)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return events;
}

void expect_rejected(const holotask::sinks::FfmpegFactory &factory, TDesc input,
                     const nlohmann::json &video_settings) {
  const std::vector<TDesc> inputs{std::move(input)};
  EXPECT_THROW(factory.infer(inputs, video_settings), std::invalid_argument);
}

void decode_u8(const std::filesystem::path &path, int &width, int &height, int &frames,
               std::vector<std::uint8_t> &result) {
  AVFormatContext *input = nullptr;
  ASSERT_EQ(avformat_open_input(&input, path.string().c_str(), nullptr, nullptr), 0);
  ASSERT_EQ(avformat_find_stream_info(input, nullptr), 0);
  const auto stream_index = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  ASSERT_GE(stream_index, 0);
  const auto *stream  = input->streams[stream_index];
  const auto *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
  ASSERT_NE(decoder, nullptr);
  AVCodecContext *codec = avcodec_alloc_context3(decoder);
  ASSERT_NE(codec, nullptr);
  ASSERT_EQ(avcodec_parameters_to_context(codec, stream->codecpar), 0);
  ASSERT_EQ(avcodec_open2(codec, decoder, nullptr), 0);
  AVPacket *packet = av_packet_alloc();
  AVFrame  *frame  = av_frame_alloc();
  ASSERT_NE(packet, nullptr);
  ASSERT_NE(frame, nullptr);

  result.clear();
  frames = 0;
  while (av_read_frame(input, packet) >= 0) {
    if (packet->stream_index == stream_index) {
      ASSERT_EQ(avcodec_send_packet(codec, packet), 0);
      while (avcodec_receive_frame(codec, frame) == 0) {
        width  = frame->width;
        height = frame->height;
        ASSERT_EQ(frame->format, AV_PIX_FMT_GRAY8);
        for (int row = 0; row < height; ++row)
          result.insert(result.end(), frame->data[0] + row * frame->linesize[0],
                        frame->data[0] + row * frame->linesize[0] + width);
        ++frames;
      }
    }
    av_packet_unref(packet);
  }
  ASSERT_EQ(avcodec_send_packet(codec, nullptr), 0);
  while (avcodec_receive_frame(codec, frame) == 0) {
    width  = frame->width;
    height = frame->height;
    ASSERT_EQ(frame->format, AV_PIX_FMT_GRAY8);
    for (int row = 0; row < height; ++row)
      result.insert(result.end(), frame->data[0] + row * frame->linesize[0],
                    frame->data[0] + row * frame->linesize[0] + width);
    ++frames;
  }
  av_frame_free(&frame);
  av_packet_free(&packet);
  avcodec_free_context(&codec);
  avformat_close_input(&input);
}

void decode_u16(const std::filesystem::path &path, int &width, int &height, int &frames,
                std::vector<std::uint16_t> &result) {
  AVFormatContext *input = nullptr;
  ASSERT_EQ(avformat_open_input(&input, path.string().c_str(), nullptr, nullptr), 0);
  ASSERT_EQ(avformat_find_stream_info(input, nullptr), 0);
  const auto stream_index = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  ASSERT_GE(stream_index, 0);
  const auto *stream  = input->streams[stream_index];
  const auto *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
  ASSERT_NE(decoder, nullptr);
  AVCodecContext *codec = avcodec_alloc_context3(decoder);
  ASSERT_NE(codec, nullptr);
  ASSERT_EQ(avcodec_parameters_to_context(codec, stream->codecpar), 0);
  ASSERT_EQ(avcodec_open2(codec, decoder, nullptr), 0);
  AVPacket *packet = av_packet_alloc();
  AVFrame  *frame  = av_frame_alloc();
  ASSERT_NE(packet, nullptr);
  ASSERT_NE(frame, nullptr);

  result.clear();
  frames                   = 0;
  const auto collect_frame = [&] {
    width  = frame->width;
    height = frame->height;
    ASSERT_EQ(frame->format, AV_PIX_FMT_GRAY16LE);
    for (int row = 0; row < height; ++row) {
      const auto *row_data =
          reinterpret_cast<const std::uint16_t *>(frame->data[0] + row * frame->linesize[0]);
      result.insert(result.end(), row_data, row_data + width);
    }
    ++frames;
  };
  while (av_read_frame(input, packet) >= 0) {
    if (packet->stream_index == stream_index) {
      ASSERT_EQ(avcodec_send_packet(codec, packet), 0);
      while (avcodec_receive_frame(codec, frame) == 0)
        collect_frame();
    }
    av_packet_unref(packet);
  }
  ASSERT_EQ(avcodec_send_packet(codec, nullptr), 0);
  while (avcodec_receive_frame(codec, frame) == 0)
    collect_frame();
  av_frame_free(&frame);
  av_packet_free(&packet);
  avcodec_free_context(&codec);
  avformat_close_input(&input);
}

} // namespace

TEST(FfmpegSinkTest, AcceptsSupportedInputsAndBatchDivisibleCount) {
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc>       u8_input{desc(DType::U8)};
  const std::vector<TDesc>       u16_input{desc(DType::U16)};
  EXPECT_EQ(factory.infer(u8_input, settings("test.mkv")).input_descs[0].dtype, DType::U8);
  EXPECT_EQ(factory.infer(u16_input, settings("test.mkv")).input_descs[0].dtype, DType::U16);
  const std::vector<TDesc> batch_input{desc(DType::U8, 2)};
  EXPECT_EQ(factory.infer(batch_input, settings("test.mkv", 2)).input_descs[0].shape[0], 2);
  expect_rejected(factory, desc(DType::U8, 3), settings("x.mkv", 2));
}

TEST(FfmpegSinkTest, RejectsInvalidInputsAndSettings) {
  holotask::sinks::FfmpegFactory factory;
  expect_rejected(factory, TDesc({1, 2, 4}, DType::F32, MemLoc::Host), settings("x.mkv"));
  expect_rejected(factory, TDesc({1, 2, 4}, DType::U8, MemLoc::Device), settings("x.mkv"));
  expect_rejected(factory, TDesc({2, 4}, DType::U8, MemLoc::Host), settings("x.mkv"));
  expect_rejected(factory, desc(), settings("x.mkv", 0));
  expect_rejected(factory, desc(), settings("x.mkv", 2, 0.0));
  expect_rejected(factory, desc(), settings("x.mkv", 2, std::numeric_limits<double>::infinity()));
  expect_rejected(factory, desc(), settings("x.mkv", 2, 25.0, "unknown", "ffv1"));
  expect_rejected(factory, desc(), settings("x.mkv", 2, 25.0, "matroska", "unknown"));
  expect_rejected(factory, desc(), settings("x.webm", 2, 25.0, "webm", "ffv1"));
  expect_rejected(factory, desc(DType::U16), settings("x.mp4", 2, 25.0, "mp4", "mpeg4"));
}

TEST(FfmpegSinkTest, ValidatesResizeAlgorithmOnlyWhenResizingIsEnabled) {
  holotask::sinks::FfmpegFactory factory;
  auto disabled = settings("x.mkv");
  disabled["resize_algorithm"] = "not-an-algorithm";
  const std::vector<TDesc> input{desc()};
  EXPECT_NO_THROW(factory.infer(input, disabled));

  auto enabled = settings("x.mkv");
  enabled["resize_to_square"] = true;
  enabled["resize_algorithm"] = "not-an-algorithm";
  expect_rejected(factory, desc(), enabled);
}

TEST(FfmpegSinkTest, ReusesOnlyWhenAllEncoderAndGeometrySettingsMatch) {
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc>       input{desc()};
  auto                           task = factory.create(input, settings("first.mkv"), {});
  auto                          *original = task.get();

  task = factory.update(std::move(task), input, settings("second.mkv"), {});
  EXPECT_EQ(task.get(), original); // Path changes are mutable while the writer is idle.

  original = task.get();
  task = factory.update(std::move(task), input, settings("third.mkv", 4), {});
  EXPECT_NE(task.get(), original); // Count is part of the reuse contract.

  original = task.get();
  auto resized = settings("fourth.mkv");
  resized["resize_to_square"] = true;
  resized["resize_algorithm"] = "CpuBilinear";
  task = factory.update(std::move(task), input, resized, {});
  EXPECT_NE(task.get(), original);
}

TEST(FfmpegSinkTest, EncodesU8FramesAndEmitsCompletion) {
  const auto path = test_path("holoflow_ffmpeg_sink_test.mkv");
  std::filesystem::remove(path);
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc>       input_descs{desc()};
  auto                           task = factory.create(input_descs, settings(path.string()), {});
  task->bind_logger(spdlog::default_logger());
  RecordingContext            recording;
  std::array<std::uint8_t, 8> values{0, 1, 2, 3, 4, 5, 6, 7};
  holoflow::core::Tensor      tensor(input_descs[0]);
  std::memcpy(tensor.data(), values.data(), values.size());
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &recording.handles.out, &recording.handles.in};

  start_recording(recording, path.string());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  const auto events = pump_until_terminal(*task, ctx, recording);

  size_t recorded = 0;
  bool   flushed = false;
  bool   finished = false;
  for (const auto &event : events) {
    const auto type = event.data.at("type");
    if (type == "recording_update")
      recorded = event.data.value("frame_recorded", 0);
    flushed |= type == "record_flushing_finished";
    finished |= type == "recording_finished";
  }
  EXPECT_EQ(recorded, 2);
  EXPECT_TRUE(flushed);
  EXPECT_TRUE(finished);

  int                       width  = 0;
  int                       height = 0;
  int                       frames = 0;
  std::vector<std::uint8_t> decoded;
  decode_u8(path, width, height, frames, decoded);
  EXPECT_EQ(decoded, std::vector<std::uint8_t>({0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(width, 4);
  EXPECT_EQ(height, 2);
  EXPECT_EQ(frames, 2);
  std::filesystem::remove(path);
}

TEST(FfmpegSinkTest, EncodesLosslessU16Frames) {
  const auto path = test_path("holoflow_ffmpeg_sink_u16.mkv");
  std::filesystem::remove(path);
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc>       input_descs{desc(DType::U16)};
  auto                           task = factory.create(input_descs, settings(path.string()), {});
  task->bind_logger(spdlog::default_logger());
  RecordingContext             recording;
  std::array<std::uint16_t, 8> values{0, 1, 255, 256, 1024, 32768, 65534, 65535};
  holoflow::core::Tensor       tensor(input_descs[0]);
  std::memcpy(tensor.data(), values.data(), sizeof(values));
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &recording.handles.out, &recording.handles.in};

  start_recording(recording, path.string());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  const auto events = pump_until_terminal(*task, ctx, recording);
  ASSERT_TRUE(std::any_of(events.begin(), events.end(), [](const auto &event) {
    return event.data.at("type") == "recording_finished";
  }));

  int                        width  = 0;
  int                        height = 0;
  int                        frames = 0;
  std::vector<std::uint16_t> decoded;
  decode_u16(path, width, height, frames, decoded);
  std::vector<std::uint16_t> expected(values.begin(), values.end());
  expected.insert(expected.end(), values.begin(), values.end());
  EXPECT_EQ(decoded, expected);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(height, 2);
  EXPECT_EQ(frames, 2);
  std::filesystem::remove(path);
}

TEST(FfmpegSinkTest, ResizesRectangularFramesToSquareOutput) {
  const auto path = test_path("holoflow_ffmpeg_sink_square.mkv");
  std::filesystem::remove(path);
  auto video_settings = settings(path.string());
  video_settings["resize_to_square"] = true;
  video_settings["resize_algorithm"] = "CpuBilinear";
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc> input_descs{desc()};
  auto task = factory.create(input_descs, video_settings, {});
  task->bind_logger(spdlog::default_logger());
  RecordingContext recording;
  holoflow::core::Tensor tensor(input_descs[0]);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &recording.handles.out, &recording.handles.in};

  start_recording(recording, path.string());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  const auto events = pump_until_terminal(*task, ctx, recording);
  ASSERT_TRUE(std::any_of(events.begin(), events.end(), [](const auto &event) {
    return event.data.at("type") == "recording_finished";
  }));
  int width = 0, height = 0, frames = 0;
  std::vector<std::uint8_t> decoded;
  decode_u8(path, width, height, frames, decoded);
  EXPECT_EQ(width, 4);
  EXPECT_EQ(height, 4);
  EXPECT_EQ(frames, 2);
  std::filesystem::remove(path);
}

TEST(FfmpegSinkTest, RemovesIncompleteOutputAfterStopRecording) {
  const auto path = test_path("holoflow_ffmpeg_sink_cancel.mkv");
  std::filesystem::remove(path);
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc>       input_descs{desc()};
  auto                           task = factory.create(input_descs, settings(path.string()), {});
  task->bind_logger(spdlog::default_logger());
  RecordingContext                     recording;
  holoflow::core::Tensor               tensor(input_descs[0]);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &recording.handles.out, &recording.handles.in};

  start_recording(recording, path.string());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  ASSERT_TRUE(recording.router.ui_try_send("ffmpeg", {{"type", "stop_recording"}}));
  recording.router.tick();
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(FfmpegSinkTest, ReportsOutputCreationFailureFromWriterThread) {
  const auto directory = test_path("holoflow_missing_ffmpeg_directory");
  std::filesystem::remove_all(directory);
  const auto path = directory / "recording.mkv";
  holotask::sinks::FfmpegFactory factory;
  const std::vector<TDesc> input_descs{desc()};
  auto task = factory.create(input_descs, settings(path.string(), 1), {});
  task->bind_logger(spdlog::default_logger());
  RecordingContext recording;
  holoflow::core::Tensor tensor(input_descs[0]);
  std::array<holoflow::core::TView, 1> inputs{tensor.view()};
  holoflow::core::SyncCtx ctx{inputs, {}, nullptr, &recording.handles.out, &recording.handles.in};

  start_recording(recording, path.string());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  const auto events = pump_until_terminal(*task, ctx, recording);
  const auto failed = std::find_if(events.begin(), events.end(), [](const auto &event) {
    return event.data.at("type") == "recording_failed";
  });
  ASSERT_NE(failed, events.end());
  EXPECT_NE(failed->data.at("message").get<std::string>().find("failed"), std::string::npos);
}
