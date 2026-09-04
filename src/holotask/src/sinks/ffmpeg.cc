// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "holotask/sinks/ffmpeg.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "holoflow/core/tensor.hh"
#include "recording_geometry.hh"
#include "recording_sink.hh"

namespace holotask::sinks {
namespace {

std::string ffmpeg_error(const char *operation, int error) {
  char message[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(error, message, sizeof(message));
  return std::string(operation) + " failed: " + message;
}

void check_ffmpeg(int result, const char *operation) {
  if (result < 0)
    throw std::runtime_error(ffmpeg_error(operation, result));
}

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("FfmpegFactory error: " + message);
}

AVPixelFormat input_pixel_format(const detail::RecordingGeometry &geometry) {
  return geometry.bits_per_pixel == 8 ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_GRAY16LE;
}

AVPixelFormat choose_pixel_format(const AVCodec *codec, const detail::RecordingGeometry &geometry) {
  const void *configurations      = nullptr;
  int         configuration_count = 0;
  check_ffmpeg(avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0,
                                            &configurations, &configuration_count),
               "avcodec_get_supported_config");
  const auto *pixel_formats = static_cast<const AVPixelFormat *>(configurations);

  const auto supports = [&](AVPixelFormat format) {
    if (pixel_formats == nullptr)
      return true;
    return std::find(pixel_formats, pixel_formats + configuration_count, format) !=
           pixel_formats + configuration_count;
  };

  if (geometry.bits_per_pixel == 16) {
    if (pixel_formats == nullptr)
      throw std::invalid_argument("FfmpegFactory error: codec pixel formats are unspecified for "
                                  "16-bit grayscale input");
    if (supports(AV_PIX_FMT_GRAY16LE))
      return AV_PIX_FMT_GRAY16LE;
    throw std::invalid_argument("FfmpegFactory error: codec does not support 16-bit grayscale");
  }

  if (pixel_formats == nullptr)
    return AV_PIX_FMT_YUV420P;

  constexpr AVPixelFormat preferred_formats[] = {
      AV_PIX_FMT_GRAY8,   AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
      AV_PIX_FMT_YUV444P, AV_PIX_FMT_NV12,    AV_PIX_FMT_NONE,
  };
  for (const auto preferred : preferred_formats) {
    if (supports(preferred))
      return preferred;
  }
  throw std::invalid_argument("FfmpegFactory error: codec has no supported 8-bit pixel format");
}

void validate_ffmpeg_settings(const FfmpegSettings            &settings,
                              const detail::RecordingGeometry &geometry) {
  check(settings.count > 0, "count must be positive");
  check(std::isfinite(settings.fps) && settings.fps > 0.0, "fps must be finite and positive");
  check(!settings.format.empty(), "format must not be empty");
  check(!settings.codec.empty(), "codec must not be empty");

  const auto *output_format = av_guess_format(settings.format.c_str(), nullptr, nullptr);
  check(output_format != nullptr, "unknown output format: " + settings.format);

  const auto *codec = avcodec_find_encoder_by_name(settings.codec.c_str());
  check(codec != nullptr, "unknown video encoder: " + settings.codec);
  check(codec->type == AVMEDIA_TYPE_VIDEO, "codec is not a video encoder: " + settings.codec);
  check(avformat_query_codec(output_format, codec->id, FF_COMPLIANCE_NORMAL) > 0,
        "codec '" + settings.codec + "' is incompatible with format '" + settings.format + "'");
  (void)choose_pixel_format(codec, geometry);
}

class FfmpegWriter final : public detail::RecordingSink {
public:
  FfmpegWriter(FfmpegSettings settings, detail::RecordingGeometry geometry)
      : settings_(std::move(settings)), geometry_(geometry) {}

  ~FfmpegWriter() override { release(); }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_events(ctx, settings_.path, settings_.count);
    if (!is_recording())
      return holoflow::core::OpResult::Ok;

    try {
      initialize();
      auto       &input        = ctx.inputs[0];
      const auto  batch_size   = static_cast<int>(input.desc.shape[0]);
      const auto  frame_count  = std::min(settings_.count - frames_buffered_, batch_size);
      const auto  frame_stride = input.desc.strides[0];
      const auto *data         = reinterpret_cast<const std::uint8_t *>(input.data());

      for (int index = 0; index < frame_count; ++index) {
        encode_frame(data + static_cast<std::size_t>(index) * frame_stride, input.desc.strides[1]);
        ++frames_buffered_;
      }

      if (frames_buffered_ >= settings_.count) {
        finalize();
        emit_finished_event(ctx);
        reset();
      }
    } catch (const std::exception &error) {
      release();
      if (!recording_path().empty())
        (void)std::remove(recording_path().c_str());
      emit_failed_event(ctx, error.what());
      reset();
    }

    return holoflow::core::OpResult::Ok;
  }

  bool can_reuse(const FfmpegSettings &settings, detail::RecordingGeometry geometry) const {
    return settings.count == settings_.count && settings.fps == settings_.fps &&
           settings.format == settings_.format && settings.codec == settings_.codec &&
           detail::same_geometry(geometry, geometry_);
  }

  void update_settings(FfmpegSettings settings) {
    settings_ = std::move(settings);
    update_recording_path(settings_.path);
  }

protected:
  void on_recording_stopped() override { release(); }
  void on_recording_reset() override { release(); }

private:
  void initialize() {
    if (format_context_ != nullptr)
      return;

    validate_ffmpeg_settings(settings_, geometry_);
    const auto *codec    = avcodec_find_encoder_by_name(settings_.codec.c_str());
    output_pixel_format_ = choose_pixel_format(codec, geometry_);

    check_ffmpeg(avformat_alloc_output_context2(&format_context_, nullptr, settings_.format.c_str(),
                                                recording_path().c_str()),
                 "avformat_alloc_output_context2");
    stream_ = avformat_new_stream(format_context_, nullptr);
    if (stream_ == nullptr)
      throw std::runtime_error("avformat_new_stream failed");

    codec_context_ = avcodec_alloc_context3(codec);
    if (codec_context_ == nullptr)
      throw std::runtime_error("avcodec_alloc_context3 failed");

    codec_context_->codec_id     = codec->id;
    codec_context_->codec_type   = AVMEDIA_TYPE_VIDEO;
    codec_context_->width        = static_cast<int>(geometry_.frame_width);
    codec_context_->height       = static_cast<int>(geometry_.frame_height);
    codec_context_->pix_fmt      = output_pixel_format_;
    codec_context_->framerate    = av_d2q(settings_.fps, 100000);
    codec_context_->time_base    = av_inv_q(codec_context_->framerate);
    codec_context_->gop_size     = 12;
    codec_context_->max_b_frames = 0;
    if ((format_context_->oformat->flags & AVFMT_GLOBALHEADER) != 0)
      codec_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    check_ffmpeg(avcodec_open2(codec_context_, codec, nullptr), "avcodec_open2");
    check_ffmpeg(avcodec_parameters_from_context(stream_->codecpar, codec_context_),
                 "avcodec_parameters_from_context");
    stream_->time_base = codec_context_->time_base;

    if ((format_context_->oformat->flags & AVFMT_NOFILE) == 0)
      check_ffmpeg(avio_open(&format_context_->pb, recording_path().c_str(), AVIO_FLAG_WRITE),
                   "avio_open");
    check_ffmpeg(avformat_write_header(format_context_, nullptr), "avformat_write_header");

    frame_  = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (frame_ == nullptr || packet_ == nullptr)
      throw std::runtime_error("Failed to allocate FFmpeg frame or packet");
    frame_->format = output_pixel_format_;
    frame_->width  = codec_context_->width;
    frame_->height = codec_context_->height;
    check_ffmpeg(av_frame_get_buffer(frame_, 32), "av_frame_get_buffer");

    sws_context_ =
        sws_getContext(codec_context_->width, codec_context_->height, input_pixel_format(geometry_),
                       codec_context_->width, codec_context_->height, output_pixel_format_,
                       SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_context_ == nullptr)
      throw std::runtime_error("sws_getContext failed");
  }

  void encode_frame(const std::uint8_t *data, std::size_t row_stride) {
    check_ffmpeg(av_frame_make_writable(frame_), "av_frame_make_writable");
    const std::uint8_t *source[]        = {data, nullptr, nullptr, nullptr};
    const int           source_stride[] = {static_cast<int>(row_stride), 0, 0, 0};
    if (sws_scale(sws_context_, source, source_stride, 0, codec_context_->height, frame_->data,
                  frame_->linesize) <= 0)
      throw std::runtime_error("sws_scale failed");

    frame_->pts = frames_buffered_;
    check_ffmpeg(avcodec_send_frame(codec_context_, frame_), "avcodec_send_frame");
    write_packets();
  }

  void write_packets() {
    while (true) {
      const auto result = avcodec_receive_packet(codec_context_, packet_);
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
        return;
      check_ffmpeg(result, "avcodec_receive_packet");
      av_packet_rescale_ts(packet_, codec_context_->time_base, stream_->time_base);
      packet_->stream_index = stream_->index;
      check_ffmpeg(av_interleaved_write_frame(format_context_, packet_),
                   "av_interleaved_write_frame");
      av_packet_unref(packet_);
    }
  }

  void finalize() {
    check_ffmpeg(avcodec_send_frame(codec_context_, nullptr), "avcodec_send_frame (flush)");
    write_packets();
    check_ffmpeg(av_write_trailer(format_context_), "av_write_trailer");
    release();
  }

  void release() noexcept {
    if (sws_context_ != nullptr)
      sws_freeContext(sws_context_);
    sws_context_ = nullptr;
    av_packet_free(&packet_);
    av_frame_free(&frame_);
    avcodec_free_context(&codec_context_);
    if (format_context_ != nullptr) {
      if (format_context_->pb != nullptr && (format_context_->oformat->flags & AVFMT_NOFILE) == 0)
        avio_closep(&format_context_->pb);
      avformat_free_context(format_context_);
    }
    format_context_ = nullptr;
    stream_         = nullptr;
  }

  FfmpegSettings            settings_;
  detail::RecordingGeometry geometry_;
  AVFormatContext          *format_context_      = nullptr;
  AVCodecContext           *codec_context_       = nullptr;
  AVStream                 *stream_              = nullptr;
  AVFrame                  *frame_               = nullptr;
  AVPacket                 *packet_              = nullptr;
  SwsContext               *sws_context_         = nullptr;
  AVPixelFormat             output_pixel_format_ = AV_PIX_FMT_NONE;
};

} // namespace

void to_json(nlohmann::json &j, const FfmpegSettings &settings) {
  j = {{"path", settings.path},
       {"count", settings.count},
       {"fps", settings.fps},
       {"format", settings.format},
       {"codec", settings.codec}};
}

void from_json(const nlohmann::json &j, FfmpegSettings &settings) {
  j.at("path").get_to(settings.path);
  j.at("count").get_to(settings.count);
  j.at("fps").get_to(settings.fps);
  j.at("format").get_to(settings.format);
  j.at("codec").get_to(settings.codec);
}

holoflow::core::InferResult FfmpegFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<FfmpegSettings>();
  check(input_descs.size() == 1, "expected exactly one input tensor");
  const auto &input = input_descs[0];
  check(input.mem_loc == holoflow::core::MemLoc::Host, "input tensor must be in Host memory");
  check(input.shape.size() == 3, "input tensor must have rank 3 (batch, height, width)");
  check(input.dtype == holoflow::core::DType::U8 || input.dtype == holoflow::core::DType::U16,
        "unsupported input dtype");
  check(input.shape[0] > 0 && input.shape[1] > 0 && input.shape[2] > 0,
        "input dimensions must be positive");
  check(input.shape[1] <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            input.shape[2] <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "input dimensions are too large for FFmpeg");
  check(input.strides.size() == 3, "input tensor must provide rank-3 strides");
  check(input.strides[1] >= input.shape[2] * holoflow::core::size_of(input.dtype),
        "input row stride is too small");
  const auto batch = static_cast<int>(input.shape[0]);
  check(settings.count > 0 && settings.count % batch == 0,
        "frame count must be positive and divisible by batch size");
  const auto geometry = detail::recording_geometry_from_desc(input);
  validate_ffmpeg_settings(settings, geometry);

  return {.input_descs   = {input},
          .output_descs  = {},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
FfmpegFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx &) const {
  infer(input_descs, jsettings);
  return std::make_unique<FfmpegWriter>(jsettings.get<FfmpegSettings>(),
                                        detail::recording_geometry_from_desc(input_descs[0]));
}

std::unique_ptr<holoflow::core::ISyncTask>
FfmpegFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *writer = dynamic_cast<FfmpegWriter *>(old_task.get());
  if (writer == nullptr)
    return create(input_descs, jsettings, ctx);

  infer(input_descs, jsettings);
  const auto settings = jsettings.get<FfmpegSettings>();
  const auto geometry = detail::recording_geometry_from_desc(input_descs[0]);
  if (!writer->can_reuse(settings, geometry))
    return create(input_descs, jsettings, ctx);

  writer->update_settings(settings);
  return old_task;
}

} // namespace holotask::sinks
