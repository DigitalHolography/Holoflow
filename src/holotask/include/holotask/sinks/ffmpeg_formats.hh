#pragma once

#include <array>
#include <span>
#include <string_view>

namespace holotask::sinks {

struct FfmpegCodecInfo {
  std::string_view label;
  std::string_view name;
};

struct FfmpegFormatInfo {
  std::string_view label;
  std::string_view name;
  std::string_view extension;
  std::span<const FfmpegCodecInfo> codecs;
};

inline constexpr std::array<FfmpegCodecInfo, 3> kMp4Codecs{{
    {"MPEG-4", "mpeg4"}, {"H.264", "libopenh264"}, {"MJPEG", "mjpeg"}}};
inline constexpr std::array<FfmpegCodecInfo, 3> kAviCodecs{{
    {"MPEG-4", "mpeg4"}, {"MJPEG", "mjpeg"}, {"FFV1", "ffv1"}}};
inline constexpr std::array<FfmpegCodecInfo, 4> kMatroskaCodecs{{
    {"MPEG-4", "mpeg4"}, {"H.264", "libopenh264"}, {"FFV1", "ffv1"},
    {"VP9", "libvpx-vp9"}}};
inline constexpr std::array<FfmpegCodecInfo, 1> kWebmCodecs{{{"VP9", "libvpx-vp9"}}};

inline constexpr std::array<FfmpegFormatInfo, 6> kFfmpegFormats{{
    {"Holo", "holo", "holo", {}},
    {"NumPy", "npy", "npy", {}},
    {"MP4", "mp4", "mp4", kMp4Codecs},
    {"AVI", "avi", "avi", kAviCodecs},
    {"Matroska", "matroska", "mkv", kMatroskaCodecs},
    {"WebM", "webm", "webm", kWebmCodecs},
}};

inline const FfmpegFormatInfo *ffmpeg_format(std::string_view name) {
  for (const auto &format : kFfmpegFormats)
    if (format.name == name)
      return &format;
  return nullptr;
}

inline bool ffmpeg_codec_is_compatible(std::string_view format, std::string_view codec) {
  const auto *info = ffmpeg_format(format);
  if (info == nullptr)
    return false;
  for (const auto &candidate : info->codecs)
    if (candidate.name == codec)
      return true;
  return false;
}

} // namespace holotask::sinks
