// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/pipelines/focus_stack.hh"
#include "holofile/holofile.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/angular_spectrum.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/temporal_fft.hh"
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

namespace holotask::pipelines {
void to_json(nlohmann::json &j, const HoloFocusSettings &s) {
  j = {{"frame", s.frame},
       {"optics", s.optics},
       {"method", s.method},
       {"window", s.window},
       {"temporal_focus_m", s.temporal_focus_m},
       {"temporal_pitch_m", s.temporal_pitch_m},
       {"temporal_filter", s.temporal_filter},
       {"filter_inner", s.filter_inner},
       {"filter_outer", s.filter_outer},
       {"playback", s.playback},
       {"playback_fps", s.playback_fps}};
}
void from_json(const nlohmann::json &j, HoloFocusSettings &s) {
  s                  = {};
  s.frame            = j.value("frame", 0);
  s.method           = j.value("method", std::string("holovibes_stft"));
  s.window           = j.value("window", 512);
  s.playback         = j.value("playback", true);
  s.playback_fps     = j.value("playback_fps", 20);
  s.temporal_focus_m = j.value("temporal_focus_m", 0.002);
  s.temporal_pitch_m = j.value("temporal_pitch_m", 12e-6);
  s.temporal_filter  = j.value("temporal_filter", true);
  s.filter_inner     = j.value("filter_inner", 20);
  s.filter_outer     = j.value("filter_outer", 362);
  if (j.contains("optics"))
    j.at("optics").get_to(s.optics);
}

namespace {
std::optional<double> number(const nlohmann::json &root, const char *pointer, double factor = 1,
                             bool positive = true) {
  const nlohmann::json::json_pointer key(pointer);
  if (!root.contains(key) || !root.at(key).is_number())
    return {};
  const double value = root.at(key).get<double>() * factor;
  return std::isfinite(value) && (!positive || value > 0) ? std::optional(value) : std::nullopt;
}
} // namespace

HoloCalibration inspect_holo_calibration(const std::filesystem::path &path) {
  holofile::Reader reader(path.string());
  const auto      &header = reader.header();
  if ((header.bits_per_pixel != 8 && header.bits_per_pixel != 16) ||
      header.endianness != holofile::Header::LITTLE_ENDIAN || header.frame_width == 0 ||
      header.frame_height == 0 || header.frame_width > 16384 || header.frame_height > 16384 ||
      header.frame_count == 0 ||
      header.frame_count > static_cast<unsigned>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "Focus explorer needs a little-endian U8/U16 hologram recording with valid dimensions.");
  const uint64_t expected = uint64_t(header.frame_width) * header.frame_height *
                            header.frame_count * (header.bits_per_pixel / 8);
  if (expected != header.data_size_in_bytes ||
      std::filesystem::file_size(path) < expected + sizeof(header))
    throw std::invalid_argument("Holofile header and data length disagree.");
  HoloCalibration result{.width  = static_cast<int>(header.frame_width),
                         .height = static_cast<int>(header.frame_height),
                         .frames = static_cast<int>(header.frame_count)};
  result.summary = std::format("{} x {}, {} frames. ", result.width, result.height, result.frames);
  const auto footer = reader.footer();
  if (!footer) {
    result.summary += "No saved optical settings. Enter wavelength, sensor pitches and propagation "
                      "range manually.";
    return result;
  }
  const auto &root = footer->pipeline_settings;
  const auto &settings =
      root.contains("Footer") && root.at("Footer").is_object() ? root.at("Footer") : root;
  result.wavelength_m = number(settings, "/compute_settings/image_rendering/lambda");
  result.propagation_m =
      number(settings, "/compute_settings/image_rendering/propagation_distance", 1, false);
  result.pixel_pitch_x_m = number(settings, "/info/pixel_pitch/x");
  result.pixel_pitch_y_m = number(settings, "/info/pixel_pitch/y");
  // Old files may contain unitless micrometre values; the current writer stores metres.
  // Make that interpretation explicit instead of treating an old value of 20 as 20 metres.
  if (header.version <= 5 && result.pixel_pitch_x_m && result.pixel_pitch_y_m &&
      *result.pixel_pitch_x_m >= 0.001 && *result.pixel_pitch_y_m >= 0.001) {
    *result.pixel_pitch_x_m *= 1e-6;
    *result.pixel_pitch_y_m *= 1e-6;
    result.summary += "Legacy pitch values interpreted as micrometres (unit assumption). ";
  }
  if (result.wavelength_m)
    result.summary += std::format("Saved wavelength {:.6g} nm. ", *result.wavelength_m * 1e9);
  if (result.pixel_pitch_x_m && result.pixel_pitch_y_m)
    result.summary += std::format("Saved sensor pitches {:.6g} / {:.6g} um. ",
                                  *result.pixel_pitch_x_m * 1e6, *result.pixel_pitch_y_m * 1e6);
  if (result.propagation_m)
    result.summary += std::format("Saved propagation {:.6g} mm. ", *result.propagation_m * 1e3);
  if (!result.wavelength_m || !result.pixel_pitch_x_m || !result.pixel_pitch_y_m ||
      !result.propagation_m)
    result.summary += "Some optical values are missing: existing field values are retained; verify "
                      "them manually. ";
  const auto fw = number(root, "/Image Width"), fh = number(root, "/Image Height"),
             fc = number(root, "/Image Number");
  if ((fw && *fw != result.width) || (fh && *fh != result.height) || (fc && *fc != result.frames))
    result.summary += "Footer dimensions/frame count disagree with the binary header. Cropping "
                      "versus resampling is unknown; verify sensor pitches. ";
  result.summary += "Saved optics are unverified estimates. Z is sensor propagation distance, not "
                    "anatomical depth. No object-space magnification is known.";
  return result;
}

void validate_holo_focus_settings(const HoloFocusSettings &settings, int width, int height,
                                  int frames) {
  if (settings.playback_fps < 1 || settings.playback_fps > 240)
    throw std::invalid_argument("Playback FPS limit must be between 1 and 240");
  if (width <= 0 || height <= 0 || frames <= 0)
    throw std::invalid_argument("Invalid .holo dimensions");
  if (settings.frame < 0 || settings.frame >= frames)
    throw std::invalid_argument("Focus frame index is outside the recording.");
  if (!std::isfinite(settings.optics.wavelength_m) || settings.optics.wavelength_m <= 0 ||
      !std::isfinite(static_cast<float>(settings.optics.wavelength_m)))
    throw std::invalid_argument("Wavelength must be positive, finite and representable as F32");
  const bool temporal = settings.method == "holovibes_stft";
  if (settings.method != "propagation" && !temporal)
    throw std::invalid_argument("Unknown .holo volume method");
  if (temporal && (width % 2 != 0 || height % 2 != 0))
    throw std::invalid_argument("Holovibes STFT requires even image width and height for the "
                                "angular-spectrum frequency shift");
  if (temporal &&
      (settings.window < 2 || settings.window > 2048 || settings.window > frames - settings.frame ||
       !std::isfinite(settings.temporal_focus_m) || !std::isfinite(settings.temporal_pitch_m) ||
       settings.temporal_pitch_m <= 0 ||
       !std::isfinite(static_cast<float>(settings.temporal_pitch_m)) ||
       !std::isfinite(static_cast<float>(settings.temporal_focus_m)) || settings.filter_inner < 0 ||
       settings.filter_outer <= settings.filter_inner ||
       static_cast<uint64_t>(settings.window) * width * height * 8 >
           uint64_t{4} * 1024 * 1024 * 1024))
    throw std::invalid_argument(
        "Invalid STFT window, propagation, sensor pitch, filter or volume size (4 GiB limit)");
  if (!temporal) {
    const holoflow::core::TDesc desc({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     holoflow::core::DType::U8, holoflow::core::MemLoc::Host);
    (void)syncs::FocusStackFactory().infer(std::span(&desc, 1), settings.optics);
  }
}

holoflow::core::GraphSpec build_focus_stack_pipeline(const std::filesystem::path &path,
                                                     const HoloFocusSettings     &settings) {
  const auto calibration = inspect_holo_calibration(path);
  validate_holo_focus_settings(settings, calibration.width, calibration.height, calibration.frames);
  const bool                temporal = settings.method == "holovibes_stft";
  holoflow::core::GraphSpec graph;
  const auto                source = boost::add_vertex(
      holoflow::core::NodeSpec{
          .name = "focus_frame",
          .kind = "Holofile",
          .settings =
              sources::HolofileSettings{.path        = path.string(),
                                        .load_kind   = sources::HolofileSettings::LoadKind::Live,
                                        .start_frame = settings.frame,
                                        .end_frame =
                                            settings.frame + (temporal ? settings.window : 1),
                                        .batch_size  = temporal ? settings.window : 1,
                                        .keep_cursor = false,
                                        .repeat      = false}},
      graph);
  auto node = [&graph](const char *name, const char *kind, const nlohmann::json &config) {
    return boost::add_vertex(
        holoflow::core::NodeSpec{.name = name, .kind = kind, .settings = config}, graph);
  };
  auto edge = [&graph](auto from, auto to) {
    boost::add_edge(from, to, holoflow::core::EdgeSpec{0, 0}, graph);
  };
  if (temporal) {
    const auto device =
        node("holo_device", "Memcpy", syncs::MemcpySettings{syncs::MemcpySettings::Target::Device});
    const auto real = node("holo_float", "Conversion",
                           syncs::ConversionSettings{syncs::ConversionSettings::Target::F32,
                                                     syncs::ConversionSettings::Strategy::Real});
    syncs::AngularSpectrumSettings propagation{.lambda =
                                                   static_cast<float>(settings.optics.wavelength_m),
                                               .dx = static_cast<float>(settings.temporal_pitch_m),
                                               .dy = static_cast<float>(settings.temporal_pitch_m),
                                               .z  = static_cast<float>(settings.temporal_focus_m),
                                               .normalize = true};
    if (settings.temporal_filter)
      propagation.filter = syncs::AngularSpectrumSettings::Filter{settings.filter_inner,
                                                                  settings.filter_outer, 0, 0};
    const auto spatial  = node("holo_angular_spectrum", "AngularSpectrum", propagation);
    const auto spectrum = node("holo_temporal_fft", "TemporalFFT", nlohmann::json::object());
    const auto magnitude =
        node("holo_temporal_volume", "Conversion",
             syncs::ConversionSettings{syncs::ConversionSettings::Target::F32,
                                       syncs::ConversionSettings::Strategy::Modulus});
    edge(source, device);
    edge(device, real);
    edge(real, spatial);
    edge(spatial, spectrum);
    edge(spectrum, magnitude);
  } else {
    const auto focus = node("focus_stack", "FocusStack", settings.optics);
    edge(source, focus);
  }
  return graph;
}
void register_focus_stack_tasks(holoflow::core::Registry &registry) {
  if (!registry.is_registered("Memcpy"))
    registry.register_sync("Memcpy", std::make_unique<syncs::MemcpyFactory>());
  if (!registry.is_registered("Conversion"))
    registry.register_sync("Conversion", std::make_unique<syncs::ConversionFactory>());
  if (!registry.is_registered("AngularSpectrum"))
    registry.register_sync("AngularSpectrum", std::make_unique<syncs::AngularSpectrumFactory>());
  if (!registry.is_registered("TemporalFFT"))
    registry.register_sync("TemporalFFT", std::make_unique<syncs::TemporalFFTFactory>());
  if (!registry.is_registered("Holofile"))
    registry.register_sync("Holofile", std::make_unique<sources::HolofileFactory>());
  if (!registry.is_registered("FocusStack"))
    registry.register_sync("FocusStack", std::make_unique<syncs::FocusStackFactory>());
}
} // namespace holotask::pipelines
