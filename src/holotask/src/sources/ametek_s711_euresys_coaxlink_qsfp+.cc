// Copyright 2025 Digital Holography Foundation
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

#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"

#include <format>
#include <functional>
#include <optional>
#include <utility>

#include <spdlog/fmt/fmt.h>

template <>
struct fmt::formatter<holotask::sources::RecordSettings> : fmt::formatter<std::string_view> {
  auto format(const holotask::sources::RecordSettings &r, fmt::format_context &ctx) const {
    return fmt::format_to(ctx.out(), "{{ file_path: {}, recording_count: {} }}", r.file_path,
                          r.recording_count);
  }
};

template <typename T> struct fmt::formatter<std::optional<T>> : fmt::formatter<std::string_view> {
  auto format(const std::optional<T> &o, fmt::format_context &ctx) const {
    if (!o.has_value())
      return fmt::format_to(ctx.out(), "None");
    return fmt::format_to(ctx.out(), "{}", *o);
  }
};

namespace holotask::sources {

bool RecordSettings::requires_rebuild(const RecordSettings &old) const {
  return recording_count > old.recording_count;
}
// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const RecordSettings &s) {
  j = nlohmann::json{{"file_path", s.file_path},
                     {"recording_count", s.recording_count},
                     {"pipeline_config", s.pipeline_config},
                     {"record_queue_count", s.record_queue_count}};
}

void from_json(const nlohmann::json &j, RecordSettings &s) {
  j.at("file_path").get_to(s.file_path);
  j.at("recording_count").get_to(s.recording_count);
  j.at("pipeline_config").get_to(s.pipeline_config);
  j.at("record_queue_count").get_to(s.record_queue_count);
}

void to_json(nlohmann::json &j, const AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j = nlohmann::json{{"cfg_path", s.cfg_path}, {"record_settings", nullptr}};
  if (s.record_settings)
    j["record_settings"] = *s.record_settings;
}

void from_json(const nlohmann::json &j, AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j.at("cfg_path").get_to(s.cfg_path);
  s.record_settings.reset();
  if (j.contains("record_settings") && !j.at("record_settings").is_null())
    s.record_settings = j.at("record_settings").get<RecordSettings>();
}

} // namespace holotask::sources

#ifdef HOLOTASK_HAS_EGRABBER

#include <EGrabber.h>
#include <EuresysGenapiErrorFormats.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#undef max
#undef min
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

#include "holofile/holofile.hh"

#include "buffer_queue/buffer_queue.hh"
#include "grabber/grabber.hh"
#include "recorder/recorder.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// Private implementation types
// -------------------------------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

int64_t signed_delta(uint64_t a, uint64_t b) {
  const auto magnitude = a >= b ? a - b : b - a;
  const auto bounded   = static_cast<int64_t>(
      (std::min)(magnitude, static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())));
  return a >= b ? bounded : -bounded;
}

template <typename T> std::string show(const std::optional<T> &value) {
  return value.has_value() ? std::to_string(*value) : "unavailable";
}

template <typename T>
std::string show_change(const std::optional<T> &value, const std::optional<T> &baseline) {
  if (!value || !baseline) {
    return "n/a";
  }
  if (*value < *baseline) {
    return "reset/wrap";
  }
  return std::to_string(*value - *baseline);
}

template <typename T>
std::string show_counter(const std::optional<T> &value, const std::optional<T> &previous,
                         const std::optional<T> &update_baseline) {
  if (!value) {
    return "unavailable";
  }
  return std::format("{} (interval +{}, update +{})", *value, show_change(value, previous),
                     show_change(value, update_baseline));
}

struct BankCounters {
  std::optional<int64_t>  rejected_frames;
  std::optional<int64_t>  broken_frames;
  std::optional<uint64_t> underrun_buffers;
  std::optional<size_t>   queued_buffers;
  std::optional<size_t>   awaiting_buffers;
};

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[AmetekS711EuresysCoaxlinkQSFPFactory] error: {}", msg);
    throw std::invalid_argument("AmetekS711EuresysCoaxlinkQSFPFactory error: " + msg);
  }
}

std::string format_genapi_error(const Euresys::genapi_error &err) {
  std::ostringstream oss;
  oss << "GenApi error: code=" << err.genapi_error_code << ", what=\"" << err.what() << "\"";

  const size_t count = err.parameter_count();
  if (count > 0) {
    oss << ", parameters=[";
    for (size_t i = 0; i < count; ++i) {
      oss << "{";
      switch (err.parameter_type(i)) {
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_STRING:
        oss << "string:" << err.string_parameter(i);
        break;
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_INTEGER:
        oss << "int:" << err.integer_parameter(i);
        break;
      case GenTL::EuresysCustomGenTL::GENAPI_ERROR_PARAMETER_TYPE_FLOAT:
        oss << "float:" << err.float_parameter(i);
        break;
      default:
        oss << "unknown";
        break;
      }
      oss << "}";
      if (i + 1 < count) {
        oss << ", ";
      }
    }
    oss << "]";
  }

  return oss.str();
}

struct RuntimeConfig {
  std::string                camera_model_name;
  std::size_t                expected_grabber_count;
  std::size_t                nb_buffers;
  std::size_t                buffer_part_count;
  std::size_t                final_height;
  std::size_t                width;
  std::string                pixel_format;
  std::size_t                bytes_per_pixel;
  std::string                banks;
  std::string                trigger_source;
  std::string                trigger_mode;
  std::string                trigger_selector;
  std::string                gain_selector;
  float                      gain;
  std::string                flat_field_correction;
  std::optional<std::string> balance_white_marker;
  double                     exposure_time;
  std::int64_t               cycle_minimum_period;
  std::vector<std::size_t>   offsets;
  std::size_t                line_width;
  std::size_t                line_pitch;
  std::size_t                stripe_height;
  std::size_t                stripe_pitch;
  std::size_t                block_height;
  std::string                stripe_arrangement;
  std::size_t                pop_timeout_ms;

  [[nodiscard]] std::size_t camera_height() const {
    if (banks == "Banks_AB") {
      check(final_height % 2 == 0, "final height must be even in Banks_AB mode");
      return final_height / 2;
    }
    return final_height;
  }
};

nlohmann::json load_cfg(const std::string &cfg_path) {
  check(!cfg_path.empty(), "cfg_path is empty");

  std::ifstream cfg_file(cfg_path);
  check(cfg_file.is_open(), std::format("could not open config file: {}", cfg_path));

  auto root = nlohmann::json::parse(cfg_file);
  check(root.contains("s711"), "config file does not contain top-level key 's711'");
  return root.at("s711");
}

RuntimeConfig parse_cfg(const nlohmann::json &cfg) {
  static const std::map<std::string, std::size_t> pixel_format_map = {
      {"Mono8", 1},
      {"Mono16", 2},
  };

  const auto pixel_format = cfg.at("PixelFormat").get<std::string>();
  check(pixel_format_map.contains(pixel_format), "unsupported PixelFormat: " + pixel_format);

  RuntimeConfig out{
      .camera_model_name      = cfg.value("CameraModelName", std::string("Phantom S711")),
      .expected_grabber_count = cfg.value("ExpectedGrabberCount", std::size_t(2)),
      .nb_buffers             = cfg.at("NbBuffers").get<std::size_t>(),
      .buffer_part_count      = cfg.at("BufferPartCount").get<std::size_t>(),
      .final_height           = cfg.value("FinalHeight", cfg.at("Height").get<std::size_t>()),
      .width                  = cfg.at("Width").get<std::size_t>(),
      .pixel_format           = pixel_format,
      .bytes_per_pixel        = pixel_format_map.at(pixel_format),
      .banks                  = cfg.value("Banks", std::string("Banks_AB")),
      .trigger_source         = cfg.at("TriggerSource").get<std::string>(),
      .trigger_mode           = cfg.at("TriggerMode").get<std::string>(),
      .trigger_selector       = cfg.at("TriggerSelector").get<std::string>(),
      .gain_selector          = cfg.at("GainSelector").get<std::string>(),
      .gain                   = cfg.at("Gain").get<float>(),
      .flat_field_correction  = cfg.at("FlatFieldCorrection").get<std::string>(),
      .balance_white_marker =
          cfg.contains("BalanceWhiteMarker")
              ? std::optional<std::string>(cfg.at("BalanceWhiteMarker").get<std::string>())
              : std::nullopt,
      .exposure_time        = cfg.at("ExposureTime").get<double>(),
      .cycle_minimum_period = cfg.at("CycleMinimumPeriod").get<std::int64_t>(),
      .offsets              = cfg.at("Offsets").get<std::vector<std::size_t>>(),
      .line_width           = cfg.value("LineWidth", cfg.at("Width").get<std::size_t>() *
                                                         pixel_format_map.at(pixel_format)),
      .line_pitch           = cfg.value("LinePitch", cfg.at("Width").get<std::size_t>() *
                                                         pixel_format_map.at(pixel_format)),
      .stripe_height        = cfg.value("StripeHeight", std::size_t(8)),
      .stripe_pitch         = cfg.value("StripePitch", std::size_t(16)),
      .block_height         = cfg.value("BlockHeight", std::size_t(8)),
      .stripe_arrangement   = cfg.value("StripeArrangement", std::string("Geometry_1X_2YM")),
      .pop_timeout_ms       = cfg.value("PopTimeoutMs", std::size_t(1000)),
  };

  check(out.expected_grabber_count == 2,
        "only two-grabber S711 Banks_AB acquisition is implemented");
  check(out.offsets.size() == out.expected_grabber_count,
        "Offsets array size must match expected grabber count");
  check(out.banks == "Banks_AB", "only Banks_AB mode is implemented");

  return out;
}

nlohmann::json normalized_cfg_json(const RuntimeConfig &cfg) {
  return nlohmann::json{
      {"CameraModelName", cfg.camera_model_name},
      {"ExpectedGrabberCount", cfg.expected_grabber_count},
      {"NbBuffers", cfg.nb_buffers},
      {"BufferPartCount", cfg.buffer_part_count},
      {"Height", cfg.final_height},
      {"Width", cfg.width},
      {"PixelFormat", cfg.pixel_format},
      {"Banks", cfg.banks},
      {"TriggerSource", cfg.trigger_source},
      {"TriggerMode", cfg.trigger_mode},
      {"TriggerSelector", cfg.trigger_selector},
      {"GainSelector", cfg.gain_selector},
      {"Gain", cfg.gain},
      {"FlatFieldCorrection", cfg.flat_field_correction},
      {"ExposureTime", cfg.exposure_time},
      {"CycleMinimumPeriod", cfg.cycle_minimum_period},
      {"Offsets", cfg.offsets},
      {"LineWidth", cfg.line_width},
      {"LinePitch", cfg.line_pitch},
      {"StripeHeight", cfg.stripe_height},
      {"StripePitch", cfg.stripe_pitch},
      {"BlockHeight", cfg.block_height},
      {"StripeArrangement", cfg.stripe_arrangement},
      {"PopTimeoutMs", cfg.pop_timeout_ms},
  };
}

void dump_cfg(const nlohmann::json &raw_cfg, const RuntimeConfig &cfg) {
  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] loaded config:\n{}", raw_cfg.dump(2));
  logger()->info(
      "[AmetekS711EuresysCoaxlinkQSFPFactory] derived config: banks={}, final_height={}, "
      "camera_height={}, width={}, pixel_format={}, line_width={}, line_pitch={}, "
      "stripe_height={}, stripe_pitch={}, block_height={}, stripe_arrangement={}",
      cfg.banks, cfg.final_height, cfg.camera_height(), cfg.width, cfg.pixel_format, cfg.line_width,
      cfg.line_pitch, cfg.stripe_height, cfg.stripe_pitch, cfg.block_height,
      cfg.stripe_arrangement);
}

std::optional<Euresys::EGrabberCameraInfo> find_camera(Euresys::EGenTL   &gentl,
                                                       const std::string &camera_name) {
  using namespace Euresys;

  EGrabberDiscovery discovery(gentl);
  discovery.discover();

  for (int i = 0; i < discovery.cameraCount(); ++i) {
    auto info = discovery.cameras(i);
    auto g    = EGrabber<>(info);

    try {
      if (g.getString<RemoteModule>("DeviceModelName") == camera_name) {
        return info;
      }
    } catch (const Euresys::genapi_error &) {
      try {
        if (g.getString<DeviceModule>("DeviceModelName") == camera_name) {
          return info;
        }
      } catch (const Euresys::genapi_error &) {
      }
    }
  }

  return std::nullopt;
}

std::size_t find_grabber_index_for_bank(Euresys::EGrabberCameraInfo &info, std::int64_t bank_id) {
  using namespace Euresys;

  for (std::size_t i = 0; i < info.grabbers.size(); ++i) {
    EGrabber<> g(info.grabbers[i]);
    if (g.getInteger<RemoteModule>("ConnectedBankID") == bank_id) {
      return i;
    }
  }

  throw std::runtime_error(std::format("could not find grabber for ConnectedBankID={}", bank_id));
}

template <typename Module>
void dump_string(Euresys::EGrabber<> &g, const std::string &prefix, const char *name) {
  try {
    logger()->info("{} {}={}", prefix, name, g.getString<Module>(std::string(name)));
  } catch (const Euresys::genapi_error &e) {
    logger()->info("{} {}=<unavailable: {}>", prefix, name, format_genapi_error(e));
  }
}

template <typename Module>
void dump_int(Euresys::EGrabber<> &g, const std::string &prefix, const char *name) {
  try {
    logger()->info("{} {}={}", prefix, name, g.getInteger<Module>(std::string(name)));
  } catch (const Euresys::genapi_error &e) {
    logger()->info("{} {}=<unavailable: {}>", prefix, name, format_genapi_error(e));
  }
}

template <typename Module>
void dump_float(Euresys::EGrabber<> &g, const std::string &prefix, const char *name) {
  try {
    logger()->info("{} {}={}", prefix, name, g.getFloat<Module>(std::string(name)));
  } catch (const Euresys::genapi_error &e) {
    logger()->info("{} {}=<unavailable: {}>", prefix, name, format_genapi_error(e));
  }
}

void dump_state(Euresys::EGrabberCameraInfo &info, const std::string &phase) {
  using namespace Euresys;

  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] ===== {} =====", phase);

  for (std::size_t i = 0; i < info.grabbers.size(); ++i) {
    EGrabber<> g(info.grabbers[i]);
    const auto prefix =
        std::format("[AmetekS711EuresysCoaxlinkQSFPFactory][{}] grabber[{}]", phase, i);

    dump_int<RemoteModule>(g, prefix, "ConnectedBankID");
    dump_string<RemoteModule>(g, prefix, "Banks");
    dump_int<RemoteModule>(g, prefix, "Width");
    dump_int<RemoteModule>(g, prefix, "Height");
    dump_string<RemoteModule>(g, prefix, "PixelFormat");
    dump_string<RemoteModule>(g, prefix, "TriggerMode");
    dump_string<RemoteModule>(g, prefix, "TriggerSource");
    dump_string<RemoteModule>(g, prefix, "TriggerSelector");
    dump_float<RemoteModule>(g, prefix, "ExposureTime");
    dump_string<RemoteModule>(g, prefix, "GainSelector");
    dump_float<RemoteModule>(g, prefix, "Gain");
    dump_string<RemoteModule>(g, prefix, "FlatFieldCorrection");

    dump_string<DeviceModule>(g, prefix, "CameraControlMethod");
    dump_string<DeviceModule>(g, prefix, "ExposureReadoutOverlap");
    dump_string<DeviceModule>(g, prefix, "ErrorSelector");
    dump_int<DeviceModule>(g, prefix, "CycleMinimumPeriod");

    dump_int<StreamModule>(g, prefix, "BufferPartCount");
    dump_int<StreamModule>(g, prefix, "LineWidth");
    dump_int<StreamModule>(g, prefix, "LinePitch");
    dump_int<StreamModule>(g, prefix, "StripeHeight");
    dump_int<StreamModule>(g, prefix, "StripePitch");
    dump_int<StreamModule>(g, prefix, "BlockHeight");
    dump_int<StreamModule>(g, prefix, "StripeOffset");
    dump_string<StreamModule>(g, prefix, "StripeArrangement");
  }
}

template <typename Module>
void set_required_string(Euresys::EGrabber<> &g, const std::string &prefix, const char *name,
                         const std::string &value) {
  try {
    g.setString<Module>(std::string(name), value);
    logger()->info("{} set {}={}", prefix, name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("{} failed to set {}={}: {}", prefix, name, value, format_genapi_error(e)));
  }
}

template <typename Module>
void set_required_int(Euresys::EGrabber<> &g, const std::string &prefix, const char *name,
                      std::int64_t value) {
  try {
    g.setInteger<Module>(std::string(name), value);
    logger()->info("{} set {}={}", prefix, name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("{} failed to set {}={}: {}", prefix, name, value, format_genapi_error(e)));
  }
}

template <typename Module>
void set_required_float(Euresys::EGrabber<> &g, const std::string &prefix, const char *name,
                        double value) {
  try {
    g.setFloat<Module>(std::string(name), value);
    logger()->info("{} set {}={}", prefix, name, value);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(
        std::format("{} failed to set {}={}: {}", prefix, name, value, format_genapi_error(e)));
  }
}

template <typename Module>
void set_optional_string(Euresys::EGrabber<> &g, const std::string &prefix, const char *name,
                         const std::string &value) {
  try {
    g.setString<Module>(std::string(name), value);
    logger()->info("{} set {}={}", prefix, name, value);
  } catch (const Euresys::genapi_error &e) {
    logger()->warn("{} could not set {}={}: {}", prefix, name, value, format_genapi_error(e));
  }
}

void apply_cfg(Euresys::EGrabberCameraInfo &info, const RuntimeConfig &cfg) {
  using namespace Euresys;

  check(info.grabbers.size() == cfg.expected_grabber_count,
        std::format("expected {} grabber(s), got {}", cfg.expected_grabber_count,
                    info.grabbers.size()));

  const auto bank_a_index = find_grabber_index_for_bank(info, 0);
  const auto bank_b_index = find_grabber_index_for_bank(info, 1);

  EGrabber<> ctrl(info.grabbers[bank_a_index]);
  const auto ctrl_prefix = "[AmetekS711EuresysCoaxlinkQSFPFactory][apply][bankA-control]";

  // Camera-side settings are shared across both banks and must be written through bank A.
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "Banks", cfg.banks);
  set_required_int<RemoteModule>(ctrl, ctrl_prefix, "Width", static_cast<std::int64_t>(cfg.width));
  set_required_int<RemoteModule>(ctrl, ctrl_prefix, "Height",
                                 static_cast<std::int64_t>(cfg.camera_height()));
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "PixelFormat", cfg.pixel_format);
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "TriggerSelector", cfg.trigger_selector);
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "TriggerMode", cfg.trigger_mode);
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "TriggerSource", cfg.trigger_source);
  set_required_float<RemoteModule>(ctrl, ctrl_prefix, "ExposureTime", cfg.exposure_time);
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "GainSelector", cfg.gain_selector);
  set_required_float<RemoteModule>(ctrl, ctrl_prefix, "Gain", cfg.gain);
  set_required_string<RemoteModule>(ctrl, ctrl_prefix, "FlatFieldCorrection",
                                    cfg.flat_field_correction);

  if (cfg.balance_white_marker.has_value()) {
    set_optional_string<RemoteModule>(ctrl, ctrl_prefix, "BalanceWhiteMarker",
                                      *cfg.balance_white_marker);
  }

  const auto camera_control_method =
      cfg.trigger_source == "SWTRIGGER" ? std::string("RC") : std::string("EXTERNAL");
  set_optional_string<DeviceModule>(ctrl, ctrl_prefix, "CameraControlMethod",
                                    camera_control_method);

  if (cfg.trigger_source == "SWTRIGGER") {
    set_optional_string<DeviceModule>(ctrl, ctrl_prefix, "ErrorSelector", "All");
    set_optional_string<DeviceModule>(ctrl, ctrl_prefix, "ExposureReadoutOverlap", "True");
    try {
      ctrl.setInteger<DeviceModule>("CycleMinimumPeriod", cfg.cycle_minimum_period);
      logger()->info("{} set CycleMinimumPeriod={}", ctrl_prefix, cfg.cycle_minimum_period);
    } catch (const Euresys::genapi_error &e) {
      logger()->warn("{} could not set CycleMinimumPeriod={}: {}", ctrl_prefix,
                     cfg.cycle_minimum_period, format_genapi_error(e));
    }
  }

  auto apply_stream = [&](std::size_t grabber_index, std::size_t stripe_offset) {
    EGrabber<> g(info.grabbers[grabber_index]);
    const auto prefix =
        std::format("[AmetekS711EuresysCoaxlinkQSFPFactory][apply] grabber[{}]", grabber_index);

    set_required_int<StreamModule>(g, prefix, "BufferPartCount",
                                   static_cast<std::int64_t>(cfg.buffer_part_count));
    set_required_int<StreamModule>(g, prefix, "LineWidth",
                                   static_cast<std::int64_t>(cfg.line_width));
    set_required_int<StreamModule>(g, prefix, "LinePitch",
                                   static_cast<std::int64_t>(cfg.line_pitch));
    set_required_int<StreamModule>(g, prefix, "StripeHeight",
                                   static_cast<std::int64_t>(cfg.stripe_height));
    set_required_int<StreamModule>(g, prefix, "StripePitch",
                                   static_cast<std::int64_t>(cfg.stripe_pitch));
    set_required_int<StreamModule>(g, prefix, "BlockHeight",
                                   static_cast<std::int64_t>(cfg.block_height));
    set_required_int<StreamModule>(g, prefix, "StripeOffset",
                                   static_cast<std::int64_t>(stripe_offset));
    set_required_string<StreamModule>(g, prefix, "StripeArrangement", cfg.stripe_arrangement);
  };

  apply_stream(bank_a_index, cfg.offsets[0]);
  apply_stream(bank_b_index, cfg.offsets[1]);
}

/**
 * Allocate a host-resident buffer pool and announce the exact same buffer slots
 * to both banks.
 *
 * The stream module writes each bank into different stripes of the same logical
 * final frame because the stream geometry has already been configured with the
 * appropriate StripeOffset / StripePitch / StripeArrangement values.
 */
HostPtr<uint8_t> allocate_shared_buffers(Grabber &grabber_a, Grabber &grabber_b,
                                         std::size_t nb_buffers, std::size_t buffer_size) {
  constexpr std::size_t safety_padding_size = 16;
  const std::size_t     actual_nb_buffers   = nb_buffers + safety_padding_size;

  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] allocating {} ({} + {}) shared "
                 "host buffers of size {} bytes",
                 actual_nb_buffers, nb_buffers, safety_padding_size, buffer_size);

  const auto total_size = buffer_size * actual_nb_buffers;

  auto buffers = curaii::make_unique_host_ptr<uint8_t>(total_size);

  Euresys::UserMemory memory(buffers.get(), total_size);

  Euresys::UserMemoryArray memory_array(memory, buffer_size);

  using Clock = std::chrono::steady_clock;

  const auto start = Clock::now();
  // --------------------------------------------------------------------------
  // Grabber A
  // --------------------------------------------------------------------------
  {
    const auto start = Clock::now();

    const auto range_a = grabber_a.announceAndQueue(memory_array);

    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    logger()->info("[allocate_shared_buffers] grabber A: "
                   "announceAndQueue {} buffers took {:.3f} ms "
                   "(range {}..{}, {} buffers)",
                   actual_nb_buffers, elapsed, range_a.begin, range_a.end, range_a.size());
  }

  // --------------------------------------------------------------------------
  // Grabber B
  // --------------------------------------------------------------------------
  {
    const auto start = Clock::now();

    const auto range_b = grabber_b.announceAndQueue(memory_array);

    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    logger()->info("[allocate_shared_buffers] grabber B: "
                   "announceAndQueue {} buffers took {:.3f} ms "
                   "(range {}..{}, {} buffers)",
                   actual_nb_buffers, elapsed, range_b.begin, range_b.end, range_b.size());
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  logger()->info("[allocate_shared_buffers] announcing buffers took {:.3f} ms ", elapsed);

  return buffers;
}

holoflow::core::DType dtype_from_pixel_format(const std::string &pixel_format) {
  static const std::map<std::string, holoflow::core::DType> dtypes = {
      {"Mono8", holoflow::core::DType::U8},
      {"Mono16", holoflow::core::DType::U16},
  };

  check(dtypes.contains(pixel_format), "unsupported PixelFormat: " + pixel_format);
  return dtypes.at(pixel_format);
}

size_t size_from_pixel_format(const std::string &pixel_format) {
  static const std::map<std::string, size_t> sizes = {
      {"Mono8", sizeof(uint8_t)},
      {"Mono16", sizeof(uint16_t)},
  };

  check(sizes.contains(pixel_format), "unsupported PixelFormat: " + pixel_format);
  return sizes.at(pixel_format);
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Task implementation (private)
// -------------------------------------------------------------------------------------------------

/**
 * Two-bank S711 source task.
 *
 * This task exposes a host tensor. The configured stream geometry makes both
 * grabbers DMA into different stripes of the same final frame buffer. Each
 * logical output frame therefore corresponds to one queued buffer slot that is
 * announced to both bank A and bank B.
 */
class AmetekS711EuresysCoaxlinkQSFP : public holoflow::core::ISyncTask {
public:
  AmetekS711EuresysCoaxlinkQSFP(const AmetekS711EuresysCoaxlinkQSFPSettings &settings,
                                RuntimeConfig runtime_cfg, HostPtr<uint8_t> &&buffers,
                                std::size_t buffer_count, std::unique_ptr<Euresys::EGenTL> &&gentl,
                                std::unique_ptr<Grabber> &&grabber_a,
                                std::unique_ptr<Grabber> &&grabber_b, std::size_t buffer_size,
                                nlohmann::json normalized_cfg)
      : settings_(settings), runtime_cfg_(std::move(runtime_cfg)), buffers_(std::move(buffers)),
        gentl_(std::move(gentl)), grabber_a_(std::move(grabber_a)),
        grabber_b_(std::move(grabber_b)), buffer_size_(buffer_size), running_(false),
        cfg_(std::move(normalized_cfg)),
        // if record is enabled, it allocates enough buffers for it
        buffer_queue_(buffer_count, [this](const CameraFrame &frame) { requeue_frame(frame); }) {
    HOLOVIBES_CHECK(gentl_ != nullptr);
    HOLOVIBES_CHECK(grabber_a_ != nullptr);
    HOLOVIBES_CHECK(grabber_b_ != nullptr);
    HOLOVIBES_CHECK(buffers_ != nullptr);
  }

  ~AmetekS711EuresysCoaxlinkQSFP() override {
    if (acquisition_thread_) {
      acquisition_thread_->request_stop();
      acquisition_thread_->join();
      acquisition_thread_.reset();
    }
    stop_raw_record();
    buffer_queue_.close();
    buffer_queue_.clear();
    stop_acquisition();
  }

  void emit_camera_record_flushing_finished_event(holoflow::core::SyncCtx    &ctx,
                                                  RecordingSession::SessionId id) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "record_flushing_finished"},
                {"id", id},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit record_flushing_finished event");
  }

  void emit_recording_finished_event(holoflow::core::SyncCtx &ctx, RecordingSession::SessionId id,
                                     size_t frames_written) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_finished"},
                {"id", id},
                {"frames_written", frames_written},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_finished event");
  }

  void emit_recording_update_event(holoflow::core::SyncCtx &ctx, RecordingSession::SessionId id,
                                   size_t frames_written, size_t writing_speed,
                                   size_t elapsed_time) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_update"},
                {"id", id},
                {"frame_recorded", frames_written},
                {"writing_speed", writing_speed},
                {"elapsed_time", elapsed_time},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_update event");
  }

  void emit_acquisition_update_event(holoflow::core::SyncCtx &ctx, RecordingSession::SessionId id,
                                     size_t frames) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "acquisition_update"},
                {"id", id},
                {"frame_acquired", frames},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit acquisition_update event");
  }

  void emit_recording_failed_event(holoflow::core::SyncCtx &ctx, RecordingSession::SessionId id,
                                   const std::string &message) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_failed"},
                {"id", id},
                {"message", message},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_failed event");
  }

  void start_raw_record(holoflow::core::SyncCtx &ctx) {
    HOLOVIBES_CHECK(ctx.event_writer != nullptr, "Camera recording requires an event writer");
    // Join a completed recorder before replacing its thread and settings.
    assert(settings_.record_settings.has_value());
    assert(recording_context_.has_value());

    if (session_still_recording()) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP:start_raw_record] a session is still "
                      "recording frames from camera");
      emit_recording_failed_event(ctx, current_recording_session_->id(),
                                  "a record is still recording camera frames");
      return;
    }

    const auto record_settings = *settings_.record_settings;
    auto       session_it      = find_next_free_session();
    if (session_it == recording_context_->sessions_.end()) {
      logger()->error(
          "[AmetekS711EuresysCoaxlinkQSFP:start_raw_record] cannot find a free recording session");
      emit_recording_failed_event(ctx, 0, "no queue available for record");
      return;
    }

    auto &session              = *session_it;
    current_recording_session_ = &session;
    session.start(record_settings, runtime_cfg_.buffer_part_count, recording_context_->recording_,
                  recording_context_->acquired_since_record_start_);

    logger()->info("[AmetekS711EuresysCoaxlinkQSFP:start_raw_record] started raw record");
  }

  void stop_raw_record() {
    auto session = current_recording_session_;
    if (session) {
      session->stop();
      current_recording_session_ = nullptr;
    }

    assert(!recording_context_->recording_.load(std::memory_order_acquire));
  }

  void update_settings(const AmetekS711EuresysCoaxlinkQSFPSettings &settings) {
    stop_raw_record();
    settings_ = settings;
  }

  void handle_events(holoflow::core::SyncCtx &ctx) {
    if (!ctx.event_reader) {
      if (log_due(last_event_log_))
        logger()->warn(
            "[AmetekS711EuresysCoaxlinkQSFP::handle_events] the given ctx has no event receiver");
      return;
    }

    using EventHandler = std::function<void(holoflow::core::SyncCtx &,
                                            const std::optional<holoflow_event::Event> &)>;
    static std::map<std::string, EventHandler> event_handlers = {
        {"start_recording",
         [this](holoflow::core::SyncCtx &ctx, const std::optional<holoflow_event::Event> &event) {
           on_start_recording_event(ctx, event);
         }},
        {"stop_recording",
         [this](holoflow::core::SyncCtx &ctx, const std::optional<holoflow_event::Event> &event) {
           on_stop_recording_event(ctx, event);
         }},
        {"update_recording_path",
         [this](holoflow::core::SyncCtx &ctx, const std::optional<holoflow_event::Event> &event) {
           on_update_record_path(ctx, event);
         }},
    };

    while (true) {
      auto event = ctx.event_reader->try_pop();
      if (!event.has_value())
        break;

      HOLOVIBES_CHECK(event->direction == holoflow_event::EventDirection::ToNode,
                      "Unexpected event direction");

      const auto type = event->data.at("type").get<std::string>();

      logger()->debug("[AmetekS711EuresysCoaxlinkQSFP::handle_events] received event: {}", type);

      auto handler_it = event_handlers.find(type);
      if (handler_it == event_handlers.end()) {
        HOLOVIBES_BUG("Unknown event type: {}", type);
      } else {
        handler_it->second(ctx, event);
      }
    }
  }

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    (void)index;
    throw std::out_of_range("AmetekS711EuresysCoaxlinkQSFP task has no inputs");
  }

  /**
   * Re-queue both bank buffers for the frame currently exposed as output 0.
   */
  void release_output(int index) override {
    if (index != 0) {
      throw std::out_of_range("AmetekS711EuresysCoaxlinkQSFP task has only one output at index 0");
    }

    buffer_queue_.release_a();
    storage_access().owned_output_storage(0).ptr = nullptr;
  }

  void log_update_lifecycle(bool replacing) {
    const std::lock_guard lock(diagnostics_mutex_);
    if (!buffer_queue_.empty() && log_due(last_pending_update_log_)) {
      logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::log_update_lifecycle] updating with "
                     "unreleased frames: {}",
                     buffer_queue_.size());
    }
    if (!running_.load(std::memory_order_acquire)) {
      return;
    }
    if (!replacing) {
      ++update_epoch_;
      update_a_ = read_bank_counters(*grabber_a_, "A"); // add mutex on these to prevent data race
      update_b_ = read_bank_counters(*grabber_b_, "B");
      first_pair_after_update_ = true;
      first_pair_update_summary_.reset();
      resume_counters_pending_ = false;
    }

    if (!log_due(last_lifecycle_log_)) {
      return;
    }

    if (replacing && running_.load(std::memory_order_acquire)) {
      logger()->warn(
          "[AmetekS711EuresysCoaxlinkQSFP::log_update_lifecycle] replacing a task while its "
          "grabbers are still acquiring");
    }
  }

  template <typename T, typename Reader>
  std::optional<T> read_diagnostic(Reader &&read, const char *label) {
    try {
      return read();
    } catch (const std::exception &e) {
      if (log_due(last_diagnostic_error_log_)) {
        logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::diagnostics] {} unavailable: {}", label,
                       e.what());
      }
      return std::nullopt;
    }
  }

  BankCounters read_bank_counters(Grabber &grabber, const char *bank) {
    const auto rejected = std::format("bank {} RejectedFrame", bank);
    const auto broken   = std::format("bank {} BrokenFrame", bank);
    const auto underrun = std::format("bank {} buffer underruns", bank);
    const auto queued   = std::format("bank {} queued buffers", bank);
    const auto awaiting = std::format("bank {} awaiting buffers", bank);
    return {
        .rejected_frames = read_diagnostic<int64_t>(
            [&] { return grabber.getInteger<Euresys::StreamModule>("EventCount[RejectedFrame]"); },
            rejected.c_str()),
        .broken_frames = read_diagnostic<int64_t>(
            [&] { return grabber.getInteger<Euresys::StreamModule>("EventCount[BrokenFrame]"); },
            broken.c_str()),
        .underrun_buffers = read_diagnostic<uint64_t>(
            [&] {
              return grabber.getInfo<Euresys::StreamModule, uint64_t>(
                  GenTL::STREAM_INFO_NUM_UNDERRUN);
            },
            underrun.c_str()),
        .queued_buffers = read_diagnostic<size_t>(
            [&] {
              return grabber.getInfo<Euresys::StreamModule, size_t>(GenTL::STREAM_INFO_NUM_QUEUED);
            },
            queued.c_str()),
        .awaiting_buffers = read_diagnostic<size_t>(
            [&] {
              return grabber.getInfo<Euresys::StreamModule, size_t>(
                  GenTL::STREAM_INFO_NUM_AWAIT_DELIVERY);
            },
            awaiting.c_str()),
    };
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (settings_.record_settings.has_value())
      handle_events(ctx);

    if (!running_.load(std::memory_order_acquire)) {
      init_record_context(ctx);
      start_acquisition(ctx);
    }

    while (!ctx.cancelled->load()) {
      const auto *next = buffer_queue_.read_a(ctx.cancelled); // return null if cancelled
      if (next == nullptr) {
        break;
      }
      if (next->base) {
        auto &storage = storage_access().owned_output_storage(0);
        storage.ptr   = next->base;

        ctx.outputs[0] = holoflow::core::TView{
            .desc    = ctx.outputs[0].desc,
            .storage = &storage,
        };
      }
      return holoflow::core::OpResult::Ok;
    }

    logger()->info("[AmetekS711EuresysCoaxlinkQSFP::execute] cancelled");
    stop_raw_record();
    running_.store(false, std::memory_order_release);
    stop_acquisition();
    buffer_queue_.clear();
    return holoflow::core::OpResult::Cancelled;
  }

  const nlohmann::json &get_cfg() const { return cfg_; }

  const AmetekS711EuresysCoaxlinkQSFPSettings &get_settings() const { return settings_; }

  void stop_acquisition() {
    if (!acquisition_thread_.has_value()) {
      logger()->warn(
          "[AmetekS711EuresysCoaxlinkQSFP::stop_acquisition] acquisition is not running");
    } else {
      acquisition_thread_->request_stop();
      acquisition_thread_->join();
      acquisition_thread_.reset();
    }

    grabber_a_->stop_if_running();
    grabber_b_->stop_if_running();
    running_.store(false);
  }

private:
  std::vector<RecordingSession>::iterator find_next_free_session() {
    assert(recording_context_);

    return std::ranges::find_if(recording_context_->sessions_, [](const auto &s) {
      return s.state() == RecordingSession::State::Free;
    });
  }

  bool session_still_recording() {
    assert(recording_context_);

    return std::ranges::any_of(recording_context_->sessions_, [](const auto &s) {
      return s.state() == RecordingSession::State::Recording;
    });
  }

  void init_record_context(holoflow::core::SyncCtx &ctx) {
    if (settings_.record_settings) {
      auto &record_settings = *settings_.record_settings;
      recording_context_.emplace();
      const auto pixel_size_in_bytes = size_from_pixel_format(runtime_cfg_.pixel_format);
      size_t     rec_count           = record_settings.recording_count;
      if (rec_count % runtime_cfg_.buffer_part_count != 0) {
        rec_count += runtime_cfg_.buffer_part_count - (rec_count % runtime_cfg_.buffer_part_count);
      }

      auto recording_buffer_size =
          runtime_cfg_.final_height * runtime_cfg_.width * rec_count * pixel_size_in_bytes;

      auto g = RecordingGeometry(static_cast<uint8_t>(pixel_size_in_bytes * 8), runtime_cfg_.width,
                                 runtime_cfg_.final_height);

      auto buffer_total_size = record_settings.record_queue_count * recording_buffer_size;
      if (buffer_total_size > recording_buffers_size_ * recording_buffers_count_) {
        logger()->info(
            "[init_record_context] allocating {} x {} bytes"
            "buffers (total={} bytes)",
            record_settings.record_queue_count, recording_buffer_size, buffer_total_size);
        recording_buffers_ = curaii::make_unique_host_ptr<std::byte>(buffer_total_size);
      } else {
        logger()->info(
            "[init_record_context] reusing old {} x {} bytes"
            "buffers (total={} bytes) as {} x {} bytes (total={} bytes)",
            recording_buffers_count_, recording_buffers_size_,
            recording_buffers_count_ * recording_buffers_size_, record_settings.record_queue_count,
            recording_buffer_size, buffer_total_size);
      }
      recording_buffers_size_  = recording_buffer_size;
      recording_buffers_count_ = record_settings.record_queue_count;

      recording_context_->invalidate(
          record_settings.record_queue_count, buffer_queue_, recording_buffers_size_,
          recording_buffers_.get(), g,
          [this, &ctx](auto id, size_t recorded_frames, size_t writing_speed, size_t elapsed_time) {
            emit_recording_update_event(ctx, id, recorded_frames, writing_speed, elapsed_time);
          },
          [this, &ctx](auto id, size_t written) {
            emit_recording_finished_event(ctx, id, written);
            logger()->info("[AmetekS711EuresysCoaxlinkQSFP] recording {} finished", id);
            buffer_queue_.unsubscribe_b();
            emit_acquisition_update_event(ctx, id, settings_.record_settings->recording_count);
            current_recording_session_ = nullptr;
          },
          [this, &ctx](auto id, const std::string &message) {
            emit_recording_failed_event(ctx, id, message);
            logger()->info("[AmetekS711EuresysCoaxlinkQSFP::start_raw_record] recording {} failed",
                           id);
            current_recording_session_ = nullptr;
          },
          [this, &ctx, &record_settings](auto id) {
            emit_camera_record_flushing_finished_event(ctx, id);
            logger()->info(
                "[AmetekS711EuresysCoaxlinkQSFP::start_raw_record] recording {} flushing finished",
                id);
          });
    } else
      recording_context_ = std::nullopt;
  }

  void start_acquisition(holoflow::core::SyncCtx &ctx) {
    try {
      // S711 Banks_AB must start bank B first, then bank A.
      grabber_b_->start_if_not_running();
      grabber_a_->start_if_not_running();
      acquisition_thread_.emplace(
          [this, &ctx](std::stop_token cancelled) { acquisition_loop(cancelled, ctx); });
      running_.store(true);
    } catch (...) {
      grabber_a_->stop_if_running();
      grabber_b_->stop_if_running();
      fail_recording(ctx, "Camera acquisition failed to start");
      buffer_queue_.close();
      stop_raw_record();
      throw;
    }
  }

  void requeue_frame(const CameraFrame &frame) {
    grabber_a_->requeue_buffer(frame.bank_a);
    grabber_b_->requeue_buffer(frame.bank_b);
  }

  void fail_recording(holoflow::core::SyncCtx &ctx, const std::string &message) {
    assert(recording_context_.has_value());
    auto session = current_recording_session_;
    if (session) {
      emit_recording_failed_event(ctx, current_recording_session_->id(), message);
      session = nullptr;
    }
  }

  bool check_for_record_failure(holoflow::core::SyncCtx &ctx) {
    bool               recording_failed = false;
    static const char *overflow_message =
        "Camera buffer queue is full; recording cancelled because a frame was dropped";
    if (recording_context_->recording_) {
      auto rem = current_recording_session_->remaining_to_record() / runtime_cfg_.buffer_part_count;
      auto recording_fail = rem > buffer_queue_.size();

      if (recording_fail) {
        emit_recording_failed_event(
            ctx, current_recording_session_->id(),
            std::format("{} (buffer queue size: {}, remaining to write: {})", overflow_message, rem,
                        buffer_queue_.size()));
      }
    }

    return recording_failed;
  }

  void update_acquisition_status(holoflow::core::SyncCtx &ctx,
                                 Clock::time_point       &last_update_event) {
    constexpr auto update_event_interval = std::chrono::milliseconds(200);

    if (!recording_context_.has_value())
      return;

    auto recording = recording_context_->recording_.load();
    if (recording) {
      recording_context_->acquired_since_record_start_.fetch_add(runtime_cfg_.buffer_part_count);
      auto now     = Clock::now();
      auto elapsed = now - last_update_event;
      assert(current_recording_session_ != nullptr);
      auto id = current_recording_session_->id();
      if (elapsed >= update_event_interval) {
        emit_acquisition_update_event(
            ctx, id,
            recording_context_->acquired_since_record_start_.load(std::memory_order_acquire));
        last_update_event = now;
      }
    }
  }

  void acquisition_loop(std::stop_token cancelled, holoflow::core::SyncCtx &ctx) {
    CameraBufferPair  pending{*grabber_a_.get(), *grabber_b_.get()};
    Clock::time_point last_update_event;
    try {
      while (!cancelled.stop_requested()) {
        try {
          pending.poll(100);
        } catch (const Euresys::gentl_error &e) {
          if (e.gc_err == GenTL::GC_ERR_TIMEOUT)
            continue;
          throw;
        }
        if (cancelled.stop_requested())
          break;

        auto base = validate_buffer_data(pending.a(), pending.b());
        if (!base.has_value()) {
          pending.release();
          if (check_for_record_failure(ctx)) {
            logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] frame not validated "
                            "while recording");
          }
          continue;
        }

        bool                          recording_failed = false;
        auto                          buffer           = Euresys::Buffer(pending.a());
        CameraFrame                   frame{pending.a(), pending.b(), *base};
        CameraBufferQueue::PushResult result;
        static const char            *overflow_message =
            "Camera buffer queue is full; recording cancelled because a frame was dropped";

        {
          // Attribute overflow to the session active at submission, and claim
          // failure before the recorder can report successful completion.
          result = buffer_queue_.try_push(frame);
          if (result != CameraBufferQueue::PushResult::Accepted)
            recording_failed = check_for_record_failure(ctx);
        }

        if (result == CameraBufferQueue::PushResult::Accepted) {
          pending.transfer();
          update_acquisition_status(ctx, last_update_event);
        } else {
          pending.release();
          if (result == CameraBufferQueue::PushResult::Closed)
            break;
          if (recording_failed) {
            logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] {}",
                            overflow_message);
            emit_recording_failed_event(ctx, current_recording_session_->id(), overflow_message);
          }
        }
      }
    } catch (const std::exception &e) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] {}", e.what());
      fail_recording(ctx, std::string("Camera acquisition failed: ") + e.what());
      buffer_queue_.close();
    } catch (...) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] unknown error");
      fail_recording(ctx, "Camera acquisition failed with an unknown error");
      buffer_queue_.close();
    }
    pending.release();
  }

  std::optional<std::byte *> validate_buffer_data(const Euresys::NewBufferData &data_a,
                                                  const Euresys::NewBufferData &data_b) {
    const std::lock_guard lock(diagnostics_mutex_);
    using namespace Euresys;
    constexpr auto DELIVERED = ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS;
    constexpr auto TIMESTAMP = GenTL::BUFFER_INFO_TIMESTAMP;

    auto buffer_a = Buffer(data_a);
    auto buffer_b = Buffer(data_b);

    try {
      uint64_t   delivered_a = buffer_a.getInfo<uint64_t>(*grabber_a_, DELIVERED);
      uint64_t   delivered_b = buffer_b.getInfo<uint64_t>(*grabber_b_, DELIVERED);
      uint64_t   ts_a        = buffer_a.getInfo<uint64_t>(*grabber_a_, TIMESTAMP);
      uint64_t   ts_b        = buffer_b.getInfo<uint64_t>(*grabber_b_, TIMESTAMP);
      std::byte *base_a =
          static_cast<std::byte *>(buffer_a.getInfo<void *>(*grabber_a_, GenTL::BUFFER_INFO_BASE));
      std::byte *base_b =
          static_cast<std::byte *>(buffer_b.getInfo<void *>(*grabber_b_, GenTL::BUFFER_INFO_BASE));
      const auto frame_id_a = buffer_a.getInfo<uint64_t>(*grabber_a_, GenTL::BUFFER_INFO_FRAMEID);
      const auto frame_id_b = buffer_b.getInfo<uint64_t>(*grabber_b_, GenTL::BUFFER_INFO_FRAMEID);

      if (base_a != base_b || delivered_a != runtime_cfg_.buffer_part_count ||
          delivered_b != runtime_cfg_.buffer_part_count || frame_id_a != frame_id_b) {
        ++rejected_pairs_since_log_;
        if (log_due(last_rejected_log_)) {
          logger()->warn(
              "[AmetekS711EuresysCoaxlinkQSFP::validate_buffer_data] rejected {} two-bank "
              "buffer pair(s): latest bank A base={}, delivered={}, ts={}, frame_id={} | bank B "
              "base={}, delivered={}, ts={}, frame_id={} | expected delivered={}",
              rejected_pairs_since_log_, static_cast<void *>(base_a), delivered_a, ts_a, frame_id_a,
              static_cast<void *>(base_b), delivered_b, ts_b, frame_id_b,
              runtime_cfg_.buffer_part_count);
          rejected_pairs_since_log_ = 0;
        }
        return std::nullopt;
      }

      const auto ts_delta = ts_a >= ts_b ? ts_a - ts_b : ts_b - ts_a;
      ++accepted_pairs_since_log_;
      if (ts_delta > max_ts_delta_since_log_) {
        max_ts_delta_since_log_ = ts_delta;
      }
      if (log_due(last_pair_log_)) {
        logger()->info("[AmetekS711EuresysCoaxlinkQSFP::validate_buffer_data] accepted {} two-bank "
                       "buffer pair(s): latest base={}, delivered={}, bank A ts={}, bank B "
                       "ts={}, max ts delta={} us",
                       accepted_pairs_since_log_, static_cast<void *>(base_a), delivered_a, ts_a,
                       ts_b, max_ts_delta_since_log_);
        accepted_pairs_since_log_ = 0;
        max_ts_delta_since_log_   = 0;
      }

      return base_a;
    } catch (const Euresys::genapi_error &err) {
      if (log_due(last_error_log_)) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] GenApi error: {}",
                        format_genapi_error(err));
      }
    } catch (const Euresys::gentl_error &err) {
      if (log_due(last_error_log_)) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] GenTL error: {}",
                        err.what());
      }
    } catch (const std::exception &err) {
      if (log_due(last_error_log_)) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] error: {}", err.what());
      }
    }
    return std::nullopt;
  }

  void on_update_record_path(holoflow::core::SyncCtx                    &ctx,
                             const std::optional<holoflow_event::Event> &event) {

    assert(recording_context_.has_value());

    const auto record_path = event->data.value("record_path", std::string{});
    if (record_path.empty()) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::on_update_record_path] Rejecting "
                      "update_record_path event with empty path");
      emit_recording_failed_event(ctx, current_recording_session_->id(),
                                  "Cannot update recording path to: empty path");
      return;
    }

    settings_.record_settings->file_path = record_path;
  }

  void on_start_recording_event(holoflow::core::SyncCtx                    &ctx,
                                const std::optional<holoflow_event::Event> &event) {

    assert(recording_context_.has_value());
    if (recording_context_->recording_.load(std::memory_order_acquire)) {
      logger()->error(
          "[AmetekS711EuresysCoaxlinkQSFP::on_start_recording_event] Ignoring duplicate "
          "start_recording event");
      emit_recording_failed_event(ctx, current_recording_session_->id(),
                                  "Recording already in progress");
      return;
    }

    const auto record_path = event->data.value("record_path", std::string{});
    if (record_path.empty()) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::on_start_recording_event] Rejecting "
                      "start_recording event with empty path");
      emit_recording_failed_event(ctx, current_recording_session_->id(),
                                  "Cannot start recording: empty path");
      return;
    }

    if (settings_.record_settings->recording_count <= 0) {
      const auto message = "Cannot start recording: invalid frame count (" +
                           std::to_string(settings_.record_settings->recording_count) + ")";
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::on_start_recording_event] {}", message);
      emit_recording_failed_event(ctx, current_recording_session_->id(), message);
      return;
    }

    // A finished worker may still be delivering its completion event.
    stop_raw_record();
    settings_.record_settings->file_path = record_path;
    try {
      start_raw_record(ctx);
    } catch (const std::exception &e) {
      emit_recording_failed_event(ctx, current_recording_session_->id(), e.what());
    }
  }

  void on_stop_recording_event(holoflow::core::SyncCtx                    &ctx,
                               const std::optional<holoflow_event::Event> &event) {
    (void)event;

    assert(recording_context_.has_value());
    if (!recording_context_->recording_.load(std::memory_order_acquire)) {
      logger()->warn(
          "[AmetekS711EuresysCoaxlinkQSFP::on_stop_recording_event] Ignoring stop_recording "
          "event while idle");
      return;
    }

    stop_raw_record();

    if (!settings_.record_settings->file_path.empty() &&
        std::remove(settings_.record_settings->file_path.c_str()) != 0 && errno != ENOENT) {
      std::error_code ec(errno, std::generic_category());
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::on_stop_recording_event] Failed to remove "
                      "incomplete recording at {}: {}",
                      settings_.record_settings->file_path, ec.message());
      emit_recording_failed_event(ctx, current_recording_session_->id(),
                                  "Failed to remove incomplete recording at " +
                                      settings_.record_settings->file_path + ": " + ec.message());
    }
  }

private:
  AmetekS711EuresysCoaxlinkQSFPSettings settings_;
  RuntimeConfig                         runtime_cfg_;
  HostPtr<uint8_t>                      buffers_;
  std::unique_ptr<Euresys::EGenTL>      gentl_;
  std::unique_ptr<Grabber>              grabber_a_;
  std::unique_ptr<Grabber>              grabber_b_;
  std::size_t                           buffer_size_;
  std::atomic<bool>                     running_;
  nlohmann::json                        cfg_;

  // Diagnostics state
  std::mutex                 diagnostics_mutex_;
  std::mutex                 requeue_error_mutex_;
  Clock::time_point          last_rejected_log_{};
  Clock::time_point          last_pair_log_{};
  Clock::time_point          last_error_log_{};
  Clock::time_point          last_requeue_error_log_{};
  Clock::time_point          last_lifecycle_log_{};
  Clock::time_point          last_pending_update_log_{};
  Clock::time_point          last_diagnostic_log_{};
  Clock::time_point          last_diagnostic_error_log_{};
  Clock::time_point          last_acquisition_log_{};
  Clock::time_point          last_event_log_{};
  uint64_t                   rejected_pairs_since_log_ = 0;
  uint64_t                   accepted_pairs_since_log_ = 0;
  uint64_t                   max_ts_delta_since_log_   = 0;
  uint64_t                   update_epoch_             = 0;
  bool                       first_pair_after_update_  = false;
  std::optional<std::string> first_pair_update_summary_;
  BankCounters               previous_counters_a_;
  BankCounters               previous_counters_b_;
  BankCounters               update_a_;
  BankCounters               update_b_;
  BankCounters               resume_a_;
  BankCounters               resume_b_;
  bool                       resume_counters_pending_ = false;
  uint64_t                   max_frame_step_a_        = 0;
  uint64_t                   max_frame_step_b_        = 0;
  uint64_t                   frame_regressions_a_     = 0;
  uint64_t                   frame_regressions_b_     = 0;
  bool                       pair_delta_seen_         = false;
  int64_t                    min_pair_delta_          = 0;
  int64_t                    max_pair_delta_          = 0;

  // Buffer Queue
  CameraBufferQueue buffer_queue_;

  std::optional<std::jthread> acquisition_thread_;

  // Record
  std::optional<RecordingContext>    recording_context_;
  curaii::unique_host_ptr<std::byte> recording_buffers_         = nullptr;
  size_t                             recording_buffers_size_    = 0;
  size_t                             recording_buffers_count_   = 0;
  RecordingSession                  *current_recording_session_ = nullptr;
};

// -------------------------------------------------------------------------------------------------
// Factory implementation
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
AmetekS711EuresysCoaxlinkQSFPFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                            const nlohmann::json &jsettings) const {
  check(input_descs.empty(), "expected zero input tensors");

  const auto settings    = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  const auto raw_cfg     = load_cfg(settings.cfg_path);
  const auto runtime_cfg = parse_cfg(raw_cfg);

  holoflow::core::TDesc odesc(
      {runtime_cfg.buffer_part_count, runtime_cfg.final_height, runtime_cfg.width},
      dtype_from_pixel_format(runtime_cfg.pixel_format), holoflow::core::MemLoc::Host);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {true},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS711EuresysCoaxlinkQSFPFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                             const nlohmann::json                  &jsettings,
                                             const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;
  check(input_descs.empty(), "expected zero input tensors");

  const auto settings    = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  const auto raw_cfg     = load_cfg(settings.cfg_path);
  const auto runtime_cfg = parse_cfg(raw_cfg);
  dump_cfg(raw_cfg, runtime_cfg);

  auto gentl       = std::make_unique<Euresys::EGenTL>();
  auto camera_info = find_camera(*gentl, runtime_cfg.camera_model_name);
  check(camera_info.has_value(),
        std::format("could not find {} camera", runtime_cfg.camera_model_name));

  apply_cfg(*camera_info, runtime_cfg);
  dump_state(*camera_info, "applied configuration");

  const auto bank_a_index = find_grabber_index_for_bank(*camera_info, 0);
  const auto bank_b_index = find_grabber_index_for_bank(*camera_info, 1);

  auto grabber_a =
      std::make_unique<Grabber>(camera_info->grabbers[bank_a_index], Grabber::EGrabberName::A);
  auto grabber_b =
      std::make_unique<Grabber>(camera_info->grabbers[bank_b_index], Grabber::EGrabberName::B);

  const holoflow::core::TDesc odesc(
      {runtime_cfg.buffer_part_count, runtime_cfg.final_height, runtime_cfg.width},
      dtype_from_pixel_format(runtime_cfg.pixel_format), holoflow::core::MemLoc::Host);

  auto buffer_size = odesc.num_bytes();
  auto record_buffer_needed =
      settings.record_settings.has_value()
          ? settings.record_settings->recording_count / runtime_cfg.buffer_part_count
          : 0;
  if (record_buffer_needed % runtime_cfg.nb_buffers != 0) {
    record_buffer_needed +=
        runtime_cfg.nb_buffers - (record_buffer_needed % runtime_cfg.nb_buffers);
  }
  auto buffer_count = std::max(record_buffer_needed, runtime_cfg.nb_buffers);
  auto buffers      = allocate_shared_buffers(*grabber_a, *grabber_b, buffer_count, buffer_size);

  return std::make_unique<AmetekS711EuresysCoaxlinkQSFP>(
      settings, runtime_cfg, std::move(buffers), buffer_count, std::move(gentl),
      std::move(grabber_a), std::move(grabber_b), buffer_size, normalized_cfg_json(runtime_cfg));
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS711EuresysCoaxlinkQSFPFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                             std::span<const holoflow::core::TDesc>     input_descs,
                                             const nlohmann::json                      &jsettings,
                                             const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)ctx;

  auto *old = dynamic_cast<AmetekS711EuresysCoaxlinkQSFP *>(old_task.get());
  if (old == nullptr || !input_descs.empty()) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings    = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  const auto raw_cfg     = load_cfg(settings.cfg_path);
  const auto runtime_cfg = parse_cfg(raw_cfg);
  const auto new_cfg     = normalized_cfg_json(runtime_cfg);

  // if record was enabled and is disabled in new settings, we don't need to rebuild
  auto record_settings_changed =
      settings.record_settings.has_value() && !old->get_settings().record_settings.has_value() ||
      settings.record_settings->requires_rebuild(*old->get_settings().record_settings);

  auto config_changed = new_cfg != old->get_cfg();
  auto need_rebuild   = record_settings_changed || config_changed;

  if (!need_rebuild) {
    old->update_settings(settings);
    old->stop_acquisition();
    old->log_update_lifecycle(false);
    return old_task;
  }
  if (config_changed) {
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] config changed");
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] old cfg: {}",
                   old->get_cfg().dump(2));
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] new cfg: {}", new_cfg.dump(2));
  }

  if (record_settings_changed) {
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] record settings changed and "
                   "requires a rebuild");
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] old record settings: {}",
                   old->get_settings().record_settings);
    logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory::update] new record settings: {}",
                   settings.record_settings);
  }

  old->stop_acquisition();
  old->log_update_lifecycle(true);
  old_task.reset(); // destroy old task and release all the buffers
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources

#else

#include <stdexcept>

namespace holotask::sources {

holoflow::core::InferResult
AmetekS711EuresysCoaxlinkQSFPFactory::infer(std::span<const holoflow::core::TDesc>,
                                            const nlohmann::json &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS711EuresysCoaxlinkQSFPFactory::create(std::span<const holoflow::core::TDesc>,
                                             const nlohmann::json &,
                                             const holoflow::core::SyncCreateCtx &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

std::unique_ptr<holoflow::core::ISyncTask> AmetekS711EuresysCoaxlinkQSFPFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc>,
    const nlohmann::json &, const holoflow::core::SyncCreateCtx &) const {
  throw std::logic_error("holotask library was built without EGrabber support");
}

} // namespace holotask::sources

#endif
