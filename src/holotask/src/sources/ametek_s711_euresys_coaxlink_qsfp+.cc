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

//#define HOLOTASK_HAS_EGRABBER 1
#ifdef HOLOTASK_HAS_EGRABBER

#include <EGrabber.h>
#include <EuresysGenapiErrorFormats.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j = nlohmann::json{{"cfg_path", s.cfg_path}};
}

void from_json(const nlohmann::json &j, AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j.at("cfg_path").get_to(s.cfg_path);
}

// -------------------------------------------------------------------------------------------------
// Private implementation types
// -------------------------------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

bool log_due(Clock::time_point &last_log) {
  const auto now = Clock::now();
  if (last_log != Clock::time_point{} && now - last_log < std::chrono::seconds(1)) {
    return false;
  }
  last_log = now;
  return true;
}

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
HostPtr<uint8_t> allocate_shared_buffers(Euresys::EGrabber<> &grabber_a,
                                         Euresys::EGrabber<> &grabber_b, std::size_t nb_buffers,
                                         std::size_t buffer_size) {
  logger()->info(
      "[AmetekS711EuresysCoaxlinkQSFPFactory] allocating {} shared host buffers of size {} bytes",
      nb_buffers, buffer_size);

  const auto total_size = buffer_size * nb_buffers;
  auto       buffers    = curaii::make_unique_host_ptr<uint8_t>(total_size);

  for (std::size_t buf_idx = 0; buf_idx < nb_buffers; ++buf_idx) {
    auto *base = buffers.get() + buf_idx * buffer_size;

    grabber_a.announceAndQueue(Euresys::UserMemory(base, buffer_size));
    grabber_b.announceAndQueue(Euresys::UserMemory(base, buffer_size));

    logger()->debug("[AmetekS711EuresysCoaxlinkQSFPFactory] announced shared buffer {} at address "
                    "{} to both grabbers",
                    buf_idx, static_cast<void *>(base));
  }

  return buffers;
}

void requeue_buffer_noexcept(Euresys::EGrabber<> &grabber, const Euresys::NewBufferData &data,
                             const char *label, Clock::time_point &last_error_log) {
  try {
    Euresys::Buffer(data).push(grabber);
  } catch (const std::exception &e) {
    if (log_due(last_error_log)) {
      logger()->error(
          "[AmetekS711EuresysCoaxlinkQSFP] failed to requeue {} buffer while handling an error: {}",
          label, e.what());
    }
  }
}

holoflow::core::DType dtype_from_pixel_format(const std::string &pixel_format) {
  static const std::map<std::string, holoflow::core::DType> dtypes = {
      {"Mono8", holoflow::core::DType::U8},
      {"Mono16", holoflow::core::DType::U16},
  };

  check(dtypes.contains(pixel_format), "unsupported PixelFormat: " + pixel_format);
  return dtypes.at(pixel_format);
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
                                std::unique_ptr<Euresys::EGenTL>     &&gentl,
                                std::unique_ptr<Euresys::EGrabber<>> &&grabber_a,
                                std::unique_ptr<Euresys::EGrabber<>> &&grabber_b,
                                std::size_t buffer_size, nlohmann::json normalized_cfg)
      : settings_(settings), runtime_cfg_(std::move(runtime_cfg)), buffers_(std::move(buffers)),
        gentl_(std::move(gentl)), grabber_a_(std::move(grabber_a)),
        grabber_b_(std::move(grabber_b)), buffer_size_(buffer_size), running_(false),
        cfg_(std::move(normalized_cfg)) {
    HOLOVIBES_CHECK(gentl_ != nullptr);
    HOLOVIBES_CHECK(grabber_a_ != nullptr);
    HOLOVIBES_CHECK(grabber_b_ != nullptr);
    HOLOVIBES_CHECK(buffers_ != nullptr);
  }

  ~AmetekS711EuresysCoaxlinkQSFP() override {
    try {
      if (running_) {
        acquisition_stopped_.store(true, std::memory_order_release);
        acquisition_thread_.join();
        grabber_a_->stop();
        grabber_b_->stop();
      }
    } catch (const std::exception &e) {
      logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::~AmetekS711EuresysCoaxlinkQSFP] {}",
                     e.what());
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

    if (!pending_a_.has_value() || !pending_b_.has_value()) {
      throw std::logic_error("release_output called with no pending two-bank frame");
    }

    Euresys::Buffer(*pending_a_).push(*grabber_a_);
    Euresys::Buffer(*pending_b_).push(*grabber_b_);
    pending_a_.reset();
    pending_b_.reset();
    acquisition_ready_.store(false, std::memory_order_release);
  }

  void log_update_lifecycle(bool replacing) {
    if ((pending_a_.has_value() || pending_b_.has_value()) && log_due(last_pending_update_log_)) {
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::update] updating with an unreleased "
                      "camera buffer: bank A pending={}, bank B pending={}",
                      pending_a_.has_value(), pending_b_.has_value());
    }
    if (!running_) {
      return;
    }
    if (!replacing) {
      ++update_epoch_;
      update_a_                = read_bank_counters(*grabber_a_, "A");
      update_b_                = read_bank_counters(*grabber_b_, "B");
      first_pair_after_update_ = true;
      first_pair_update_summary_.reset();
      resume_counters_pending_ = false;
    }
    if (!log_due(last_lifecycle_log_)) {
      return;
    }
    if (replacing) {
      logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::update] replacing a task while its "
                     "grabbers are still acquiring");
    } else {
      logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::update] reusing a task whose grabbers "
                     "kept acquiring during the pipeline update; epoch={}, queued frames may be "
                     "stale",
                     update_epoch_);
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

  BankCounters read_bank_counters(Euresys::EGrabber<> &grabber, const char *bank) {
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

  std::string part_timing(Euresys::Buffer &buffer_a, Euresys::Buffer &buffer_b,
                          uint64_t delivered_a, uint64_t delivered_b) {
    if (delivered_a != runtime_cfg_.buffer_part_count ||
        delivered_b != runtime_cfg_.buffer_part_count) {
      return "incomplete pair";
    }
    constexpr auto PART_TIMESTAMPS = Euresys::ge::BUFFER_INFO_CUSTOM_PART_TIMESTAMPS;
    const auto     a               = read_diagnostic<std::vector<char>>(
        [&] { return buffer_a.getInfo<std::vector<char>>(*grabber_a_, PART_TIMESTAMPS); },
        "bank A part timestamps");
    const auto b = read_diagnostic<std::vector<char>>(
        [&] { return buffer_b.getInfo<std::vector<char>>(*grabber_b_, PART_TIMESTAMPS); },
        "bank B part timestamps");
    const auto expected_bytes = runtime_cfg_.buffer_part_count * sizeof(uint64_t);
    if (!a || !b || a->size() != expected_bytes || b->size() != expected_bytes) {
      if (a && b && log_due(last_diagnostic_error_log_)) {
        logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::diagnostics] part timestamp sizes "
                       "A={}, B={}, expected={}",
                       a->size(), b->size(), expected_bytes);
      }
      return "unavailable";
    }

    uint64_t first_a = 0, first_b = 0, last_a = 0, last_b = 0;
    int64_t  min_delta = 0, max_delta = 0;
    for (size_t i = 0; i < runtime_cfg_.buffer_part_count; ++i) {
      uint64_t ts_a, ts_b;
      std::memcpy(&ts_a, a->data() + i * sizeof(uint64_t), sizeof(uint64_t));
      std::memcpy(&ts_b, b->data() + i * sizeof(uint64_t), sizeof(uint64_t));
      const auto delta = signed_delta(ts_a, ts_b);
      if (i == 0) {
        first_a   = ts_a;
        first_b   = ts_b;
        min_delta = max_delta = delta;
      } else {
        min_delta = (std::min)(min_delta, delta);
        max_delta = (std::max)(max_delta, delta);
      }
      last_a = ts_a;
      last_b = ts_b;
    }
    return std::format("first A/B={}/{}, last A/B={}/{}, A-B first/last={}/{}, range=[{},{}] us",
                       first_a, first_b, last_a, last_b, signed_delta(first_a, first_b),
                       signed_delta(last_a, last_b), min_delta, max_delta);
  }

  void record_frame_id(const std::optional<uint64_t> &current, std::optional<uint64_t> &previous,
                       uint64_t &max_step, uint64_t &regressions) {
    if (!current) {
      previous.reset();
      return;
    }
    if (previous) {
      if (*current < *previous) {
        ++regressions;
      } else {
        max_step = (std::max)(max_step, *current - *previous);
      }
    }
    previous = current;
  }

  void log_bank_diagnostics(const char *bank, const std::optional<uint64_t> &frame_id,
                            uint64_t max_step, uint64_t regressions, const BankCounters &current,
                            BankCounters &previous, const BankCounters &update_baseline) {
    logger()->info(
        "[AmetekS711EuresysCoaxlinkQSFP::diagnostics] epoch={} bank {}: frame ID={}, "
        "max raw ID step={}, ID decreases/wraps={}, rejected frames={}, broken frames={}, "
        "lost buffers (underrun)={}, queued buffers={}, awaiting delivery={}",
        update_epoch_, bank, show(frame_id), max_step, regressions,
        show_counter(current.rejected_frames, previous.rejected_frames,
                     update_baseline.rejected_frames),
        show_counter(current.broken_frames, previous.broken_frames, update_baseline.broken_frames),
        show_counter(current.underrun_buffers, previous.underrun_buffers,
                     update_baseline.underrun_buffers),
        show(current.queued_buffers), show(current.awaiting_buffers));
    previous = current;
  }

  void log_resume_counters(const char *bank, const BankCounters &resumed,
                           const BankCounters &at_update) {
    logger()->info("[AmetekS711EuresysCoaxlinkQSFP::diagnostics] epoch={} first resumed bank "
                   "{}: rejected frames={} (since update +{}), broken frames={} (since update "
                   "+{}), lost buffers={} (since update +{}), queued update/resume={}/{}, "
                   "awaiting update/resume={}/{}",
                   update_epoch_, bank, show(resumed.rejected_frames),
                   show_change(resumed.rejected_frames, at_update.rejected_frames),
                   show(resumed.broken_frames),
                   show_change(resumed.broken_frames, at_update.broken_frames),
                   show(resumed.underrun_buffers),
                   show_change(resumed.underrun_buffers, at_update.underrun_buffers),
                   show(at_update.queued_buffers), show(resumed.queued_buffers),
                   show(at_update.awaiting_buffers), show(resumed.awaiting_buffers));
  }

  void record_pair_diagnostics(Euresys::Buffer &buffer_a, Euresys::Buffer &buffer_b,
                               uint64_t delivered_a, uint64_t delivered_b, uint64_t ts_a,
                               uint64_t ts_b, const Euresys::NewBufferData &data_a,
                               const Euresys::NewBufferData &data_b) {
    const auto read_frame_id = [&](Euresys::Buffer &buffer, Euresys::EGrabber<> &grabber,
                                   const char *label, Clock::time_point &retry_at) {
      if (Clock::now() < retry_at) {
        return std::optional<uint64_t>{};
      }
      auto frame_id = read_diagnostic<uint64_t>(
          [&] { return buffer.getInfo<uint64_t>(grabber, GenTL::BUFFER_INFO_FRAMEID); }, label);
      if (!frame_id) {
        // An unsupported info command must not throw on every high-rate buffer.
        retry_at = Clock::now() + std::chrono::seconds(1);
      }
      return frame_id;
    };
    const auto frame_a = read_frame_id(buffer_a, *grabber_a_, "bank A frame ID", frame_id_retry_a_);
    const auto frame_b = read_frame_id(buffer_b, *grabber_b_, "bank B frame ID", frame_id_retry_b_);
    record_frame_id(frame_a, previous_frame_a_, max_frame_step_a_, frame_regressions_a_);
    record_frame_id(frame_b, previous_frame_b_, max_frame_step_b_, frame_regressions_b_);

    const auto delta = signed_delta(ts_a, ts_b);
    if (!pair_delta_seen_) {
      min_pair_delta_ = max_pair_delta_ = delta;
      pair_delta_seen_                  = true;
    } else {
      min_pair_delta_ = (std::min)(min_pair_delta_, delta);
      max_pair_delta_ = (std::max)(max_pair_delta_, delta);
    }

    const bool due = log_due(last_diagnostic_log_);
    if (first_pair_after_update_) {
      resume_a_                = read_bank_counters(*grabber_a_, "A");
      resume_b_                = read_bank_counters(*grabber_b_, "B");
      resume_counters_pending_ = true;
      first_pair_update_summary_ =
          std::format("A/B ts={}/{}, A-B={} us, event A-B={} us, frame IDs={}/{}, parts: {}", ts_a,
                      ts_b, delta, signed_delta(data_a.timestamp, data_b.timestamp), show(frame_a),
                      show(frame_b), part_timing(buffer_a, buffer_b, delivered_a, delivered_b));
      first_pair_after_update_ = false;
    }
    if (!due) {
      return;
    }

    logger()->info("[AmetekS711EuresysCoaxlinkQSFP::diagnostics] epoch={} latest A/B ts="
                   "{}/{}, signed A-B={} us, interval range=[{},{}] us, event A-B={} us, "
                   "first after update={}, parts: {}",
                   update_epoch_, ts_a, ts_b, delta, min_pair_delta_, max_pair_delta_,
                   signed_delta(data_a.timestamp, data_b.timestamp),
                   first_pair_update_summary_.value_or("n/a"),
                   part_timing(buffer_a, buffer_b, delivered_a, delivered_b));

    const auto counters_a = read_bank_counters(*grabber_a_, "A");
    const auto counters_b = read_bank_counters(*grabber_b_, "B");
    log_bank_diagnostics("A", frame_a, max_frame_step_a_, frame_regressions_a_, counters_a,
                         previous_counters_a_, update_a_);
    log_bank_diagnostics("B", frame_b, max_frame_step_b_, frame_regressions_b_, counters_b,
                         previous_counters_b_, update_b_);
    if (resume_counters_pending_) {
      log_resume_counters("A", resume_a_, update_a_);
      log_resume_counters("B", resume_b_, update_b_);
      resume_counters_pending_ = false;
    }
    first_pair_update_summary_.reset();
    update_a_         = {};
    update_b_         = {};
    max_frame_step_a_ = max_frame_step_b_ = 0;
    frame_regressions_a_ = frame_regressions_b_ = 0;
    pair_delta_seen_                            = false;
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    using namespace Euresys;
    constexpr auto DELIVERED = ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS;
    constexpr auto TIMESTAMP = GenTL::BUFFER_INFO_TIMESTAMP;

    HOLOVIBES_CHECK(!pending_a_.has_value() && !pending_b_.has_value(),
                    "execute called while previous buffer pair is still held");

    if (!running_) {
      // S711 Banks_AB must start bank B first, then bank A.
      grabber_b_->enableEvent<Euresys::NewBufferData>();
      grabber_b_->start();
      grabber_a_->enableEvent<Euresys::NewBufferData>();
      grabber_a_->start();
      running_ = true;
      // TODO start acquisition thread
      acquisition_thread_ = std::thread([this] {
        logger()->info("[AmetekS711EuresysCoaxlinkQSFP::execute] Starting acquisition thread");
        acquisition_loop();
        logger()->info("[AmetekS711EuresysCoaxlinkQSFP::execute] Stopping acquisition thread");
       });
    }

    while (!ctx.cancelled->load()) {
        if (acquisition_ready_.load(std::memory_order_acquire) && !pending_a_.has_value() && !pending_b_.has_value()) {
          auto &storage = storage_access().owned_output_storage(0);
          storage.ptr   = base_a;

          ctx.outputs[0] = holoflow::core::TView{
              .desc    = ctx.outputs[0].desc,
              .storage = &storage,
          };
        }
    }
    return holoflow::core::OpResult::Cancelled;
  }

  const nlohmann::json &get_cfg() const { return cfg_; }

private:
  AmetekS711EuresysCoaxlinkQSFPSettings settings_;
  RuntimeConfig                         runtime_cfg_;
  HostPtr<uint8_t>                      buffers_;
  std::unique_ptr<Euresys::EGenTL>      gentl_;
  std::unique_ptr<Euresys::EGrabber<>>  grabber_a_;
  std::unique_ptr<Euresys::EGrabber<>>  grabber_b_;
  std::size_t                           buffer_size_;
  bool                                  running_;
  nlohmann::json                        cfg_;
  std::optional<Euresys::NewBufferData> pending_a_;
  std::optional<Euresys::NewBufferData> pending_b_;

  // Diagnostics state
  Clock::time_point          last_rejected_log_{};
  Clock::time_point          last_pair_log_{};
  Clock::time_point          last_error_log_{};
  Clock::time_point          last_requeue_error_log_{};
  Clock::time_point          last_lifecycle_log_{};
  Clock::time_point          last_pending_update_log_{};
  Clock::time_point          last_diagnostic_log_{};
  Clock::time_point          last_diagnostic_error_log_{};
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
  std::optional<uint64_t>    previous_frame_a_;
  std::optional<uint64_t>    previous_frame_b_;
  Clock::time_point          frame_id_retry_a_{};
  Clock::time_point          frame_id_retry_b_{};
  uint64_t                   max_frame_step_a_    = 0;
  uint64_t                   max_frame_step_b_    = 0;
  uint64_t                   frame_regressions_a_ = 0;
  uint64_t                   frame_regressions_b_ = 0;
  bool                       pair_delta_seen_     = false;
  int64_t                    min_pair_delta_      = 0;
  int64_t                    max_pair_delta_      = 0;

  // acquisition thread
  std::thread       acquisition_thread_;
  std::atomic<bool> acquisition_ready_ = false;
  std::atomic<bool> acquisition_stop_  = false;

  void acquisition_loop() {
    while (!acquisition_stop_.load(std::memory_order_acquire)) {
      try {
        const auto timeout_ms = runtime_cfg_.pop_timeout_ms;

        auto                                  data_a = grabber_a_->pop(timeout_ms);
        std::optional<Euresys::NewBufferData> data_b;
        try {
          data_b = grabber_b_->pop(timeout_ms);
        } catch (...) {
          requeue_buffer_noexcept(*grabber_a_, data_a, "bank A", last_requeue_error_log_);
          throw;
        }

        auto buffer_a = Buffer(data_a);
        auto buffer_b = Buffer(*data_b);

        uint64_t   delivered_a;
        uint64_t   delivered_b;
        uint64_t   ts_a;
        uint64_t   ts_b;
        std::byte *base_a;
        std::byte *base_b;
        try {
          delivered_a = buffer_a.getInfo<uint64_t>(*grabber_a_, DELIVERED);
          delivered_b = buffer_b.getInfo<uint64_t>(*grabber_b_, DELIVERED);
          ts_a        = buffer_a.getInfo<uint64_t>(*grabber_a_, TIMESTAMP);
          ts_b        = buffer_b.getInfo<uint64_t>(*grabber_b_, TIMESTAMP);
          base_a      = static_cast<std::byte *>(
              buffer_a.getInfo<void *>(*grabber_a_, GenTL::BUFFER_INFO_BASE));
          base_b = static_cast<std::byte *>(
              buffer_b.getInfo<void *>(*grabber_b_, GenTL::BUFFER_INFO_BASE));
        } catch (...) {
          requeue_buffer_noexcept(*grabber_a_, data_a, "bank A", last_requeue_error_log_);
          requeue_buffer_noexcept(*grabber_b_, *data_b, "bank B", last_requeue_error_log_);
          throw;
        }

        try {
          record_pair_diagnostics(buffer_a, buffer_b, delivered_a, delivered_b, ts_a, ts_b, data_a,
                                  *data_b);
        } catch (const std::exception &e) {
          if (log_due(last_diagnostic_error_log_)) {
            logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::diagnostics] failed: {}", e.what());
          }
        }

        if (base_a != base_b || delivered_a != runtime_cfg_.buffer_part_count ||
            delivered_b != runtime_cfg_.buffer_part_count) {
          ++rejected_pairs_since_log_;
          if (log_due(last_rejected_log_)) {
            logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] rejected {} two-bank "
                           "buffer pair(s): latest bank A base={}, delivered={}, ts={} | bank B "
                           "base={}, delivered={}, ts={} | expected delivered={}",
                           rejected_pairs_since_log_, static_cast<void *>(base_a), delivered_a,
                           ts_a, static_cast<void *>(base_b), delivered_b, ts_b,
                           runtime_cfg_.buffer_part_count);
            rejected_pairs_since_log_ = 0;
          }
          requeue_buffer_noexcept(*grabber_a_, data_a, "bank A", last_requeue_error_log_);
          requeue_buffer_noexcept(*grabber_b_, *data_b, "bank B", last_requeue_error_log_);
          continue;
        }

        const auto ts_delta = ts_a >= ts_b ? ts_a - ts_b : ts_b - ts_a;
        ++accepted_pairs_since_log_;
        if (ts_delta > max_ts_delta_since_log_) {
          max_ts_delta_since_log_ = ts_delta;
        }
        if (log_due(last_pair_log_)) {
          logger()->info("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] accepted {} two-bank "
                         "buffer pair(s): latest base={}, delivered={}, bank A ts={}, bank B "
                         "ts={}, max ts delta={} us",
                         accepted_pairs_since_log_, static_cast<void *>(base_a), delivered_a, ts_a,
                         ts_b, max_ts_delta_since_log_);
          accepted_pairs_since_log_ = 0;
          max_ts_delta_since_log_   = 0;
        }

        // just discard all frames if the previous one is still held by the pipeline
        // TODO buffer them and throw only when a certain threshold is reached
        if (pending_a_.has_value() || pending_b_.has_value()) {
          logger()->warn(
              "acquisition thread produced a new buffer pair while the previous one is still "
              "held");
          requeue_buffer_noexcept(*grabber_a_, data_a, "bank A", last_requeue_error_log_);
          requeue_buffer_noexcept(*grabber_b_, *data_b, "bank B", last_requeue_error_log_);

        } else {
          pending_a_ = std::move(data_a);
          pending_b_ = std::move(*data_b);
          acquisition_ready_.store(true, std::memory_order_release);
        }
        return holoflow::core::OpResult::Ok;

      } catch (const Euresys::genapi_error &err) {
        if (log_due(last_error_log_)) {
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] GenApi error: {}",
                          format_genapi_error(err));
        }
      } catch (const Euresys::gentl_error &err) {
        if (log_due(last_error_log_)) {
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] GenTL error: {}", err.what());
        }
      } catch (const std::exception &err) {
        if (log_due(last_error_log_)) {
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::acquisition_loop] error: {}", err.what());
        }
      }
    }
  }
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

  dump_state(*camera_info, "before");
  apply_cfg(*camera_info, runtime_cfg);
  dump_state(*camera_info, "after");

  const auto bank_a_index = find_grabber_index_for_bank(*camera_info, 0);
  const auto bank_b_index = find_grabber_index_for_bank(*camera_info, 1);

  auto grabber_a = std::make_unique<Euresys::EGrabber<>>(camera_info->grabbers[bank_a_index]);
  auto grabber_b = std::make_unique<Euresys::EGrabber<>>(camera_info->grabbers[bank_b_index]);

  const holoflow::core::TDesc odesc(
      {runtime_cfg.buffer_part_count, runtime_cfg.final_height, runtime_cfg.width},
      dtype_from_pixel_format(runtime_cfg.pixel_format), holoflow::core::MemLoc::Host);

  auto buffer_size = odesc.num_bytes();
  auto buffers =
      allocate_shared_buffers(*grabber_a, *grabber_b, runtime_cfg.nb_buffers, buffer_size);

  return std::make_unique<AmetekS711EuresysCoaxlinkQSFP>(
      settings, runtime_cfg, std::move(buffers), std::move(gentl), std::move(grabber_a),
      std::move(grabber_b), buffer_size, normalized_cfg_json(runtime_cfg));
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

  if (new_cfg == old->get_cfg()) {
    old->log_update_lifecycle(false);
    return old_task;
  }

  old->log_update_lifecycle(true);
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources

#else

#include <stdexcept>

namespace holotask::sources {

void to_json(nlohmann::json &j, const AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j = nlohmann::json{{"cfg_path", s.cfg_path}};
}

void from_json(const nlohmann::json &j, AmetekS711EuresysCoaxlinkQSFPSettings &s) {
  j.at("cfg_path").get_to(s.cfg_path);
}

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
