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

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const RecordSettings &s) {
  j = nlohmann::json{{"file_path", s.file_path},
                    {"recording_count", s.recording_count},
                    {"pipeline_config", s.pipeline_config}};
}

void from_json(const nlohmann::json &j, RecordSettings &s) {
  j.at("file_path").get_to(s.file_path);
  j.at("recording_count").get_to(s.recording_count);
  j.at("pipeline_config").get_to(s.pipeline_config);
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

// #define HOLOTASK_HAS_EGRABBER 1
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
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

#include "holofile/holofile.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

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

class Grabber;
void requeue_buffer_noexcept(Grabber &grabber, const Euresys::NewBufferData &data,
                             const char *label, Clock::time_point &last_error_log,
                             std::mutex &last_error_log_mutex);

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

class Grabber : public Euresys::EGrabber<Euresys::CallbackSingleThread> {
public:
  enum class EGrabberName : uint8_t { A = 0, B = 1 };
  using EnqueueBufferCallback =
      std::function<void(size_t producer_id, const Euresys::NewBufferData &)>;

  Grabber(Euresys::EGrabberInfo info, EGrabberName name,
          std::optional<EnqueueBufferCallback> enqueue_callback = std::nullopt)
      : EGrabber<Euresys::CallbackSingleThread>(info), name_{name},
        enqueue_callback_{enqueue_callback} {
    enableEvent<Euresys::NewBufferData>();
  }

  void set_enqueue_callback(EnqueueBufferCallback &&enqueue_callback) {
    enqueue_callback_ = std::move(enqueue_callback);
  }

private:
  virtual void onNewBufferEvent(const Euresys::NewBufferData &data) override {
    if (enqueue_callback_.has_value())
      enqueue_callback_->operator()(static_cast<size_t>(name_), data);
    else
      requeue_buffer_noexcept(*this, data, "no enqueue callback set", last_error_log_,
                              last_error_log_mutex_);
  }

private:
  EGrabberName                         name_;
  std::optional<EnqueueBufferCallback> enqueue_callback_;
  Clock::time_point                    last_error_log_;
  std::mutex                           last_error_log_mutex_;
};

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
  constexpr size_t safety_padding_size = 16;
  size_t           actual_nb_buffers   = nb_buffers + safety_padding_size;
  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] allocating {} ({} + {}) shared host "
                 "buffers of size {} bytes",
                 actual_nb_buffers, nb_buffers, safety_padding_size, buffer_size);

  const auto total_size = buffer_size * actual_nb_buffers;
  auto       buffers    = curaii::make_unique_host_ptr<uint8_t>(total_size);

  for (std::size_t buf_idx = 0; buf_idx < actual_nb_buffers; ++buf_idx) {
    auto *base = buffers.get() + buf_idx * buffer_size;

    grabber_a.announceAndQueue(Euresys::UserMemory(base, buffer_size));
    grabber_b.announceAndQueue(Euresys::UserMemory(base, buffer_size));

    logger()->debug("[AmetekS711EuresysCoaxlinkQSFPFactory] announced shared buffer {} at address "
                    "{} to both grabbers",
                    buf_idx, static_cast<void *>(base));
  }

  return buffers;
}

void requeue_buffer_noexcept(Grabber &grabber, const Euresys::NewBufferData &data,
                             const char *label, Clock::time_point &last_error_log,
                             std::mutex &last_error_log_mutex) {
  try {
    Euresys::Buffer(data).push(grabber);
  } catch (const std::exception &e) {
    const std::lock_guard lock(last_error_log_mutex);
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

struct CameraFrame {
  Euresys::NewBufferData bank_a;
  Euresys::NewBufferData bank_b;
  std::byte             *base{};
};

class CameraBufferQueue {
public:
  using DType                     = CameraFrame;
  using BufferDataReleaseCallback = std::function<void(const Euresys::NewBufferData &)>;
  using BufferAssemblerCallback   = std::function<std::optional<CameraFrame>(
      Euresys::NewBufferData &&, Euresys::NewBufferData &&)>;

  static constexpr size_t producer_count = 2;

  CameraBufferQueue(size_t capacity, BufferDataReleaseCallback part0_release_callback,
                    BufferDataReleaseCallback part1_release_callback,
                    BufferAssemblerCallback   buffer_assembler_callback)
      : capacity_{capacity}, slots_{std::make_unique<Slot[]>(capacity)},
        part0_release_callback_{std::move(part0_release_callback)},
        part1_release_callback_{std::move(part1_release_callback)},
        buffer_assembler_callback_{std::move(buffer_assembler_callback)} {
    if (capacity_ == 0) {
      throw std::invalid_argument("CameraBufferQueue capacity must be > 0");
    }
  }

  ~CameraBufferQueue() {
    closed_.store(true, std::memory_order_release);

    for (size_t i = 0; i < capacity_; ++i) {
      auto &slot = slots_[i];

      const auto state = slot.state.load(std::memory_order_relaxed);

      if (state == SlotState::Collecting) {
        /*
         * One or both producers may have arrived.
         *
         * The queue still owns whichever buffers are present.
         */
        if (slot.part0.has_value()) {
          part0_release_callback_(*slot.part0);
          slot.part0.reset();
        }

        if (slot.part1.has_value()) {
          part1_release_callback_(*slot.part1);
          slot.part1.reset();
        }

        continue;
      }

      if (state == SlotState::Ready) {
        /*
         * A Ready frame belongs to the queue/readers.
         */
        if (slot.readers.load(std::memory_order_relaxed) != 0) {
          release_callback(std::move(slot.data));
        }

        continue;
      }

      /*
       * Dropped:
       *
       * The assembler already took ownership of both input
       * buffers and is responsible for releasing/requeueing them.
       *
       * Empty:
       *
       * Nothing to release.
       */
    }
  }

  CameraBufferQueue(const CameraBufferQueue &)            = delete;
  CameraBufferQueue &operator=(const CameraBufferQueue &) = delete;

  CameraBufferQueue(CameraBufferQueue &&)            = delete;
  CameraBufferQueue &operator=(CameraBufferQueue &&) = delete;

  /**
   * Push one half of a frame.
   *
   * The nth frame submitted by producer 0 is paired with the
   * nth frame submitted by producer 1.
   */
  void push(size_t producer, const Euresys::NewBufferData &frame) {
    assert(producer < producer_count);

    const auto release_frame = [&] {
      if (producer == 0) {
        part0_release_callback_(frame);
      } else {
        part1_release_callback_(frame);
      }
    };

    /*
     * Once closed, don't accept new buffers.
     */
    if (closed_.load(std::memory_order_acquire)) {
      release_frame();
      return;
    }

    /*
     * Each producer gets its own monotonically increasing
     * generation number.
     */
    const size_t generation =
        producer_write_index_[producer].fetch_add(1, std::memory_order_relaxed);

    auto &slot = slots_[generation % capacity_];

    /*
     * Claim the slot for this generation.
     *
     * This is the important part of the protocol:
     *
     *     invalid_generation -> generation
     *
     * Only one producer can perform this transition.
     */
    for (;;) {
      if (closed_.load(std::memory_order_acquire)) {
        release_frame();
        return;
      }

      size_t expected = invalid_generation;

      if (slot.generation.compare_exchange_weak(expected, generation, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {

        /*
         * We are the first producer for this generation.
         */
        assert(slot.state.load(std::memory_order_relaxed) == SlotState::Empty);

        assert(slot.parts_ready.load(std::memory_order_relaxed) == 0);

        assert(!slot.part0.has_value());
        assert(!slot.part1.has_value());

        assert(slot.readers.load(std::memory_order_acquire) == 0);

        slot.state.store(SlotState::Collecting, std::memory_order_release);

        break;
      }

      /*
       * The slot already belongs to some generation.
       *
       * It is possible that this is our generation: the other
       * producer got here first.
       */
      if (expected == generation) {
        /*
         * The first producer has claimed the slot.
         *
         * Wait until it has initialized the Collecting state.
         */
        while (slot.state.load(std::memory_order_acquire) == SlotState::Empty) {

          if (closed_.load(std::memory_order_acquire)) {
            release_frame();
            return;
          }

          std::this_thread::yield();
        }

        assert(slot.state.load(std::memory_order_acquire) == SlotState::Collecting);

        break;
      }

      /*
       * This slot belongs to another generation.
       *
       * This is normal when a producer gets ahead of the other
       * producer or when the queue is full.
       */
      std::this_thread::yield();
    }

    /*
     * Store our half.
     *
     * Producer 0 only touches part0.
     * Producer 1 only touches part1.
     */
    if (producer == 0) {
      slot.part0 = frame;
    } else {
      slot.part1 = frame;
    }

    /*
     * Publish that our producer has arrived.
     */
    const unsigned bit = 1u << static_cast<unsigned>(producer);

    const unsigned previous = slot.parts_ready.fetch_or(bit, std::memory_order_acq_rel);

    /*
     * Only the second producer assembles the frame.
     */
    if ((previous | bit) == producer_mask()) {
      assert(slot.generation.load(std::memory_order_acquire) == generation);
      complete_generation(slot);
    }
  }

  /**
   * Read the next frame for reader A.
   *
   * Returns nullptr if cancelled or the queue is closed before
   * another frame becomes available.
   */
  [[nodiscard]]
  const DType *read_a(const std::atomic<bool> *cancelled = nullptr) {
    return read(read_index_a_, cancelled);
  }

  /**
   * Read the next frame for reader B.
   *
   * Returns nullptr if cancelled or the queue is closed while waiting.
   */
  [[nodiscard]]
  const DType *read_b(const std::atomic<bool> *cancelled = nullptr) {
    return read(read_index_b_, cancelled);
  }

  void close() {
    logger()->info("[CameraBufferQueue::close] queue closed");
    closed_.store(true, std::memory_order_release);
  }

  void release_a() { release(read_index_a_); }

  void release_b() { release(read_index_b_); }

  void subscribe_b() {
    const std::lock_guard lock(publication_mutex_);
    if (reader_b_active_.load(std::memory_order_acquire)) {
      throw std::logic_error("CameraBufferQueue reader B is already active");
    }

    /*
     * B starts at the current publication point.
     */
    read_index_b_.store(write_index_.load(std::memory_order_acquire), std::memory_order_release);
    reader_b_active_.store(true, std::memory_order_release);
  }

  void unsubscribe_b() {
    const std::lock_guard lock(publication_mutex_);
    if (!reader_b_active_.exchange(false, std::memory_order_acq_rel)) {
      return;
    }

    const auto end = write_index_.load(std::memory_order_acquire);

    while (read_index_b_.load(std::memory_order_relaxed) != end) {
      release_b();
    }
  }

  [[nodiscard]]
  bool empty_b() const {
    return read_index_b_.load(std::memory_order_relaxed) ==
           write_index_.load(std::memory_order_acquire);
  }

  [[nodiscard]]
  size_t capacity() const {
    return capacity_;
  }

  /**
   * Number of published generations that have not yet been
   * consumed by the oldest active reader.
   */
  [[nodiscard]]
  size_t size() const {
    const auto write = write_index_.load(std::memory_order_acquire);

    const auto oldest = oldest_read_index();

    return write - oldest;
  }

  [[nodiscard]]
  bool empty() const {
    return size() == 0;
  }

private:
  enum class SlotState : unsigned char {
    Empty,
    Collecting,
    Ready,
    Dropped,
  };

  static constexpr size_t invalid_generation = std::numeric_limits<size_t>::max();

  static constexpr unsigned releasing_ = std::numeric_limits<unsigned>::max();

  static constexpr unsigned producer_mask() { return (1u << producer_count) - 1u; }

  struct Slot {
    /*
     * Generation currently owning this slot.
     *
     * invalid_generation means that the slot is reusable.
     *
     * This is the authoritative mechanism preventing a fast
     * producer from overwriting a still-collecting generation.
     */
    std::atomic<size_t> generation{invalid_generation};

    /*
     * Valid when state == Ready.
     */
    DType data;

    /*
     * Valid when state == Collecting.
     */
    std::optional<Euresys::NewBufferData> part0;
    std::optional<Euresys::NewBufferData> part1;

    /*
     * bit 0 -> producer 0 arrived
     * bit 1 -> producer 1 arrived
     */
    std::atomic<unsigned> parts_ready{0};

    /*
     * Lifetime state of this generation.
     */
    std::atomic<SlotState> state{SlotState::Empty};

    /*
     * Number of readers owning this generation.
     *
     * 0:
     *     no readers
     *
     * 1:
     *     reader A OR reader B
     *
     * 2:
     *     readers A+B
     *
     * releasing_:
     *     the final reader is currently releasing the frame
     */
    std::atomic<unsigned> readers{0};
  };

  /**
   * Called by the second producer.
   *
   * The assembler takes ownership of both NewBufferData objects.
   *
   * If the assembler returns nullopt, it is responsible for
   * releasing/requeueing both buffers.
   */
  void complete_generation(Slot &slot) {
    assert(slot.state.load(std::memory_order_acquire) == SlotState::Collecting);

    assert(slot.parts_ready.load(std::memory_order_acquire) == producer_mask());

    assert(slot.part0.has_value());
    assert(slot.part1.has_value());

    /*
     * The assembler takes ownership of both buffers.
     */
    auto assembled = buffer_assembler_callback_(std::move(*slot.part0), std::move(*slot.part1));

    /*
     * The queue no longer owns these buffers.
     *
     * This is true whether assembly succeeded or failed.
     */
    slot.part0.reset();
    slot.part1.reset();

    if (assembled.has_value()) {
      /*
       * Successfully assembled frame.
       */
      slot.data = std::move(*assembled);

      /*
       * Publish the complete frame.
       *
       * Everything written above becomes visible to readers
       * through this release/acquire pair.
       */
      slot.state.store(SlotState::Ready, std::memory_order_release);
    } else {
      /*
       * Assembly failed.
       *
       * The assembler already handled both input buffers.
       *
       * This generation is nevertheless considered completed
       * from the ordering perspective. Readers will skip it.
       */
      slot.state.store(SlotState::Dropped, std::memory_order_release);
    }

    /*
     * Make this generation and any consecutive completed
     * generations visible to readers.
     */
    publish_completed();
  }

  /**
   * Advance write_index through all consecutive completed
   * generations.
   *
   * Example:
   *
   *     generation 0 = Collecting
   *     generation 1 = Ready
   *     generation 2 = Ready
   *
   * write_index remains 0.
   *
   * Once generation 0 becomes Ready/Dropped, this function
   * advances:
   *
   *     0 -> 1 -> 2 -> 3
   */
  void publish_completed() {
    // Subscription changes and ownership assignment share the publication boundary.
    const std::lock_guard lock(publication_mutex_);
    for (;;) {
      const size_t current = write_index_.load(std::memory_order_acquire);

      auto &slot = slots_[current % capacity_];

      /*
       * Make sure this is actually the slot for `current`.
       *
       * This also protects against stale state from a reused slot.
       */
      const size_t generation = slot.generation.load(std::memory_order_acquire);

      if (generation != current) {
        return;
      }

      const SlotState state = slot.state.load(std::memory_order_acquire);

      if (state == SlotState::Empty || state == SlotState::Collecting) {
        /*
         * The next generation isn't complete yet.
         */
        return;
      }

      /*
       * Ready or Dropped.
       */
      const unsigned reader_count = reader_b_active_.load(std::memory_order_acquire) ? 2u : 1u;
      slot.readers.store(reader_count, std::memory_order_relaxed);
      size_t expected = current;

      if (write_index_.compare_exchange_strong(expected, current + 1, std::memory_order_release,
                                               std::memory_order_acquire)) {
        continue;
      }
    }
  }

  /**
   * Return the oldest active reader position.
   */
  [[nodiscard]]
  size_t oldest_read_index() const {
    const auto a = read_index_a_.load(std::memory_order_acquire);

    if (!reader_b_active_.load(std::memory_order_acquire)) {
      return a;
    }

    const auto b = read_index_b_.load(std::memory_order_acquire);

    return std::min(a, b);
  }

  /**
   * Read the next published generation.
   *
   * Dropped generations are consumed internally and skipped.
   */
  [[nodiscard]]
  const DType *read(std::atomic<size_t> &reader, const std::atomic<bool> *cancelled) {

    size_t current = reader.load(std::memory_order_relaxed);

    for (;;) {
      if (closed_.load(std::memory_order_acquire) ||
          (cancelled && cancelled->load(std::memory_order_acquire))) {
        return nullptr;
      }

      /*
       * Wait for a published generation.
       */
      while (current == write_index_.load(std::memory_order_acquire)) {

        if (closed_.load(std::memory_order_acquire)) {
          return nullptr;
        }

        if (cancelled && cancelled->load(std::memory_order_acquire)) {
          return nullptr;
        }

        std::this_thread::yield();

        current = reader.load(std::memory_order_relaxed);
      }

      auto &slot = slots_[current % capacity_];

/*
 * This is the important validation that was missing from
 * the previous implementation.
 *
 * A published generation must still own its slot.
 */
#ifndef NDEBUG
      const size_t generation = slot.generation.load(std::memory_order_acquire);
#endif

      assert(generation == current);

      const SlotState state = slot.state.load(std::memory_order_acquire);

      if (state == SlotState::Ready) {
        /*
         * The acquire on state synchronizes with the producer's
         * release store of Ready.
         */
        return &slot.data;
      }

      if (state == SlotState::Dropped) {
        /*
         * No CameraFrame exists for this generation.
         *
         * Consume our reader ownership and move on.
         */
        acknowledge_dropped(slot);

        reader.store(current + 1, std::memory_order_release);

        current++;
        continue;
      }

      /*
       * A reader must never observe Empty or Collecting for a
       * generation that write_index has published.
       */
      assert(false && "Published generation is not complete");
    }
  }

  /**
   * Release a Ready or Dropped generation.
   */
  void release(std::atomic<size_t> &reader) {
    const size_t current = reader.load(std::memory_order_relaxed);

    auto &slot = slots_[current % capacity_];

#ifndef NDEBUG
    const size_t generation = slot.generation.load(std::memory_order_acquire);
#endif

    assert(generation == current);

    const SlotState state = slot.state.load(std::memory_order_acquire);

    if (state == SlotState::Dropped) {
      acknowledge_dropped(slot);

      reader.store(current + 1, std::memory_order_release);

      return;
    }

    assert(state == SlotState::Ready);

    unsigned readers = slot.readers.load(std::memory_order_acquire);

    for (;;) {
      assert(readers != 0);
      assert(readers != releasing_);

      if (readers == 1) {
        /*
         * We are the final reader.
         *
         * Claim the final release operation.
         */
        if (slot.readers.compare_exchange_strong(readers, releasing_, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {

          /*
           * No other reader can access the frame now.
           */
          release_callback(std::move(slot.data));

          assert(!slot.part0.has_value());
          assert(!slot.part1.has_value());

          slot.parts_ready.store(0, std::memory_order_relaxed);

          slot.readers.store(0, std::memory_order_release);
          /*
           * The generation is no longer readable.
           */
          slot.state.store(SlotState::Empty, std::memory_order_release);

          /*
           * Finally release ownership of the slot.
           *
           * A producer waiting on this generation can now claim
           * it with CAS(invalid_generation -> new_generation).
           */
          slot.generation.store(invalid_generation, std::memory_order_release);

          break;
        }
      } else {
        /*
         * Another reader still owns the frame.
         */
        if (slot.readers.compare_exchange_strong(readers, readers - 1, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
          break;
        }
      }
    }

    reader.store(current + 1, std::memory_order_release);
  }

  /**
   * A dropped generation still has reader ownership.
   *
   * Once the final reader acknowledges it, the slot becomes
   * reusable.
   */
  void acknowledge_dropped(Slot &slot) {
    const unsigned remaining = slot.readers.fetch_sub(1, std::memory_order_acq_rel);

    assert(remaining != 0);
    assert(remaining != releasing_);

    if (remaining == 1) {
      /*
       * Last reader of the dropped generation.
       */
      slot.parts_ready.store(0, std::memory_order_relaxed);

      slot.state.store(SlotState::Empty, std::memory_order_release);

      slot.readers.store(0, std::memory_order_relaxed);
      slot.generation.store(invalid_generation, std::memory_order_release);
    }
  }

  /**
   * Release both buffers belonging to a successfully assembled
   * CameraFrame.
   */
  void release_callback(DType &&data) {
    part0_release_callback_(data.bank_a);
    part1_release_callback_(data.bank_b);
  }

private:
  static constexpr size_t cache_line_size = std::hardware_destructive_interference_size;

  const size_t capacity_;

  std::atomic<bool> closed_{false};
  std::mutex publication_mutex_;

  std::unique_ptr<Slot[]> slots_;

  BufferDataReleaseCallback part0_release_callback_;
  BufferDataReleaseCallback part1_release_callback_;

  /*
   * The assembler owns both input buffers once called.
   *
   * On failure it must requeue/release them itself.
   */
  BufferAssemblerCallback buffer_assembler_callback_;

  /*
   * First unpublished generation.
   *
   * Readers only see generations < write_index_.
   */
  alignas(cache_line_size) std::atomic<size_t> write_index_{0};

  /*
   * Independent producer generation counters.
   */
  alignas(cache_line_size) std::atomic<size_t> producer_write_index_[producer_count]{0, 0};

  /*
   * Reader positions.
   */
  alignas(cache_line_size) std::atomic<size_t> read_index_a_{0};

  alignas(cache_line_size) std::atomic<size_t> read_index_b_{0};

  alignas(cache_line_size) std::atomic<bool> reader_b_active_{false};
};

// read B
class Recorder {
public:
  struct RecordingGeometry {
    uint8_t bits_per_pixel;
    size_t  frame_width;
    size_t  frame_height;
  };

  Recorder(const std::string &file_path, uint32_t frame_count, size_t buffer_part_count,
           CameraBufferQueue &queue, const RecordingGeometry &g,
           const nlohmann::json &pipeline_settings)
      : writer_{file_path,
                holofile::Header{
                    .magic_number   = holofile::Header::MAGIC_NUMBER_LE,
                    .version        = holofile::Header::CURRENT_VERSION,
                    .bits_per_pixel = g.bits_per_pixel,
                    .frame_width    = static_cast<uint32_t>(g.frame_width),
                    .frame_height   = static_cast<uint32_t>(g.frame_height),
                    .frame_count    = frame_count,
                    .data_size_in_bytes =
                        frame_count * g.frame_height * g.frame_width * g.bits_per_pixel / 8,
                    .endianness = holofile::Header::LITTLE_ENDIAN,
                },
                holofile::Footer{pipeline_settings}},
        frame_to_record_{frame_count}, buffer_part_count_{buffer_part_count}, current_frame_{0},
        queue_{queue} {
    if (buffer_part_count == 0)
      throw std::invalid_argument("Cannot record with zero buffer parts");
    queue_.subscribe_b();
    logger()->info("Create recorder with width: {}, height: {}, bits_per_pixel: {}, frame_count: {}", g.frame_width, g.frame_height, g.bits_per_pixel, frame_count);
  }

  ~Recorder() {
    queue_.unsubscribe_b();
    writer_.flush();
  }

  size_t execute(std::stop_token cancelled) {
    batch_ = 0;
    std::atomic<bool> stop_requested{false};
    std::stop_callback on_stop(cancelled, [&] {
      stop_requested.store(true, std::memory_order_release);
    });

    while (current_frame_ < frame_to_record_ && !cancelled.stop_requested()) {
      const auto *frame = queue_.read_b(&stop_requested);
      if (!frame) {
        if (cancelled.stop_requested())
          break;
        throw std::runtime_error("Camera queue closed before recording completed");
      }
      if (!frame->base)
        throw std::runtime_error("Camera recording buffer has no data");

      const auto to_write = std::min(buffer_part_count_, frame_to_record_ - current_frame_);
      writer_.write_frames(reinterpret_cast<const uint8_t *>(frame->base), to_write);
      queue_.release_b();
      current_frame_ += to_write;
      ++batch_;
      //logger()->debug("[Recorder::execute] batch: {}, current_frame: {}, to_write: {}", batch_, current_frame_, to_write);
    }

    if (!cancelled.stop_requested())
      writer_.write_footer();
    return current_frame_;
  }

private:
  holofile::Writer   writer_;
  size_t             frame_to_record_;
  size_t             batch_;
  size_t             buffer_part_count_;
  size_t             current_frame_;
  CameraBufferQueue &queue_;
};

void recorder_worker(const holotask::sources::RecordSettings &settings,
                     const Recorder::RecordingGeometry &g, size_t buffer_part_count,
                     CameraBufferQueue &queue, std::stop_token cancelled,
                     std::atomic<bool> &recording, std::function<void(size_t)> finished_callback,
                     std::function<void(const std::string &)> failed_callback) {
  logger()->info("[Recorder] started recorder thread");
  try {
    size_t frames_written;
    {
      Recorder rec{settings.file_path, static_cast<uint32_t>(settings.recording_count),
                   buffer_part_count, queue, g, settings.pipeline_config};
      frames_written = rec.execute(cancelled);
    } // Release reader B and close the file before notifying the UI.
    if (!cancelled.stop_requested())
      finished_callback(frames_written);
  } catch (const std::exception &e) {
    logger()->error("[Recorder] recording failed: {}", e.what());
    if (!cancelled.stop_requested()) {
      try {
        failed_callback(e.what());
      } catch (const std::exception &notify_error) {
        logger()->error("[Recorder] failed to report recording error: {}", notify_error.what());
      } catch (...) {
        logger()->error("[Recorder] failed to report recording error");
      }
    }
  } catch (...) {
    logger()->error("[Recorder] recording failed with an unknown error");
    if (!cancelled.stop_requested()) {
      try {
        failed_callback("Unknown camera recording error");
      } catch (...) {
        logger()->error("[Recorder] failed to report recording error");
      }
    }
  }
  recording.store(false, std::memory_order_release);
  logger()->info("[Recorder] stopped recorder thread");
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
                                std::unique_ptr<Euresys::EGenTL> &&gentl,
                                std::unique_ptr<Grabber>         &&grabber_a,
                                std::unique_ptr<Grabber> &&grabber_b, std::size_t buffer_size,
                                nlohmann::json normalized_cfg)
      : settings_(settings), runtime_cfg_(std::move(runtime_cfg)), buffers_(std::move(buffers)),
        gentl_(std::move(gentl)), grabber_a_(std::move(grabber_a)),
        grabber_b_(std::move(grabber_b)), buffer_size_(buffer_size), running_(false),
        cfg_(std::move(normalized_cfg)),
        buffer_queue_(
            runtime_cfg_.nb_buffers,
            [this](const auto &data) {
              requeue_buffer_noexcept(*grabber_a_, data, "bank A", last_requeue_error_log_,
                                      requeue_error_mutex_);
            },
            [this](const auto &data) {
              requeue_buffer_noexcept(*grabber_b_, data, "bank B", last_requeue_error_log_,
                                      requeue_error_mutex_);
            },
            [this](const auto &part0, const auto &part1) {
              auto buf  = Euresys::Buffer(part0);
              auto base = static_cast<std::byte *>(
                  buf.getInfo<void *>(*this->grabber_a_, GenTL::BUFFER_INFO_BASE));
              if (!validate_buffer_data(part0, part1)) {
                requeue_buffer_noexcept(*grabber_a_, part0, "bank A", last_requeue_error_log_,
                                        requeue_error_mutex_);
                requeue_buffer_noexcept(*grabber_b_, part1, "bank B", last_requeue_error_log_,
                                        requeue_error_mutex_);

                return std::optional<CameraBufferQueue::DType>();
              }

              auto res = std::optional<CameraBufferQueue::DType>(
                  {.bank_a = std::move(part0), .bank_b = std::move(part1), .base = base});

              return res;
            }) {
    HOLOVIBES_CHECK(gentl_ != nullptr);
    HOLOVIBES_CHECK(grabber_a_ != nullptr);
    HOLOVIBES_CHECK(grabber_b_ != nullptr);
    HOLOVIBES_CHECK(buffers_ != nullptr);

    grabber_a_->set_enqueue_callback(
        [this](size_t producer_id, const auto &data) { buffer_queue_.push(producer_id, data); });
    grabber_b_->set_enqueue_callback(
        [this](size_t producer_id, const auto &data) { buffer_queue_.push(producer_id, data); });

  }

  ~AmetekS711EuresysCoaxlinkQSFP() override {
    buffer_queue_.close();
    stop_raw_record();
    try {
      if (running_) {
        grabber_a_->stop();
        grabber_b_->stop();
      }
    } catch (const std::exception &e) {
      logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::~AmetekS711EuresysCoaxlinkQSFP] {}",
                     e.what());
    }
  }

  void emit_finished_event(holoflow::core::SyncCtx &ctx, size_t frames_written) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_finished"},
                {"path", settings_.record_settings->file_path},
                {"frames_written", frames_written},
            },
        .ts = std::chrono::steady_clock::now(),
    };
    HOLOVIBES_CHECK(ctx.event_writer->try_push(std::move(event)),
                    "Failed to emit recording_finished event");
  }

  void emit_failed_event(holoflow::core::SyncCtx &ctx, const std::string &message) {
    auto event = holoflow_event::Event{
        .direction = holoflow_event::EventDirection::ToUi,
        .node_id   = "",
        .data =
            nlohmann::json{
                {"type", "recording_failed"},
                {"path", settings_.record_settings->file_path},
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
    stop_raw_record();
    const auto record_settings = *settings_.record_settings;
    recording_.store(true, std::memory_order_release);
    try {
      record_thread_.emplace([this, event_writer = *ctx.event_writer,
                             record_settings](std::stop_token cancelled) mutable {
        holoflow::core::SyncCtx event_ctx{};
        event_ctx.event_writer = &event_writer;
        recorder_worker(record_settings,
                        {static_cast<uint8_t>(runtime_cfg_.bytes_per_pixel * 8),
                         runtime_cfg_.width, runtime_cfg_.final_height},
                        runtime_cfg_.buffer_part_count, buffer_queue_, cancelled, recording_,
                        [this, &event_ctx](size_t written) {
                          emit_finished_event(event_ctx, written);
                        },
                        [this, &event_ctx](const std::string &message) {
                          emit_failed_event(event_ctx, message);
                        });
      });
    } catch (...) {
      recording_.store(false, std::memory_order_release);
      throw;
    }
    logger()->info("[AmetekS711EuresysCoaxlinkQSFP:start_raw_record] started raw record");
  }

  void stop_raw_record() {
    if (record_thread_) {
      record_thread_->request_stop();
      record_thread_->join();
      record_thread_.reset();
    }
    recording_.store(false, std::memory_order_release);
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

    while (true) {
      auto event = ctx.event_reader->try_pop();
      if (!event.has_value())
        break;

      HOLOVIBES_CHECK(event->direction == holoflow_event::EventDirection::ToNode,
                      "Unexpected event direction");

      const auto type = event->data.at("type").get<std::string>();

      logger()->debug("[AmetekS711EuresysCoaxlinkQSFP::handle_events] received event: {}", type);
      if (type == "start_recording") {
        if (recording_.load(std::memory_order_acquire)) {
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::handle_events] Ignoring duplicate "
                          "start_recording event");
          emit_failed_event(ctx, "Recording already in progress");
          continue;
        }

        const auto record_path = event->data.value("record_path", std::string{});
        if (record_path.empty()) {
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::handle_events] Rejecting "
                          "start_recording event with empty path");
          emit_failed_event(ctx, "Cannot start recording: empty path");
          continue;
        }

        if (settings_.record_settings->recording_count <= 0) {
          const auto message = "Cannot start recording: invalid frame count (" +
                               std::to_string(settings_.record_settings->recording_count) + ")";
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::handle_events] {}", message);
          emit_failed_event(ctx, message);
          continue;
        }

        // A finished worker may still be delivering its completion event.
        stop_raw_record();
        settings_.record_settings->file_path = record_path;
        try {
          start_raw_record(ctx);
        } catch (const std::exception &e) {
          emit_failed_event(ctx, e.what());
        }

      } else if (type == "stop_recording") {
        if (!recording_.load(std::memory_order_acquire)) {
          logger()->warn("[AmetekS711EuresysCoaxlinkQSFP::handle_events] Ignoring stop_recording "
                         "event while idle");
          continue;
        }

        stop_raw_record();

        if (!settings_.record_settings->file_path.empty() &&
            std::remove(settings_.record_settings->file_path.c_str()) != 0 && errno != ENOENT) {
          std::error_code ec(errno, std::generic_category());
          logger()->error("[AmetekS711EuresysCoaxlinkQSFP::handle_events] Failed to remove "
                          "incomplete recording at {}: {}",
                          settings_.record_settings->file_path, ec.message());
          emit_failed_event(ctx, "Failed to remove incomplete recording at " +
                                     settings_.record_settings->file_path + ": " + ec.message());
        }

      } else {
        HOLOVIBES_BUG("Unknown event type: {}", type);
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
      logger()->error("[AmetekS711EuresysCoaxlinkQSFP::update] updating with unreleased frames: {}",
                      buffer_queue_.size());
    }
    if (!running_) {
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

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (settings_.record_settings.has_value())
      handle_events(ctx);

    if (!running_) {
      // S711 Banks_AB must start bank B first, then bank A.
      grabber_b_->start();
      grabber_a_->start();
      running_ = true;
    }

    while (!ctx.cancelled->load()) {
      const auto *next = buffer_queue_.read_a(ctx.cancelled); // return null if cancelled
      if (next == nullptr) {
        return holoflow::core::OpResult::Cancelled;
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
    return holoflow::core::OpResult::Cancelled;
  }

  const nlohmann::json &get_cfg() const { return cfg_; }

private:
  bool validate_buffer_data(const Euresys::NewBufferData &data_a,
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
        return false;
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

      return true;
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
    return false;
  }

private:
  AmetekS711EuresysCoaxlinkQSFPSettings settings_;
  RuntimeConfig                         runtime_cfg_;
  HostPtr<uint8_t>                      buffers_;
  std::unique_ptr<Euresys::EGenTL>      gentl_;
  std::unique_ptr<Grabber>              grabber_a_;
  std::unique_ptr<Grabber>              grabber_b_;
  std::size_t                           buffer_size_;
  bool                                  running_;
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

  CameraBufferQueue           buffer_queue_;
  std::atomic<bool>           recording_ = false;
  std::optional<std::jthread> record_thread_;
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

  // TODO change to Grabbers
  auto grabber_a =
      std::make_unique<Grabber>(camera_info->grabbers[bank_a_index], Grabber::EGrabberName::A);
  auto grabber_b =
      std::make_unique<Grabber>(camera_info->grabbers[bank_b_index], Grabber::EGrabberName::B);

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
    old->update_settings(settings);
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
