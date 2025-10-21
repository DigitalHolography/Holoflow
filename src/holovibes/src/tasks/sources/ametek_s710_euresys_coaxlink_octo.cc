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

#include "ametek_s710_euresys_coaxlink_octo.hh"

#include <EGrabber.h>
#include <EuresysGenapiErrorFormats.h>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks::sources {

void to_json(nlohmann::json &j, const AmetekS710EuresysCoaxlinkOctoSettings &s) {
  j = nlohmann::json{
      {"cfg_path", s.cfg_path},
  };
}

void from_json(const nlohmann::json &j, AmetekS710EuresysCoaxlinkOctoSettings &s) {
  j.at("cfg_path").get_to(s.cfg_path);
}

AmetekS710EuresysCoaxlinkOcto::AmetekS710EuresysCoaxlinkOcto(
    const AmetekS710EuresysCoaxlinkOctoSettings &settings, HostPtr<uint8_t> &&buffers,
    std::unique_ptr<Euresys::EGenTL> &&gentl, std::unique_ptr<Euresys::EGrabber<>> &&grabber)
    : settings_(settings), buffers_(std::move(buffers)), gentl_(std::move(gentl)),
      grabber_(std::move(grabber)), running_(false) {
  HOLOVIBES_CHECK(grabber_ != nullptr);
  HOLOVIBES_CHECK(gentl_ != nullptr);
  HOLOVIBES_CHECK(buffers_ != nullptr);
}

holoflow::core::OpResult AmetekS710EuresysCoaxlinkOcto::execute(holoflow::core::SyncCtx &ctx) {
  using namespace Euresys;
  constexpr auto DELIVERED = ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS;
  constexpr auto TIMESTAMP = GenTL::BUFFER_INFO_TIMESTAMP;

  if (!running_) {
    grabber_->start();
    running_ = true;
  }

  while (!ctx.cancelled->load()) {
    try {

      auto buffer    = ScopedBuffer(*grabber_, 1000);
      auto delivered = buffer.getInfo<uint64_t>(DELIVERED);
      auto ts        = buffer.getInfo<uint64_t>(TIMESTAMP);

      logger()->trace("[AmetekS710EuresysCoaxlinkOcto] Acquired buffer with {} parts, timestamp {}",
                      delivered, ts);

      const auto *idata = buffer.getInfo<void *>(GenTL::BUFFER_INFO_BASE);
      auto       *odata = ctx.outputs[0].data;
      std::memcpy(odata, idata, ctx.outputs[0].desc.num_bytes());
      return holoflow::core::OpResult::Ok;
    } catch (const Euresys::genapi_error &err) {
      logger()->error("[AmetekS710EuresysCoaxlinkOcto] GenApi error while acquiring buffer: {}",
                      err.what());
    } catch (const Euresys::gentl_error &err) {
      logger()->error("[AmetekS710EuresysCoaxlinkOcto] GenTL error while acquiring buffer: {}",
                      err.what());
    } catch (const std::exception &err) {
      logger()->error("[AmetekS710EuresysCoaxlinkOcto] Error while acquiring buffer: {}",
                      err.what());
    }
  }
  return holoflow::core::OpResult::Cancelled;
}

namespace {

std::string format_genapi_error(const Euresys::genapi_error &err) {
  std::ostringstream oss;

  oss << "GenApi error: code=" << err.genapi_error_code << " (" << err.genapi_error_code
      << "), what=\"" << err.what() << "\"";

  size_t count = err.parameter_count();
  if (count > 0) {
    oss << ", parameters=[";
    for (size_t i = 0; i < count; ++i) {
      oss << "{";
      auto type = err.parameter_type(i);
      switch (type) {
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
      if (i + 1 < count)
        oss << ", ";
    }
    oss << "]";
  }

  return oss.str();
}

std::optional<Euresys::EGrabberCameraInfo> find_camera(Euresys::EGenTL   &gentl,
                                                       const std::string &camera_name) {
  using namespace Euresys;

  EGrabberDiscovery discovery(gentl);
  discovery.discover();
  for (int i = 0; i < discovery.cameraCount(); ++i) {
    auto info = discovery.cameras(i);
    auto g    = EGrabber(info);
    auto name = g.getString<DeviceModule>("DeviceModelName");
    if (name == camera_name) {
      return info;
    }
  }

  return std::nullopt;
}

void configure_grabber(Euresys::EGrabberCameraInfo &info, const nlohmann::json &cfg) {
  using namespace Euresys;

  // Utilies to deduce configuration
  const std::map<std::size_t, std::string> banks_map = {
      {2, "Banks_AB"},
      {4, "Banks_ABCD"},
  };

  // Retrieve configuration parameters
  const std::size_t width                        = cfg.at("Width");
  const std::size_t height                       = cfg.at("Height");
  const std::size_t nb_grabbers                  = info.grabbers.size();
  const std::string pixel_format                 = cfg.at("PixelFormat");
  const std::string trigger_source               = cfg.at("TriggerSource");
  const std::string trigger_mode                 = cfg.at("TriggerMode");
  const std::size_t stripe_height                = height / nb_grabbers;
  const std::size_t stripe_pitch                 = height;
  const std::size_t block_height                 = stripe_height;
  const std::string camera_control_method        = "RC";
  const std::size_t exposure_time                = cfg.at("ExposureTime");
  const std::size_t cycle_min_period             = cfg.at("CycleMinimumPeriod");
  const std::string gain_selector                = cfg.at("GainSelector");
  const float       gain                         = cfg.at("Gain");
  const std::string balance_white_marker         = cfg.at("BalanceWhiteMarker");
  const std::string flat_field_correction        = cfg.at("FlatFieldCorrection");
  const std::size_t buffer_part_count            = cfg.at("BufferPartCount");
  const std::string banks                        = banks_map.at(nb_grabbers);
  const std::string exposure_readout_overlap     = "True";
  const std::string error_selector               = "All";
  const std::string stripe_arrangement           = "Geometry_1X_2YM";
  const std::string statistics_sampling_selector = "LastSecond";
  const std::string lut_configuration            = "M_10x8";

  try {
    // Device-level master configuration
    auto g = EGrabber(info);
    g.execute<DeviceModule>("DeviceReset");
    g.setString<RemoteModule>("Banks", banks);
    g.setString<DeviceModule>("CameraControlMethod", camera_control_method);
    if (trigger_source == "SWTRIGGER") {
      g.setString<DeviceModule>("ErrorSelector", error_selector);
      g.setInteger<DeviceModule>("CycleMinimumPeriod", cycle_min_period);
      g.setString<DeviceModule>("ExposureReadoutOverlap", exposure_readout_overlap);
    }

    // Remote configuration
    g.setInteger<RemoteModule>("Width", width);
    g.setInteger<RemoteModule>("Height", height / nb_grabbers);
    g.setString<RemoteModule>("PixelFormat", pixel_format);
    g.setString<RemoteModule>("TriggerMode", trigger_mode);
    g.setString<RemoteModule>("TriggerSource", trigger_source);
    g.setInteger<RemoteModule>("ExposureTime", exposure_time);
    g.setString<RemoteModule>("BalanceWhiteMarker", balance_white_marker);
    g.setString<RemoteModule>("GainSelector", gain_selector);
    g.setFloat<RemoteModule>("Gain", gain);
    g.setString<RemoteModule>("FlatFieldCorrection", flat_field_correction);

    // Stream configuration
    g.setString<StreamModule>("StripeArrangement", stripe_arrangement);
    g.setInteger<StreamModule>("LinePitch", width);
    g.setInteger<StreamModule>("LineWidth", width);
    g.setInteger<StreamModule>("StripeHeight", stripe_height);
    g.setInteger<StreamModule>("StripePitch", stripe_pitch);
    g.setInteger<StreamModule>("BlockHeight", block_height);
    g.setString<StreamModule>("LUTConfiguration", lut_configuration);
    g.setInteger<StreamModule>("BufferPartCount", buffer_part_count);
    g.setString<StreamModule>("StatisticsSamplingSelector", statistics_sampling_selector);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(format_genapi_error(e));
  }
}

HostPtr<uint8_t> allocate_buffers(Euresys::EGrabber<> &g, std::size_t nb_buffers,
                                  std::size_t buffer_size) {
  const auto size    = buffer_size * nb_buffers;
  auto       buffers = curaii::make_unique_host_ptr<uint8_t>(size);

  for (int buf_idx = 0; buf_idx < nb_buffers; ++buf_idx) {
    auto *buff_ptr = buffers.get() + buf_idx * buffer_size;
    auto  memory   = Euresys::UserMemory(buff_ptr, buffer_size);
    g.announceAndQueue(memory);
  }

  return buffers;
}

} // namespace

holoflow::core::InferResult
AmetekS710EuresysCoaxlinkOctoFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                            const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[AmetekS710EuresysCoaxlinkOctoFactory::infer] error: {}", msg);
      throw std::invalid_argument("AmetekS710EuresysCoaxlinkOctoFactory inference error: " + msg);
    }
  };

  std::map<std::string, holoflow::core::DType> dtypes = {
      {"Mono8", holoflow::core::DType::U8},
      {"Mono16", holoflow::core::DType::U16},
  };

  auto settings = jsettings.get<AmetekS710EuresysCoaxlinkOctoSettings>();

  // Validate
  check(input_descs.size() == 0, "Expected zero input tensors");
  check(!settings.cfg_path.empty(), "cfg_path is empty");
  std::ifstream cfg_file(settings.cfg_path);
  auto          error = std::strerror(errno);
  auto error_msg      = std::format("Could not open config file: {}, {}", settings.cfg_path, error);
  check(cfg_file.is_open(), error_msg);
  auto        cfg    = nlohmann::json::parse(cfg_file).at("s710");
  std::string format = cfg.at("PixelFormat");
  check(dtypes.contains(format), "Unsupported PixelFormat: " + format);

  // Success
  size_t                width      = cfg.at("Width");
  size_t                height     = cfg.at("Height");
  size_t                batch_size = cfg.at("BufferPartCount");
  holoflow::core::DType dtype      = dtypes.at(format);
  auto                  loc        = holoflow::core::MemLoc::Host;
  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {holoflow::core::TDesc{
           .shape   = {batch_size, height, width},
           .dtype   = dtype,
           .mem_loc = loc,
      }},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS710EuresysCoaxlinkOctoFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                             const nlohmann::json                  &jsettings,
                                             const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;

  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<AmetekS710EuresysCoaxlinkOctoSettings>();
  auto cfg_file = std::ifstream(settings.cfg_path);
  auto cfg      = nlohmann::json::parse(cfg_file).at("s710");

  // Setup GenTL
  auto gentl       = std::make_unique<Euresys::EGenTL>();
  auto camera_info = find_camera(*gentl, "Phantom S710");

  if (!camera_info.has_value()) {
    logger()->error("[AmetekS710EuresysCoaxlinkOctoFactory::create] Could not find Phantom S710");
    throw std::runtime_error("Could not find Phantom S710 camera");
  }

  configure_grabber(*camera_info, cfg);
  auto grabber     = std::make_unique<Euresys::EGrabber<>>(*camera_info);
  auto buffer_size = infer.output_descs[0].num_bytes();
  auto buffers     = allocate_buffers(*grabber, cfg.at("BufferPartCount"), buffer_size);

  // Success
  auto *task = new AmetekS710EuresysCoaxlinkOcto(settings, std::move(buffers), std::move(gentl),
                                                 std::move(grabber));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS710EuresysCoaxlinkOctoFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                             std::span<const holoflow::core::TDesc>     input_descs,
                                             const nlohmann::json                      &jsettings,
                                             const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)ctx;

  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<AmetekS710EuresysCoaxlinkOctoSettings>();
  auto cfg_file = std::ifstream(settings.cfg_path);
  auto cfg      = nlohmann::json::parse(cfg_file).at("s710");

  auto *old = dynamic_cast<AmetekS710EuresysCoaxlinkOcto *>(old_task.get());
  HOLOVIBES_CHECK(old != nullptr, "Old task is not of type AmetekS710EuresysCoaxlinkOcto");
  auto old_cfg_file = std::ifstream(old->settings_.cfg_path);
  auto old_cfg      = nlohmann::json::parse(old_cfg_file).at("s710");

  auto same_cfg      = cfg == old_cfg;
  auto reuse_grabber = same_cfg;
  auto reuse_buffers = same_cfg;

  auto gentl   = std::move(old->gentl_);
  auto grabber = std::move(old->grabber_);
  auto buffers = std::move(old->buffers_);

  if (!reuse_grabber) {
    logger()->info("[AmetekS710EuresysCoaxlinkOctoFactory::update] Recreating grabber");
    auto camera_info = find_camera(*gentl, "Phantom S710");
    if (!camera_info.has_value()) {
      logger()->error("[AmetekS710EuresysCoaxlinkOctoFactory::update] Could not find Phantom S710");
      throw std::runtime_error("Could not find Phantom S710 camera");
    }
    configure_grabber(*camera_info, cfg);
  }

  if (!reuse_buffers) {
    logger()->info("[AmetekS710EuresysCoaxlinkOctoFactory::update] Reallocating buffers");
    auto buffer_size = infer.output_descs[0].num_bytes();
    buffers          = allocate_buffers(*grabber, cfg.at("BufferPartCount"), buffer_size);
  }

  // Success
  auto *task = new AmetekS710EuresysCoaxlinkOcto(settings, std::move(buffers), std::move(gentl),
                                                 std::move(grabber));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks::sources