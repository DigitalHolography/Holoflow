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

#ifdef HOLOTASK_HAS_EGRABBER

#include <EGrabber.h>
#include <EuresysGenapiErrorFormats.h>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>

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

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[AmetekS711EuresysCoaxlinkQSFPFactory] error: {}", msg);
    throw std::invalid_argument("AmetekS711EuresysCoaxlinkQSFPFactory error: " + msg);
  }
}

std::string format_genapi_error(const Euresys::genapi_error &err) {
  std::ostringstream oss;
  oss << "GenApi error: code=" << err.genapi_error_code << ", what=\"" << err.what() << "\"";

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
  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] configuring grabber");

  // Bank mapping: 2-grabber vs 4-grabber configurations
  const std::map<std::size_t, std::string> banks_map = {
      {2, "Banks_AB"},
      {4, "Banks_ABCD"},
  };

  // Extract configuration parameters
  const auto width                 = cfg.at("Width").get<std::size_t>();
  const auto height                = cfg.at("Height").get<std::size_t>();
  const auto nb_grabbers           = info.grabbers.size();
  const auto pixel_format          = cfg.at("PixelFormat").get<std::string>();
  const auto trigger_source        = cfg.at("TriggerSource").get<std::string>();
  const auto trigger_mode          = cfg.at("TriggerMode").get<std::string>();
  const auto exposure_time         = cfg.at("ExposureTime").get<std::size_t>();
  const auto cycle_min_period      = cfg.at("CycleMinimumPeriod").get<std::size_t>();
  const auto gain_selector         = cfg.at("GainSelector").get<std::string>();
  const auto gain                  = cfg.at("Gain").get<float>();
  const auto balance_white_marker  = cfg.at("BalanceWhiteMarker").get<std::string>();
  const auto flat_field_correction = cfg.at("FlatFieldCorrection").get<std::string>();
  const auto buffer_part_count     = cfg.at("BufferPartCount").get<std::size_t>();

  // Computed parameters
  const auto stripe_height            = height / nb_grabbers;
  const auto stripe_pitch             = height;
  const auto block_height             = stripe_height;
  const auto camera_control_method    = "RC";
  const auto exposure_readout_overlap = "True";
  const auto error_selector           = "All";
  const auto stripe_arrangement       = "Geometry_1X_2YM";
  const auto statistics_sampling_sel  = "LastSecond";
  const auto lut_configuration        = "M_10x8";
  const auto banks                    = banks_map.at(nb_grabbers);

  try {
    // Device-level master configuration
    auto g = Euresys::EGrabber(info);
    g.execute<Euresys::DeviceModule>("DeviceReset");
    g.setString<Euresys::RemoteModule>("Banks", banks);
    g.setString<Euresys::DeviceModule>("CameraControlMethod", camera_control_method);

    if (trigger_source == "SWTRIGGER") {
      g.setString<Euresys::DeviceModule>("ErrorSelector", error_selector);
      g.setInteger<Euresys::DeviceModule>("CycleMinimumPeriod", cycle_min_period);
      g.setString<Euresys::DeviceModule>("ExposureReadoutOverlap", exposure_readout_overlap);
    }

    // Remote configuration
    g.setInteger<Euresys::RemoteModule>("Width", width);
    g.setInteger<Euresys::RemoteModule>("Height", height / nb_grabbers);
    g.setString<Euresys::RemoteModule>("PixelFormat", pixel_format);
    g.setString<Euresys::RemoteModule>("TriggerMode", trigger_mode);
    g.setString<Euresys::RemoteModule>("TriggerSource", trigger_source);
    g.setInteger<Euresys::RemoteModule>("ExposureTime", exposure_time);
    g.setString<Euresys::RemoteModule>("BalanceWhiteMarker", balance_white_marker);
    g.setString<Euresys::RemoteModule>("GainSelector", gain_selector);
    g.setFloat<Euresys::RemoteModule>("Gain", gain);
    g.setString<Euresys::RemoteModule>("FlatFieldCorrection", flat_field_correction);

    // Stream configuration
    g.setString<Euresys::StreamModule>("StripeArrangement", stripe_arrangement);
    g.setInteger<Euresys::StreamModule>("LinePitch", width);
    g.setInteger<Euresys::StreamModule>("LineWidth", width);
    g.setInteger<Euresys::StreamModule>("StripeHeight", stripe_height);
    g.setInteger<Euresys::StreamModule>("StripePitch", stripe_pitch);
    g.setInteger<Euresys::StreamModule>("BlockHeight", block_height);
    g.setString<Euresys::StreamModule>("LUTConfiguration", lut_configuration);
    g.setInteger<Euresys::StreamModule>("BufferPartCount", buffer_part_count);
    g.setString<Euresys::StreamModule>("StatisticsSamplingSelector", statistics_sampling_sel);
  } catch (const Euresys::genapi_error &e) {
    throw std::runtime_error(format_genapi_error(e));
  }
}

// Allocate nb_buffers pinned host buffers of buffer_size bytes each, announce
// them all to the grabber, and return the backing allocation.  The grabber
// queues each buffer immediately so the camera can start filling them.
HostPtr<uint8_t> allocate_buffers(Euresys::EGrabber<> &g, std::size_t nb_buffers,
                                  std::size_t buffer_size) {
  logger()->info("[AmetekS711EuresysCoaxlinkQSFPFactory] allocating {} buffers of size {} bytes",
                 nb_buffers, buffer_size);
  const auto size    = buffer_size * nb_buffers;
  auto       buffers = curaii::make_unique_host_ptr<uint8_t>(size);

  for (size_t buf_idx = 0; buf_idx < nb_buffers; ++buf_idx) {
    auto *buff_ptr = buffers.get() + buf_idx * buffer_size;
    auto  memory   = Euresys::UserMemory(buff_ptr, buffer_size);
    g.announceAndQueue(memory);
    logger()->debug("[AmetekS711EuresysCoaxlinkQSFPFactory] announced and queued buffer {} at address {}", buf_idx, static_cast<void *>(buff_ptr));
  }

  return buffers;
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Task implementation (private)
// -------------------------------------------------------------------------------------------------

class AmetekS711EuresysCoaxlinkQSFP : public holoflow::core::ISyncTask {
public:
  AmetekS711EuresysCoaxlinkQSFP(const AmetekS711EuresysCoaxlinkQSFPSettings &settings,
                                HostPtr<uint8_t>                           &&buffers,
                                std::unique_ptr<Euresys::EGenTL>           &&gentl,
                                std::unique_ptr<Euresys::EGrabber<>>       &&grabber,
                                std::size_t buffer_size, const nlohmann::json &cfg)
      : settings_(settings), buffers_(std::move(buffers)), gentl_(std::move(gentl)),
        grabber_(std::move(grabber)), buffer_size_(buffer_size), running_(false), cfg_(cfg),
        pending_data_(std::nullopt) {
    HOLOVIBES_CHECK(grabber_ != nullptr);
    HOLOVIBES_CHECK(gentl_ != nullptr);
    HOLOVIBES_CHECK(buffers_ != nullptr);
  }

  // ------------------------------------------------------------------------------------------------
  // ISyncTask interface
  // ------------------------------------------------------------------------------------------------

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    (void)index;
    throw std::out_of_range("AmetekS711EuresysCoaxlinkQSFP task has no inputs");
  }

  // Called by the framework after the downstream consumer is done with output 0.
  // We re-queue the buffer so the grabber can DMA the next frame into it.
  void release_output(int index) override {
    if (index != 0) {
      throw std::out_of_range("AmetekS711EuresysCoaxlinkQSFP task has only one output at index 0");
    }
    if (!pending_data_.has_value()) {
      throw std::logic_error("release_output called with no pending buffer");
    }

    // Reconstruct a Buffer from the stored NewBufferData and push it back to the grabber's
    // input FIFO so the camera can DMA the next frame into it.
    Euresys::Buffer(*pending_data_).push(*grabber_);
    pending_data_.reset();
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    using namespace Euresys;
    constexpr auto DELIVERED = ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS;
    constexpr auto TIMESTAMP = GenTL::BUFFER_INFO_TIMESTAMP;

    HOLOVIBES_CHECK(!pending_data_.has_value(),
                    "execute called while previous buffer is still held");

    if (!running_) {
      grabber_->start();
      running_ = true;
    }

    while (!ctx.cancelled->load()) {
      try {
        // pop() returns NewBufferData without re-queuing. We wrap it in a
        // temporary Buffer only for the getInfo queries, then store the
        // NewBufferData so release_output can push it back later.
        auto data   = grabber_->pop(1000);
        auto buffer = Buffer(data);

        auto delivered = buffer.getInfo<uint64_t>(*grabber_, DELIVERED);
        auto ts        = buffer.getInfo<uint64_t>(*grabber_, TIMESTAMP);

        logger()->trace(
            "[AmetekS711EuresysCoaxlinkQSFP::execute] buffer with {} parts, timestamp {}",
            delivered, ts);

        // buffer.getInfo<std::byte *>(...) produces a linker error.
        auto *frame_data_v = buffer.getInfo<void *>(*grabber_, GenTL::BUFFER_INFO_BASE);
        auto *frame_data   = static_cast<std::byte *>(frame_data_v);

        // Point the output tensor directly at the DMA'd memory — no memcpy.
        auto &storage = storage_access().owned_output_storage(0);
        storage.ptr   = frame_data;

        ctx.outputs[0] = holoflow::core::TView{
            .desc    = ctx.outputs[0].desc,
            .storage = &storage,
        };

        // Keep the NewBufferData alive so release_output can push the buffer back.
        pending_data_ = data;
        return holoflow::core::OpResult::Ok;

      } catch (const Euresys::genapi_error &err) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::execute] GenApi error: {}", err.what());
      } catch (const Euresys::gentl_error &err) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::execute] GenTL error: {}", err.what());
      } catch (const std::exception &err) {
        logger()->error("[AmetekS711EuresysCoaxlinkQSFP::execute] error: {}", err.what());
      }
    }
    return holoflow::core::OpResult::Cancelled;
  }

  // Expose config for update() comparison
  const nlohmann::json &get_cfg() const { return cfg_; }

private:
  AmetekS711EuresysCoaxlinkQSFPSettings settings_;
  HostPtr<uint8_t>                      buffers_;
  std::unique_ptr<Euresys::EGenTL>      gentl_;
  std::unique_ptr<Euresys::EGrabber<>>  grabber_;
  std::size_t                           buffer_size_;
  bool                                  running_;
  nlohmann::json                        cfg_;

  // Holds the NewBufferData of the frame currently exposed as output 0.
  // Cleared in release_output() after pushing the buffer back to the grabber.
  std::optional<Euresys::NewBufferData> pending_data_;
};

// -------------------------------------------------------------------------------------------------
// Factory implementation
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
AmetekS711EuresysCoaxlinkQSFPFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                            const nlohmann::json &jsettings) const {
  // clang-format off
  check(input_descs.size() == 0, "expected zero input tensors");

  auto settings = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  check(!settings.cfg_path.empty(), "cfg_path is empty");

  std::ifstream cfg_file(settings.cfg_path);
  check(cfg_file.is_open(), std::format("could not open config file: {}", settings.cfg_path));

  auto        cfg    = nlohmann::json::parse(cfg_file).at("s711");
  std::string format = cfg.at("PixelFormat");
  // clang-format on

  static const std::map<std::string, holoflow::core::DType> dtypes = {
      {"Mono8", holoflow::core::DType::U8},
      {"Mono16", holoflow::core::DType::U16},
  };

  check(dtypes.contains(format), "unsupported PixelFormat: " + format);

  const auto batch_size = cfg.at("BufferPartCount").get<size_t>();
  const auto height     = cfg.at("Height").get<size_t>();
  const auto width      = cfg.at("Width").get<size_t>();
  const auto dtype      = dtypes.at(format);
  const auto loc        = holoflow::core::MemLoc::Host;

  holoflow::core::TDesc odesc({batch_size, height, width}, dtype, loc);

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

  auto settings = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  auto cfg_file = std::ifstream(settings.cfg_path);
  auto cfg      = nlohmann::json::parse(cfg_file).at("s711");

  // Setup GenTL
  auto gentl       = std::make_unique<Euresys::EGenTL>();
  auto camera_info = find_camera(*gentl, "Phantom S711");

  check(camera_info.has_value(), "could not find Phantom S711 camera");

  configure_grabber(*camera_info, cfg);
  auto grabber      = std::make_unique<Euresys::EGrabber<>>(*camera_info);
  auto infer_result = this->infer(input_descs, jsettings);
  auto buffer_size  = infer_result.output_descs[0].num_bytes();
  auto nb_buffers   = cfg.at("NbBuffers").get<std::size_t>();
  auto buffers      = allocate_buffers(*grabber, nb_buffers, buffer_size);

  return std::make_unique<AmetekS711EuresysCoaxlinkQSFP>(
      settings, std::move(buffers), std::move(gentl), std::move(grabber), buffer_size, cfg);
}

std::unique_ptr<holoflow::core::ISyncTask>
AmetekS711EuresysCoaxlinkQSFPFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                             std::span<const holoflow::core::TDesc>     input_descs,
                                             const nlohmann::json                      &jsettings,
                                             const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)ctx;

  auto *old = dynamic_cast<AmetekS711EuresysCoaxlinkQSFP *>(old_task.get());
  if (old == nullptr || input_descs.size() != 0) {
    return create(input_descs, jsettings, ctx);
  }

  auto settings = jsettings.get<AmetekS711EuresysCoaxlinkQSFPSettings>();
  auto cfg_file = std::ifstream(settings.cfg_path);
  auto cfg      = nlohmann::json::parse(cfg_file).at("s711");

  if (cfg == old->get_cfg()) {
    return old_task; // Reuse if config unchanged
  }

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