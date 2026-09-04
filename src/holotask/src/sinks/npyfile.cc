#include "holotask/sinks/npyfile.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "logger.hh"

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow_event/router.hh"
#include "recording_geometry.hh"

namespace npyfile {
namespace {
void write_checked(FILE *file, const void *data, std::size_t size, const char *what) {
  if (std::fwrite(data, 1, size, file) != size)
    throw std::runtime_error(std::string("Failed to write NPY ") + what);
}

std::string header_text(const Header &header) {
  std::string result = "{'descr': '" + header.dtype + "', 'fortran_order': False, 'shape': (";
  for (std::size_t i = 0; i < header.shape.size(); ++i) {
    if (i)
      result += ", ";
    result += std::to_string(header.shape[i]);
  }
  if (header.shape.size() == 1)
    result += ",";
  result += "), }";
  const auto padding = (16 - ((10 + result.size() + 1) % 16)) % 16;
  result.append(padding, ' ');
  result.push_back('\n');
  return result;
}
} // namespace

Writer::Writer(const std::string &path, const Header &header) {
  file_ = std::fopen(path.c_str(), "wb");
  if (!file_)
    throw std::runtime_error("Failed to open NPY file for writing: " + path);
  try {
    constexpr std::uint8_t preamble[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    write_checked(file_, preamble, sizeof(preamble), "preamble");
    const auto text = header_text(header);
    if (text.size() > UINT16_MAX)
      throw std::invalid_argument("NPY header is too large");
    const auto         length         = static_cast<std::uint16_t>(text.size());
    const std::uint8_t length_bytes[] = {static_cast<std::uint8_t>(length),
                                         static_cast<std::uint8_t>(length >> 8)};
    write_checked(file_, length_bytes, sizeof(length_bytes), "header length");
    write_checked(file_, text.data(), text.size(), "header");
  } catch (...) {
    std::fclose(file_);
    file_ = nullptr;
    throw;
  }
}

Writer::~Writer() {
  if (file_)
    std::fclose(file_);
}

void Writer::write_frames(const void *data, std::size_t count, std::size_t frame_size) {
  write_checked(file_, data, count * frame_size, "data");
}
} // namespace npyfile

namespace holotask::sinks {
namespace {
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

void copy_bytes(void *dst, const void *src, std::size_t size) { std::memcpy(dst, src, size); }

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("NpyfileFactory error: " + message);
}

class NpyfileWriter final : public holoflow::core::ISyncTask {
public:
  NpyfileWriter(NpyfileSettings settings, detail::RecordingGeometry geometry,
                std::size_t frame_size)
      : settings_(std::move(settings)), geometry_(geometry), frame_size_(frame_size),
        ring_(curaii::make_unique_host_ptr<std::uint8_t>(static_cast<std::size_t>(settings_.count) *
                                                         frame_size)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_events(ctx);
    if (!recording_)
      return holoflow::core::OpResult::Ok;
    const auto batch = static_cast<int>(ctx.inputs[0].desc.shape[0]);
    const auto count = std::min(settings_.count - buffered_, batch);
    copy_bytes(ring_.get() + static_cast<std::size_t>(buffered_) * frame_size_,
               ctx.inputs[0].data(), static_cast<std::size_t>(count) * frame_size_);
    buffered_ += count;
    if (buffered_ >= settings_.count) {
      logger()->info("[NpyfileWriter] Frame buffer full ({} frames); flushing to disk...",
                     settings_.count);
      try {
        npyfile::Writer writer(settings_.path,
                               {.shape = {static_cast<std::size_t>(settings_.count),
                                          geometry_.frame_height, geometry_.frame_width},
                                .dtype = geometry_.bits_per_pixel == 8 ? "<u1" : "<u2"});
        writer.write_frames(ring_.get(), static_cast<std::size_t>(settings_.count), frame_size_);
        emit(ctx, "recording_finished", {});
        logger()->info("[NpyfileWriter] Recording complete: {} frames written to {}",
                       buffered_, settings_.path);
      } catch (const std::exception &error) {
        std::remove(settings_.path.c_str());
        emit(ctx, "recording_failed", {{"message", error.what()}});
        logger()->error("[NpyfileWriter] Failed to finalize recording at {}: {}", settings_.path,
                        error.what());
      }
      recording_ = false;
      buffered_  = 0;
    }
    return holoflow::core::OpResult::Ok;
  }

  bool can_reuse(const NpyfileSettings &settings, detail::RecordingGeometry geometry,
                 std::size_t frame_size) const {
    return settings.use_buffer == settings_.use_buffer && settings.count == settings_.count &&
           detail::same_geometry(geometry, geometry_) && frame_size == frame_size_;
  }
  void update_settings(NpyfileSettings settings) {
    const auto path = settings_.path;
    settings_       = std::move(settings);
    if (recording_)
      settings_.path = path;
  }

private:
  void handle_events(holoflow::core::SyncCtx &ctx) {
    if (!ctx.event_reader)
      return;
    while (auto event = ctx.event_reader->try_pop()) {
      const auto type = event->data.at("type").get<std::string>();
      if (type == "start_recording") {
        if (recording_) {
          emit(ctx, "recording_failed", {{"message", "Recording already in progress"}});
          continue;
        }
        settings_.path = event->data.value("record_path", std::string{});
        if (settings_.path.empty()) {
          emit(ctx, "recording_failed", {{"message", "Cannot start recording: empty path"}});
          continue;
        }
        recording_ = true;
        buffered_  = 0;
      } else if (type == "stop_recording") {
        if (recording_ && !settings_.path.empty() && std::remove(settings_.path.c_str()) != 0 &&
            errno != ENOENT)
          emit(ctx, "recording_failed", {{"message", "Failed to remove incomplete recording"}});
        recording_ = false;
        buffered_  = 0;
      } else {
        HOLOVIBES_BUG("Unknown event type: {}", type);
      }
    }
  }
  void emit(holoflow::core::SyncCtx &ctx, const char *type, nlohmann::json data) {
    if (!ctx.event_writer)
      return;
    data["type"] = type;
    data["path"] = settings_.path;
    if (std::string(type) == "recording_finished")
      data["frames_written"] = settings_.count;
    (void)ctx.event_writer->try_push({.direction = holoflow_event::EventDirection::ToUi,
                                      .node_id   = "",
                                      .data      = std::move(data),
                                      .ts        = std::chrono::steady_clock::now()});
  }

  NpyfileSettings           settings_;
  detail::RecordingGeometry geometry_;
  std::size_t               frame_size_;
  HostPtr<std::uint8_t>     ring_;
  bool                      recording_ = false;
  int                       buffered_  = 0;
};
} // namespace

void to_json(nlohmann::json &j, const NpyfileSettings &s) {
  j = {{"path", s.path}, {"count", s.count}, {"use_buffer", s.use_buffer}};
}
void from_json(const nlohmann::json &j, NpyfileSettings &s) {
  j.at("path").get_to(s.path);
  j.at("count").get_to(s.count);
  if (j.contains("use_buffer"))
    j.at("use_buffer").get_to(s.use_buffer);
}

holoflow::core::InferResult NpyfileFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                  const nlohmann::json &json) const {
  const auto s = json.get<NpyfileSettings>();
  check(s.use_buffer, "use_buffer must be true");
  check(s.count > 0, "count must be positive");
  check(inputs.size() == 1, "expected exactly one input tensor");
  const auto &input = inputs[0];
  check(input.mem_loc == holoflow::core::MemLoc::Host, "input tensor must be in Host memory");
  check(input.shape.size() == 3, "input tensor must have rank 3 (batch, height, width)");
  check(input.dtype == holoflow::core::DType::U8 || input.dtype == holoflow::core::DType::U16,
        "unsupported input dtype");
  const auto batch = static_cast<int>(input.shape[0]);
  check(batch > 0 && s.count % batch == 0, "frame count must be divisible by batch size");
  return {.input_descs   = {input},
          .output_descs  = {},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
NpyfileFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                       const holoflow::core::SyncCreateCtx &) const {
  infer(inputs, json);
  auto s = json.get<NpyfileSettings>();
  auto g = detail::recording_geometry_from_desc(inputs[0]);
  return std::make_unique<NpyfileWriter>(std::move(s), g, detail::recording_frame_byte_size(g));
}

std::unique_ptr<holoflow::core::ISyncTask>
NpyfileFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old,
                       std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                       const holoflow::core::SyncCreateCtx &ctx) const {
  auto *writer = dynamic_cast<NpyfileWriter *>(old.get());
  if (!writer)
    return create(inputs, json, ctx);
  infer(inputs, json);
  auto s    = json.get<NpyfileSettings>();
  auto g    = detail::recording_geometry_from_desc(inputs[0]);
  auto size = detail::recording_frame_byte_size(g);
  if (!writer->can_reuse(s, g, size))
    return create(inputs, json, ctx);
  writer->update_settings(std::move(s));
  return old;
}
} // namespace holotask::sinks
