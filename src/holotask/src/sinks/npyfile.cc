#include "holotask/sinks/npyfile.hh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "logger.hh"

#include "curaii/cuda.hh"
#include "recording_geometry.hh"
#include "recording_sink.hh"

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

class NpyfileWriter final : public detail::RecordingSink {
public:
  NpyfileWriter(NpyfileSettings settings, detail::RecordingGeometry geometry,
                std::size_t frame_size)
      : settings_(std::move(settings)), geometry_(geometry), frame_size_(frame_size),
        ring_(curaii::make_unique_host_ptr<std::uint8_t>(static_cast<std::size_t>(settings_.count) *
                                                         frame_size)),
        writer_{std::nullopt} {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    handle_events(ctx);
    if (!is_recording())
      return holoflow::core::OpResult::Ok;

    if (frames_buffered_ < settings_.count) {
      const auto batch = static_cast<int>(ctx.inputs[0].desc.shape[0]);
      const auto count = std::min(settings_.count - frames_buffered_, batch);
      copy_bytes(ring_.get() + static_cast<std::size_t>(frames_buffered_) * frame_size_,
                 ctx.inputs[0].data(), static_cast<std::size_t>(count) * frame_size_);
      frames_buffered_ += count;
      monitor_.update(ctx, frames_buffered_, 0);
    } else {
      try {
        flush_to_disk(ctx);
      } catch (const std::exception &error) {
        std::remove(settings_.path.c_str());
        emit_failed_event(ctx, error.what());
        logger()->error("[NpyfileWriter] Failed to finalize recording at {}: {}", settings_.path,
                        error.what());
      }
    }
    return holoflow::core::OpResult::Ok;
  }

  bool can_reuse(const NpyfileSettings &settings, detail::RecordingGeometry geometry,
                 std::size_t frame_size) const {
    return settings.use_buffer == settings_.use_buffer && settings.count == settings_.count &&
           detail::same_geometry(geometry, geometry_) && frame_size == frame_size_;
  }
  void update_settings(NpyfileSettings settings) {
    settings_ = std::move(settings);
    update_recording_path(settings_.path);
  }

protected:
  virtual void recording_reset_callback() override {
    frames_written_ = 0;
    start_flush_    = true;
    writer_.reset();
  }
  void recording_path_update_callback() override { settings_.path = recording_path(); }

  virtual void recording_stopped_callback() override { reset(); }

private:
  void make_writer() {
    const auto header = npyfile::Header{.shape = {static_cast<std::size_t>(settings_.count),
                                                  geometry_.frame_height, geometry_.frame_width},
                                        .dtype = geometry_.bits_per_pixel == 8 ? "<u1" : "<u2"};
    writer_.emplace(settings_.path, header);
  }

  void flush_to_disk(holoflow::core::SyncCtx &ctx) {

    if (start_flush_) {
      logger()->info("[NpyFileWriter] Frame buffer full ({} frames); flushing to disk...",
                     settings_.count);
      start_flush_    = false;
      frames_written_ = 0;
      make_writer();
    }

    assert(writer_.has_value());

    const auto remaining  = settings_.count - frames_written_;
    const auto batch_size = static_cast<int>(ctx.inputs[0].desc.shape[0]);
    const auto to_write   = std::min(remaining, batch_size);

    if (to_write > 0) {
      const auto *buffer = ring_.get() + frames_written_ * frame_size_;
      writer_->write_frames(buffer, to_write, frame_size_);
      frames_written_ += to_write;
      monitor_.update(ctx, frames_buffered_, frames_written_, false);
    } else {
      start_flush_ = true;
      emit_finished_event(ctx);
      logger()->info("[NpyFileWriter] Recording complete: {} frames written to {}", frames_written_,
                     settings_.path);
      writer_.reset();
    }
  }

private:
  NpyfileSettings                settings_;
  detail::RecordingGeometry      geometry_;
  std::size_t                    frame_size_;
  int                            frames_written_ = 0;
  HostPtr<std::uint8_t>          ring_;
  std::optional<npyfile::Writer> writer_;
  bool                           start_flush_ = true;
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
