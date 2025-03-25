#include "holovibes/sources/holofile_source.hh"

#include <cassert>
#include <cstdlib>
#include <spdlog/spdlog.h>

#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     HolofileSource Implementation
// ==========================================================================

HolofileSource::HolofileSource(const SourceMeta &meta, cudaStream_t stream,
                               const std::string &path, int start_frame,
                               int end_frame, int batch_size,
                               LoadKind load_kind, HolofileReader reader,
                               uint8_t *internal_buffer,
                               unique_host_ptr<uint8_t> host_buffer,
                               unique_device_ptr<uint8_t> device_buffer)
    : Source(meta, stream), path_(path), start_frame_(start_frame),
      end_frame_(end_frame), batch_size_(batch_size), load_kind_(load_kind),
      frame_index_(start_frame), reader_(std::move(reader)),
      header_(reader_.header()), internal_buffer_(internal_buffer),
      host_buffer_(std::move(host_buffer)),
      device_buffer_(std::move(device_buffer)) {}

tl::expected<void, Error> HolofileSource::run(TensorView otens) {
  size_t frame_size =
      header_.frame_height * header_.frame_width * header_.bits_per_pixel / 8;

  size_t size = batch_size_ * frame_size;

  // Loop back to start
  if (frame_index_ + batch_size_ > end_frame_) {
    if (load_kind_ == LoadKind::READ_LIVE) {
      auto result = reader_.seek(start_frame_);
      if (!result) {
        holovibes_logger()->warn("Failed to seek to start_frame: {}",
                                 result.error().message());
        return tl::unexpected(Error::INTERNAL_ERROR);
      }
    }
    frame_index_ = start_frame_;
  }

  // Read frames
  if (load_kind_ == LoadKind::READ_LIVE) {
    auto result = reader_.read_frames(internal_buffer_, batch_size_);
    if (!result) {
      holovibes_logger()->warn("Failed to read frames: {}",
                               result.error().message());
      return tl::unexpected(Error::INTERNAL_ERROR);
    }
  }

  cudaError_t error = cudaSuccess;
  switch (load_kind_) {
  case LoadKind::READ_LIVE:
    error = cudaMemcpyAsync(otens.data(), internal_buffer_, size,
                            cudaMemcpyHostToDevice, stream_);
    break;
  case LoadKind::LOAD_IN_CPU:
    error = cudaMemcpyAsync(otens.data(),
                            internal_buffer_ + frame_index_ * frame_size, size,
                            cudaMemcpyHostToDevice, stream_);
    break;
  case LoadKind::LOAD_IN_GPU:
    holovibes_logger()->info("otens.data() = {}", otens.data());
    error = cudaMemcpyAsync(otens.data(),
                            internal_buffer_ + frame_index_ * frame_size, size,
                            cudaMemcpyDeviceToDevice, stream_);
    break;
  }

  if (error != cudaSuccess) {
    holovibes_logger()->warn("CUDA call failed with error: {}",
                             cudaGetErrorString(error));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return {};
}

// ==========================================================================
//                     HolofileSourceFactory Implementation
// ==========================================================================

tl::expected<SourceMeta, Error>
HolofileSourceFactory::type_check(const json &jparams) {
  auto params = jparams.get<Params>();

  if (params.batch_size <= 0) {
    holovibes_logger()->warn("Invalid batch_size: {}", params.batch_size);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.start_frame < 0) {
    holovibes_logger()->warn("Invalid start_frame: {}", params.start_frame);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.start_frame + params.batch_size > params.end_frame) {
    holovibes_logger()->warn(
        "batch size does not fit in [start_frame, end_frame[");
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.load_kind != "READ_LIVE" && params.load_kind != "LOAD_IN_CPU" &&
      params.load_kind != "LOAD_IN_GPU") {
    holovibes_logger()->warn("Invalid load_kind: {}", params.load_kind);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto result = HolofileReader::open(params.path);
  if (!result) {
    holovibes_logger()->warn("Failed to open Holofile: {}",
                             result.error().message());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto reader = std::move(result.value());
  auto header = reader.header();

  if (header.bits_per_pixel != 8) {
    holovibes_logger()->warn("Only 8-bit frames are supported.");
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (params.end_frame > static_cast<int>(header.frame_count)) {
    holovibes_logger()->warn("end_frame is greater than file frames count");
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  TensorMeta ometa(DataType::U8, MemoryLocation::DEVICE,
                   {(size_t)params.batch_size, (size_t)header.frame_height,
                    (size_t)header.frame_width});

  return SourceMeta(ometa);
}

tl::expected<std::unique_ptr<Source>, Error>
HolofileSourceFactory::create(const json &jparams, cudaStream_t stream) {
  auto meta_result = type_check(jparams);
  if (!meta_result) {
    holovibes_logger()->warn("type check failed");
    return tl::unexpected(meta_result.error());
  }

  auto meta = meta_result.value();
  auto params = jparams.get<Params>();

  HolofileSource::LoadKind load_kind;
  if (params.load_kind == "READ_LIVE") {
    load_kind = HolofileSource::LoadKind::READ_LIVE;
  } else if (params.load_kind == "LOAD_IN_CPU") {
    load_kind = HolofileSource::LoadKind::LOAD_IN_CPU;
  } else if (params.load_kind == "LOAD_IN_GPU") {
    load_kind = HolofileSource::LoadKind::LOAD_IN_GPU;
  } else {
    holovibes_logger()->critical("UNREACHABLE!");
    std::exit(EXIT_FAILURE);
  }

  auto result = HolofileReader::open(params.path);
  if (!result) {
    holovibes_logger()->warn("Failed to open Holofile: {}",
                             result.error().message());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto reader = std::move(result.value());
  auto header = reader.header();

  auto seek_result = reader.seek(params.start_frame);
  if (!seek_result) {
    holovibes_logger()->warn("Failed to seek to start_frame: {}",
                             seek_result.error().message());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  size_t frame_size =
      header.frame_height * header.frame_width * header.bits_per_pixel / 8;

  unique_host_ptr<uint8_t> host_buffer = nullptr;
  unique_device_ptr<uint8_t> device_buffer = nullptr;
  uint8_t *internal_buffer = nullptr;
  size_t size = 0;
  size_t nb_frames = 0;

  switch (load_kind) {
  case HolofileSource::LoadKind::READ_LIVE:
    size = params.batch_size * frame_size;
    host_buffer = make_unique_host_ptr<uint8_t>(size);
    internal_buffer = host_buffer.get();
    break;
  case HolofileSource::LoadKind::LOAD_IN_CPU:
    nb_frames = params.end_frame - params.start_frame;
    size = nb_frames * frame_size;
    host_buffer = make_unique_host_ptr<uint8_t>(size);
    internal_buffer = host_buffer.get();
    break;
  case HolofileSource::LoadKind::LOAD_IN_GPU:
    nb_frames = params.end_frame - params.start_frame;
    size = nb_frames * frame_size;
    device_buffer = make_unique_device_ptr<uint8_t>(size, stream);
    internal_buffer = device_buffer.get();
    break;
  }

  if (load_kind == HolofileSource::LoadKind::LOAD_IN_CPU) {
    auto result = reader.read_frames(internal_buffer, nb_frames);
    if (!result) {
      holovibes_logger()->warn("Failed to read frames: {}",
                               result.error().message());
      return tl::unexpected(Error::INTERNAL_ERROR);
    }
  } else if (load_kind == HolofileSource::LoadKind::LOAD_IN_GPU) {
    auto tmp_buffer = make_unique_host_ptr<uint8_t>(size);
    auto result = reader.read_frames(tmp_buffer.get(), nb_frames);
    if (!result) {
      holovibes_logger()->warn("Failed to read frames: {}",
                               result.error().message());
      return tl::unexpected(Error::INTERNAL_ERROR);
    }

    auto error = cudaMemcpyAsync(internal_buffer, tmp_buffer.get(), size,
                                 cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
      holovibes_logger()->warn("CUDA call failed with error: {}",
                               cudaGetErrorString(error));
      return tl::unexpected(Error::INTERNAL_ERROR);
    }
  }

  auto *source = new HolofileSource(
      meta, stream, params.path, params.start_frame, params.end_frame,
      params.batch_size, load_kind, std::move(reader), internal_buffer,
      std::move(host_buffer), std::move(device_buffer));

  return std::unique_ptr<HolofileSource>(source);
}

} // namespace dh
