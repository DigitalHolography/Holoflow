#include "holovibes/sources/holofile_source.hh"

#include <cassert>
#include <cstdlib>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holovibes/holovibes.hh"

using namespace std::chrono_literals;

namespace dh {

// ==========================================================================
//                     HolofileSource Implementation
// ==========================================================================

HolofileSource::HolofileSource(
    const SourceMeta &meta, cudaStream_t stream, const std::string &path,
    int start_frame, int end_frame, int batch_size, LoadKind load_kind,
    HolofileReader reader, uint8_t *internal_buffer,
    curaii::cuda::unique_host_ptr<uint8_t> host_buffer,
    curaii::cuda::unique_device_ptr<uint8_t> device_buffer)
    : Source(meta, stream), path_(path), start_frame_(start_frame),
      end_frame_(end_frame), batch_size_(batch_size), load_kind_(load_kind),
      frame_index_(start_frame), reader_(std::move(reader)),
      header_(reader_.header()), internal_buffer_(internal_buffer),
      host_buffer_(std::move(host_buffer)),
      device_buffer_(std::move(device_buffer)) {}

void HolofileSource::run(TensorView otens) {
  size_t frame_size =
      header_.frame_height * header_.frame_width * header_.bits_per_pixel / 8;

  size_t size = batch_size_ * frame_size;

  // Loop back to start
  if (frame_index_ + batch_size_ > end_frame_) {
    holovibes_logger()->info("Loop back");
    if (load_kind_ == LoadKind::READ_LIVE) {
      auto result = reader_.seek(start_frame_);
      if (!result) {
        throw std::runtime_error(fmt::format(
            "Failed to seek to start_frame: {}", result.error().message()));
      }
    }
    frame_index_ = start_frame_;
  }

  switch (load_kind_) {
  case LoadKind::READ_LIVE: {
    auto result =
        reader_.read_frames(static_cast<uint8_t *>(otens.data()), batch_size_);
    if (!result) {
      throw std::runtime_error(
          fmt::format("Failed to read frames: {}", result.error().message()));
    }
  } break;
  case LoadKind::LOAD_IN_CPU: {
    int64_t *src = (int64_t *)(internal_buffer_ + frame_index_ * frame_size);
    int64_t *dst = (int64_t *)otens.data();
    int64_t nb_iter = size / 8;
#pragma omp parallel for num_threads(12) schedule(static)
    for (int64_t i = 0; i < nb_iter; i++) {
      dst[i] = src[i];
    }
    // CUDA_CHECK(cudaMemcpyAsync(otens.data(),
    //                            internal_buffer_ + frame_index_ * frame_size,
    //                            size, cudaMemcpyHostToHost, stream_));
    break;
  }
  case LoadKind::LOAD_IN_GPU:
    CUDA_CHECK(cudaMemcpyAsync(otens.data(),
                               internal_buffer_ + frame_index_ * frame_size,
                               size, cudaMemcpyDeviceToDevice, stream_));
    break;
  }

  frame_index_ += batch_size_;
}

// ==========================================================================
//                     HolofileSourceFactory Implementation
// ==========================================================================

SourceMeta HolofileSourceFactory::type_check(const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::runtime_error(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.batch_size > 0, "batch_size <= 0");
  check(params.start_frame >= 0, "start_frame < 0");
  check(params.start_frame + params.batch_size <= params.end_frame,
        "start_frame + batch_size > end_frame");
  check(params.load_kind == "READ_LIVE" || params.load_kind == "LOAD_IN_CPU" ||
            params.load_kind == "LOAD_IN_GPU",
        "load_kind is not in [READ_LIVE, LOAD_IN_CPU_LOAD_IN_GPU]");

  // 2) Holofile sanity
  auto result = HolofileReader::open(params.path);
  if (!result) {
    throw std::runtime_error(
        fmt::format("Failed to open Holofile: {}", result.error().message()));
  }
  auto reader = std::move(result.value());
  auto header = reader.header();

  DataType data_type;
  switch (header.bits_per_pixel) {
  case 8:
    data_type = DataType::U8;
    break;
  case 16:
    data_type = DataType::U16;
    break;
  default:
    throw std::invalid_argument("file data type not in [u8, u16]");
  }

  check(params.end_frame <= static_cast<int>(header.frame_count),
        "end_frame > file frame count");

  // 3) Success
  MemoryLocation location = params.load_kind == "LOAD_IN_GPU"
                                ? MemoryLocation::DEVICE
                                : MemoryLocation::HOST;

  TensorMeta ometa(data_type, location,
                   {(size_t)params.batch_size, (size_t)header.frame_height,
                    (size_t)header.frame_width});

  return SourceMeta(ometa);
}

std::unique_ptr<Source> HolofileSourceFactory::create(const json &jparams,
                                                      cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(jparams);
  auto params = jparams.get<Params>();

  HolofileSource::LoadKind load_kind;
  if (params.load_kind == "READ_LIVE") {
    load_kind = HolofileSource::LoadKind::READ_LIVE;
  } else if (params.load_kind == "LOAD_IN_CPU") {
    load_kind = HolofileSource::LoadKind::LOAD_IN_CPU;
  } else if (params.load_kind == "LOAD_IN_GPU") {
    load_kind = HolofileSource::LoadKind::LOAD_IN_GPU;
  } else {
    DH_BUG("unreachable statement reached");
  }

  // 2) Open file
  auto result = HolofileReader::open(params.path);
  if (!result) {
    throw std::runtime_error(
        fmt::format("Failed to open Holofile: {}", result.error().message()));
  }
  auto reader = std::move(result.value());
  auto header = reader.header();

  // 3) Seek to first frame
  auto seek_result = reader.seek(params.start_frame);
  if (!seek_result) {
    throw std::runtime_error(fmt::format("Failed to seek to start_frame: {}",
                                         seek_result.error().message()));
  }

  // 4) Allocations
  size_t frame_size =
      header.frame_height * header.frame_width * header.bits_per_pixel / 8;

  curaii::cuda::unique_host_ptr<uint8_t> host_buffer = nullptr;
  curaii::cuda::unique_device_ptr<uint8_t> device_buffer = nullptr;
  uint8_t *internal_buffer = nullptr;
  size_t size = 0;
  size_t nb_frames = 0;

  switch (load_kind) {
  case HolofileSource::LoadKind::READ_LIVE:
    // Do nothing here as the read buffer will be the one provided to the run
    // method.
    break;
  case HolofileSource::LoadKind::LOAD_IN_CPU:
    nb_frames = params.end_frame - params.start_frame;
    size = nb_frames * frame_size;
    host_buffer = curaii::cuda::make_unique_host_ptr<uint8_t>(size);
    internal_buffer = host_buffer.get();
    break;
  case HolofileSource::LoadKind::LOAD_IN_GPU:
    nb_frames = params.end_frame - params.start_frame;
    size = nb_frames * frame_size;
    device_buffer = curaii::cuda::make_unique_device_ptr<uint8_t>(size);
    internal_buffer = device_buffer.get();
    break;
  }

  // 5) Preload frames
  if (load_kind == HolofileSource::LoadKind::LOAD_IN_CPU) {
    auto result = reader.read_frames(internal_buffer, nb_frames);
    if (!result) {
      throw std::runtime_error(
          fmt::format("Failed to read frames: {}", result.error().message()));
    }
  } else if (load_kind == HolofileSource::LoadKind::LOAD_IN_GPU) {
    auto tmp_buffer = curaii::cuda::make_unique_host_ptr<uint8_t>(size);
    auto result = reader.read_frames(tmp_buffer.get(), nb_frames);
    if (!result) {
      throw std::runtime_error(
          fmt::format("Failed to read frames: {}", result.error().message()));
    }

    CUDA_CHECK(cudaMemcpyAsync(internal_buffer, tmp_buffer.get(), size,
                               cudaMemcpyHostToDevice));
  }

  // 5) Assemble source
  auto *source = new HolofileSource(
      meta, stream, params.path, params.start_frame, params.end_frame,
      params.batch_size, load_kind, std::move(reader), internal_buffer,
      std::move(host_buffer), std::move(device_buffer));

  return std::unique_ptr<HolofileSource>(source);
}

} // namespace dh
