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

#include "holotask/sources/holofile.hh"

#include <cstddef>
#include <cstdint>
#include <map>
#include <omp.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::sources {

void to_json(nlohmann::json &j, const HolofileSettings::LoadKind &lk) {
  std::map<HolofileSettings::LoadKind, std::string> lk_to_str = {
      {HolofileSettings::LoadKind::Live, "Live"},
      {HolofileSettings::LoadKind::CPUCached, "CPUCached"},
      {HolofileSettings::LoadKind::GPUCached, "GPUCached"},
  };

  HOLOVIBES_CHECK(lk_to_str.contains(lk), "Invalid LoadKind enum value");
  j = lk_to_str[lk];
}

void from_json(const nlohmann::json &j, HolofileSettings::LoadKind &lk) {
  std::map<std::string, HolofileSettings::LoadKind> str_to_lk = {
      {"Live", HolofileSettings::LoadKind::Live},
      {"CPUCached", HolofileSettings::LoadKind::CPUCached},
      {"GPUCached", HolofileSettings::LoadKind::GPUCached},
  };

  auto key = j.get<std::string>();
  if (!str_to_lk.contains(key)) {
    throw std::invalid_argument("Invalid LoadKind string: " + key);
  }
  lk = str_to_lk[key];
}

void to_json(nlohmann::json &j, const HolofileSettings &hs) {
  j = nlohmann::json{
      {"path", hs.path},           {"load_kind", hs.load_kind},   {"start_frame", hs.start_frame},
      {"end_frame", hs.end_frame}, {"batch_size", hs.batch_size},
  };
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("load_kind").get_to(hs.load_kind);
  j.at("start_frame").get_to(hs.start_frame);
  j.at("end_frame").get_to(hs.end_frame);
  j.at("batch_size").get_to(hs.batch_size);
}

namespace {

void mt_memcpy(void *dst, const void *src, const std::size_t n) {
  constexpr int NUM_THREADS = 2;
  auto         *dst_bytes   = static_cast<std::uint8_t *>(dst);
  const auto   *src_bytes   = static_cast<const std::uint8_t *>(src);

  const std::size_t chunk_size = n / NUM_THREADS;
  const std::size_t remainder  = n % NUM_THREADS;

#pragma omp parallel num_threads(NUM_THREADS)
  {
    const int         tid       = omp_get_thread_num();
    const std::size_t offset    = tid * chunk_size;
    std::size_t       this_size = chunk_size;

    if (tid == NUM_THREADS - 1) {
      this_size += remainder;
    }

    if (this_size > 0) {
      std::memcpy(dst_bytes + offset, src_bytes + offset, this_size);
    }
  }
}

} // namespace

Holofile::Holofile(const HolofileSettings &settings, holofile::Reader &&reader,
                   const holofile::Header &header, int frame_idx, std::byte *buf,
                   HostPtr<std::byte> &&h_buf, DevPtr<std::byte> &&d_buf, cudaStream_t stream)
    : settings_(settings), reader_(std::move(reader)), header_(header), frame_idx_(frame_idx),
      buf_(buf), h_buf_(std::move(h_buf)), d_buf_(std::move(d_buf)), stream_(stream) {}

holoflow::core::OpResult Holofile::execute(holoflow::core::SyncCtx &ctx) {
  size_t pixels_per_frame = header_.frame_width * header_.frame_height;
  size_t bits_per_frame   = pixels_per_frame * header_.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;
  size_t bytes_per_batch  = settings_.batch_size * bytes_per_frame;

  // Loop back to start when EOF would be bypassed when reading next batch.
  if (frame_idx_ + settings_.batch_size > settings_.end_frame) {
    if (settings_.load_kind == HolofileSettings::LoadKind::Live) {
      reader_.seek(settings_.start_frame);
    }
    frame_idx_ = settings_.start_frame;
  }

  // Read frames into buffer.
  auto *odata = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());
  switch (settings_.load_kind) {
  case HolofileSettings::LoadKind::Live:
    reader_.read_frames(odata, settings_.batch_size);
    break;

  case HolofileSettings::LoadKind::CPUCached:
    mt_memcpy(odata, buf_ + frame_idx_ * bytes_per_frame, bytes_per_batch);
    break;

  case HolofileSettings::LoadKind::GPUCached:
    CUDA_CHECK(cudaMemcpyAsync(odata, buf_ + frame_idx_ * bytes_per_frame, bytes_per_batch,
                               cudaMemcpyDeviceToDevice, stream_));
  }

  frame_idx_ += settings_.batch_size;
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
HolofileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[HolofileFactory::infer] error: {}", msg);
      throw std::invalid_argument("HolofileFactory inference error: " + msg);
    }
  };
  std::map<size_t, holoflow::core::DType> bpp_to_dtype = {
      {8, holoflow::core::DType::U8},
      {16, holoflow::core::DType::U16},
  };

  auto settings = jsettings.get<HolofileSettings>();
  auto reader   = holofile::Reader(settings.path);
  auto header   = reader.header();

  // Validate
  check(input_descs.size() == 0, "Holofile task must have no inputs");
  check(settings.start_frame < settings.end_frame, "Invalid frame range");
  check(settings.batch_size > 0, "Batch size must be positive");
  check(settings.end_frame <= (int)header.frame_count, "end_frame exceeds total frames in file");
  check(settings.batch_size <= (settings.end_frame - settings.start_frame),
        "Batch size exceeds available frames");
  check(bpp_to_dtype.contains(header.bits_per_pixel),
        "Unsupported bits_per_pixel: " + std::to_string(header.bits_per_pixel));

  // Success
  holoflow::core::TDesc odesc(
      {(size_t)settings.batch_size, header.frame_height, header.frame_width},
      bpp_to_dtype[header.bits_per_pixel],
      settings.load_kind == HolofileSettings::LoadKind::GPUCached ? holoflow::core::MemLoc::Device
                                                                  : holoflow::core::MemLoc::Host);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<HolofileSettings>();

  // Setup reader
  auto reader = holofile::Reader(settings.path);
  auto header = reader.header();
  reader.seek(settings.start_frame);

  size_t pixels_per_frame = header.frame_width * header.frame_height;
  size_t bits_per_frame   = pixels_per_frame * header.bits_per_pixel;
  size_t bytes_per_frame  = bits_per_frame / 8;
  size_t frames_to_load   = settings.end_frame - settings.start_frame;
  size_t bytes_to_load    = frames_to_load * bytes_per_frame;

  // Setup buffers
  using curaii::make_unique_device_ptr;
  using curaii::make_unique_host_ptr;
  std::byte         *buf   = nullptr;
  HostPtr<std::byte> h_buf = nullptr;
  DevPtr<std::byte>  d_buf = nullptr;

  switch (settings.load_kind) {
  case HolofileSettings::LoadKind::Live:
    // No preloading
    break;

  case HolofileSettings::LoadKind::CPUCached:
    h_buf = make_unique_host_ptr<std::byte>(bytes_to_load);
    buf   = h_buf.get();
    break;

  case HolofileSettings::LoadKind::GPUCached:
    d_buf = make_unique_device_ptr<std::byte>(bytes_to_load);
    buf   = d_buf.get();
    break;
  }

  // Preload if needed
  switch (settings.load_kind) {
  case HolofileSettings::LoadKind::Live:
    // No preloading
    break;

  case HolofileSettings::LoadKind::CPUCached:
    reader.read_frames(reinterpret_cast<uint8_t *>(h_buf.get()), frames_to_load);
    break;

  case HolofileSettings::LoadKind::GPUCached: {
    auto temp_buf = make_unique_host_ptr<std::byte>(bytes_to_load);
    reader.read_frames(reinterpret_cast<uint8_t *>(temp_buf.get()), frames_to_load);
    CUDA_CHECK(cudaMemcpyAsync(d_buf.get(), temp_buf.get(), bytes_to_load, cudaMemcpyHostToDevice,
                               ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  } break;
  }

  // Success
  auto *task = new Holofile(settings, std::move(reader), header, settings.start_frame, buf,
                            std::move(h_buf), std::move(d_buf), ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  // Validate
  auto infer        = this->infer(input_descs, jsettings);
  auto settings     = jsettings.get<HolofileSettings>();
  auto old_holofile = dynamic_cast<Holofile *>(old_task.get());
  HOLOVIBES_CHECK(old_holofile != nullptr, "old_task is not a Holofile instance");

  // Update
  bool same_cfg = (old_holofile->settings_.path == settings.path) &&
                  (old_holofile->settings_.load_kind == settings.load_kind) &&
                  (old_holofile->settings_.start_frame == settings.start_frame) &&
                  (old_holofile->settings_.end_frame == settings.end_frame) &&
                  (old_holofile->settings_.batch_size == settings.batch_size);

  bool reuse_buf = same_cfg && settings.load_kind != HolofileSettings::LoadKind::Live;

  if (reuse_buf) {
    logger()->debug("[HolofileFactory::update] Reusing existing Holofile task");
    auto reader = std::move(old_holofile->reader_);
    auto header = reader.header();
    auto buf    = old_holofile->buf_;
    auto h_buf  = std::move(old_holofile->h_buf_);
    auto d_buf  = std::move(old_holofile->d_buf_);
    reader.seek(settings.start_frame);
    int   frame_idx = settings.start_frame;
    auto *task = new Holofile(settings, std::move(reader), header, frame_idx, buf, std::move(h_buf),
                              std::move(d_buf), ctx.stream);
    return std::unique_ptr<holoflow::core::ISyncTask>(task);
  }

  // Fallback to recreate
  logger()->debug("[HolofileFactory::update] Recreating Holofile task");
  return this->create(input_descs, jsettings, ctx);
}

} // namespace holotask::sources