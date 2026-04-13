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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <omp.h>

#include "bug.hh"
#include "logger.hh"

#include "curaii/cuda.hh"
#include "holofile/holofile.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sources {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const HolofileSettings::LoadKind &lk) {
  static const std::map<HolofileSettings::LoadKind, std::string> lk_to_str = {
      {HolofileSettings::LoadKind::Live, "Live"},
      {HolofileSettings::LoadKind::CPUCached, "CPUCached"},
      {HolofileSettings::LoadKind::GPUCached, "GPUCached"},
  };

  HOLOVIBES_CHECK(lk_to_str.contains(lk), "Invalid LoadKind enum value");
  j = lk_to_str.at(lk);
}

void from_json(const nlohmann::json &j, HolofileSettings::LoadKind &lk) {
  static const std::map<std::string, HolofileSettings::LoadKind> str_to_lk = {
      {"Live", HolofileSettings::LoadKind::Live},
      {"CPUCached", HolofileSettings::LoadKind::CPUCached},
      {"GPUCached", HolofileSettings::LoadKind::GPUCached},
  };

  auto key = j.get<std::string>();
  if (!str_to_lk.contains(key)) {
    throw std::invalid_argument("Invalid LoadKind string: " + key);
  }
  lk = str_to_lk.at(key);
}

void to_json(nlohmann::json &j, const HolofileSettings &hs) {
  j = {
      {"path", hs.path},           {"load_kind", hs.load_kind},   {"start_frame", hs.start_frame},
      {"end_frame", hs.end_frame}, {"batch_size", hs.batch_size}, {"keep_cursor", hs.keep_cursor},
  };
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("load_kind").get_to(hs.load_kind);
  j.at("start_frame").get_to(hs.start_frame);
  j.at("end_frame").get_to(hs.end_frame);
  j.at("batch_size").get_to(hs.batch_size);
  hs.keep_cursor = j.value("keep_cursor", true);
}

// -------------------------------------------------------------------------------------------------
// Private implementation
// -------------------------------------------------------------------------------------------------

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[HolofileFactory::infer] error: {}", msg);
    throw std::invalid_argument("HolofileFactory inference error: " + msg);
  }
}

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

// -------------------------------------------------------------------------------------------------
// Holofile task implementation (private to this translation unit)
// -------------------------------------------------------------------------------------------------

class Holofile : public holoflow::core::ISyncTask {
public:
  // -- Configuration ------------------------------------------------------------------------------
  HolofileSettings                  settings;
  std::unique_ptr<holofile::Reader> reader;
  holofile::Header                  header;
  int                               frame_idx;
  holoflow::core::TDesc             odesc;

  // -- Buffers ------------------------------------------------------------------------------------
  std::byte         *buf;   // Non-owning view of the active buffer
  HostPtr<std::byte> h_buf; // Owned CPU buffer (if any)
  DevPtr<std::byte>  d_buf; // Owned GPU buffer (if any)

  cudaStream_t stream; // Stream for GPU transfers

  // -- ISyncTask interface ------------------------------------------------------------------------
  std::optional<holoflow::core::TView> acquire_input(int index) override {
    (void)index; // Unused since there are no inputs
    throw std::out_of_range("Holofile task has no inputs");
  }

  void release_output(int index) override {
    if (settings.load_kind == HolofileSettings::LoadKind::Live) {
      throw std::logic_error("Cannot release output for Live load kind");
    }

    if (index != 0) {
      throw std::out_of_range("Holofile task has only one output at index 0");
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    size_t pixels_per_frame = header.frame_width * header.frame_height;
    size_t bits_per_frame   = pixels_per_frame * header.bits_per_pixel;
    size_t bytes_per_frame  = bits_per_frame / 8;

    // Loop back to start when EOF would be bypassed when reading next batch
    if (frame_idx + settings.batch_size > settings.end_frame) {
      if (settings.load_kind == HolofileSettings::LoadKind::Live) {
        reader->seek(settings.start_frame);
      }
      frame_idx = settings.start_frame;
    }

    // Read frames into buffer
    if (settings.load_kind == HolofileSettings::LoadKind::Live) {
      auto *odata = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());
      reader->read_frames(odata, settings.batch_size);
    } else {
      std::byte *data    = buf + frame_idx * bytes_per_frame;
      auto      &storage = storage_access().owned_output_storage(0);
      storage.ptr        = data;

      ctx.outputs[0] = holoflow::core::TView{
          .desc    = odesc,
          .storage = &storage,
      };
    }

    frame_idx += settings.batch_size;
    return holoflow::core::OpResult::Ok;
  }
};

} // namespace

// -------------------------------------------------------------------------------------------------
// HolofileFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
HolofileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  static const std::map<size_t, holoflow::core::DType> bpp_to_dtype = {
      {8, holoflow::core::DType::U8},
      {16, holoflow::core::DType::U16},
  };

  auto settings = jsettings.get<HolofileSettings>();
  auto reader   = std::make_unique<holofile::Reader>(settings.path);
  auto header   = reader->header();

  // clang-format off
  check(input_descs.size() == 0, "Holofile task must have no inputs");
  check(settings.start_frame < settings.end_frame, "Invalid frame range");
  check(settings.batch_size > 0, "Batch size must be positive");
  check(settings.end_frame <= static_cast<int>(header.frame_count), "end_frame exceeds total frames in file");
  check(settings.batch_size <= (settings.end_frame - settings.start_frame), "Batch size exceeds available frames");
  check(bpp_to_dtype.contains(header.bits_per_pixel), "Unsupported bits_per_pixel: " + std::to_string(header.bits_per_pixel));
  // clang-format on

  holoflow::core::TDesc odesc(
      {static_cast<size_t>(settings.batch_size), header.frame_height, header.frame_width},
      bpp_to_dtype.at(header.bits_per_pixel),
      settings.load_kind == HolofileSettings::LoadKind::GPUCached ? holoflow::core::MemLoc::Device
                                                                  : holoflow::core::MemLoc::Host);

  bool owned_output = settings.load_kind != HolofileSettings::LoadKind::Live;

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {owned_output},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<HolofileSettings>();

  // Setup reader
  auto reader = std::make_unique<holofile::Reader>(settings.path);
  auto header = reader->header();
  reader->seek(settings.start_frame);

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
    reader->read_frames(reinterpret_cast<uint8_t *>(h_buf.get()), frames_to_load);
    break;

  case HolofileSettings::LoadKind::GPUCached: {
    auto temp_buf = make_unique_host_ptr<std::byte>(bytes_to_load);
    reader->read_frames(reinterpret_cast<uint8_t *>(temp_buf.get()), frames_to_load);
    CUDA_CHECK(cudaMemcpyAsync(d_buf.get(), temp_buf.get(), bytes_to_load, cudaMemcpyHostToDevice,
                               ctx.stream));
    // Stream sync removed here!
  } break;
  }

  // Construct task directly
  auto task       = std::make_unique<Holofile>();
  task->settings  = settings;
  task->reader    = std::move(reader);
  task->header    = header;
  task->frame_idx = settings.start_frame;
  task->odesc     = infer.output_descs[0];
  task->buf       = buf;
  task->h_buf     = std::move(h_buf);
  task->d_buf     = std::move(d_buf);
  task->stream    = ctx.stream;

  return task;
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old_holofile = dynamic_cast<Holofile *>(old_task.get());
  if (old_holofile == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<HolofileSettings>();

  bool is_live     = settings.load_kind == HolofileSettings::LoadKind::Live;
  bool path_same   = (old_holofile->settings.path == settings.path);
  bool kind_same   = (old_holofile->settings.load_kind == settings.load_kind);
  bool start_same  = (old_holofile->settings.start_frame == settings.start_frame);
  bool end_same    = (old_holofile->settings.end_frame == settings.end_frame);
  bool bounds_same = start_same && end_same;
  bool can_reuse   = path_same && kind_same && (is_live || bounds_same);

  if (!can_reuse) {
    logger()->debug(
        "[HolofileFactory::update] Cannot reuse existing task (path_same={}, kind_same={}, "
        "bounds_same={})",
        path_same, kind_same, bounds_same);
    return create(input_descs, jsettings, ctx);
  }

  logger()->debug("[HolofileFactory::update] Reusing existing Holofile task");

  // Transfer ownership of heavy resources
  auto reader = std::move(old_holofile->reader);
  auto header = reader->header();
  auto buf    = old_holofile->buf;
  auto h_buf  = std::move(old_holofile->h_buf);
  auto d_buf  = std::move(old_holofile->d_buf);

  // Resolve cursor index
  int frame_idx = settings.start_frame;
  if (settings.keep_cursor) {
    frame_idx = old_holofile->frame_idx;
    // Clamp just in case the bounds were shrunk past the current cursor
    if (frame_idx < settings.start_frame || frame_idx >= settings.end_frame) {
      frame_idx = settings.start_frame;
    }
  }

  if (is_live && frame_idx != old_holofile->frame_idx) {
    logger()->debug(
        "[HolofileFactory::update] Seeking reader from frame {} to {} due to cursor change",
        old_holofile->frame_idx, frame_idx);
    reader->seek(frame_idx);
  }

  // Construct task directly
  auto task       = std::make_unique<Holofile>();
  task->settings  = settings;
  task->reader    = std::move(reader);
  task->header    = header;
  task->frame_idx = frame_idx;
  task->odesc     = infer.output_descs[0];
  task->buf       = buf;
  task->h_buf     = std::move(h_buf);
  task->d_buf     = std::move(d_buf);
  task->stream    = ctx.stream;

  return task;
}

} // namespace holotask::sources
