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
#include "holoflow/runtime/tracing.hh"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <string_view>
#include <system_error>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstddef>
#include <cstdint>
#include <map>
#include <omp.h>
#include <thread>
#include <windows.h>

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
      {"path", hs.path},
      {"load_kind", hs.load_kind},
      {"start_frame", hs.start_frame},
      {"loop", hs.loop},
      {"tail", hs.tail},
      {"end_frame", hs.end_frame},
      {"batch_size", hs.batch_size},
      {"keep_cursor", hs.keep_cursor},
  };

  if (hs.max_fps.has_value()) {
    j["max_fps"] = *hs.max_fps;
  }
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("load_kind").get_to(hs.load_kind);
  j.at("start_frame").get_to(hs.start_frame);
  j.at("end_frame").get_to(hs.end_frame);
  j.at("batch_size").get_to(hs.batch_size);
  hs.max_fps = std::nullopt;
  if (j.contains("max_fps") && !j.at("max_fps").is_null()) {
    hs.max_fps = j.at("max_fps").get<int>();
  }
  hs.loop        = j.value("loop", true);
  hs.tail        = j.value("tail", std::string{"error"});
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

using holoflow::runtime::tracing::ScopedTrace;

size_t checked_product(size_t a, size_t b) {
  if (b && a > (std::numeric_limits<size_t>::max)() / b)
    throw std::overflow_error("Holofile payload size exceeds address space");
  return a * b;
}

size_t preload_chunk_size(size_t batch_bytes, size_t page_size) {
  constexpr size_t mib    = 1024 * 1024;
  size_t           target = 64;
  if (const char *raw = std::getenv("HOLOFLOW_HOLOFILE_PRELOAD_CHUNK_MIB")) {
    const std::string_view value(raw);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), target);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
      throw std::invalid_argument(
          "HOLOFLOW_HOLOFILE_PRELOAD_CHUNK_MIB must be a nonnegative integer");
  }
  if (!target || !batch_bytes)
    return 0;
  const auto alignment = checked_product(batch_bytes / std::gcd(batch_bytes, page_size), page_size);
  const auto target_bytes = checked_product(target, mib);
  return std::max(alignment, target_bytes / alignment * alignment);
}

class RegisteredPreload {
public:
  RegisteredPreload(size_t bytes, size_t chunk, size_t page) : chunk_(chunk), page_(page) {
    ScopedTrace timing("Holofile Reserve CPU Cache", "detail");
    if (bytes > (std::numeric_limits<size_t>::max)() - (page - 1))
      throw std::overflow_error("Holofile page-rounded cache exceeds address space");
    base_ = static_cast<std::byte *>(VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_READWRITE));
    if (!base_)
      throw std::system_error(GetLastError(), std::system_category(), "Reserve Holofile CPU cache");
  }

  ~RegisteredPreload() noexcept {
    bool released = true;
    while (registered_) {
      const auto status = cudaHostUnregister(base_ + (--registered_) * chunk_);
      if (status != cudaSuccess) {
        released = false;
        logger()->error("Holofile cache unregister failed: {}", cudaGetErrorString(status));
      }
    }
    // Avoid releasing pages still registered with CUDA.
    if (released && !VirtualFree(base_, 0, MEM_RELEASE))
      logger()->error("Holofile cache release failed: {}", GetLastError());
  }

  RegisteredPreload(const RegisteredPreload &)            = delete;
  RegisteredPreload &operator=(const RegisteredPreload &) = delete;

  std::byte *data() const { return base_; }
  bool       supports_batch(size_t bytes) const { return bytes != 0 && chunk_ % bytes == 0; }

  void load(holofile::Reader &reader, size_t bytes, size_t frame_bytes) {
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    std::mutex              mutex;
    std::condition_variable ready;
    size_t                  claimed = 0, available = 0;
    bool                    stopped = false;
    std::exception_ptr      worker_error;
    std::thread             worker([&] {
      try {
        holoflow::runtime::tracing::set_thread_name("Holofile Cache Registration");
        CUDA_CHECK(cudaSetDevice(device));
        for (size_t offset = 0, index = 0; offset < bytes; ++index) {
          {
            std::unique_lock lock(mutex);
            ready.wait(lock, [&] { return stopped || claimed >= index; });
            if (stopped)
              return;
          }
          const auto payload   = std::min(chunk_, bytes - offset);
          const auto committed = (payload + page_ - 1) / page_ * page_;
          auto      *address   = base_ + offset;
          {
            ScopedTrace timing("Holofile Commit and Register Chunk", "detail");
            if (VirtualAlloc(address, committed, MEM_COMMIT, PAGE_READWRITE) != address)
              throw std::system_error(GetLastError(), std::system_category(),
                                      "Commit Holofile CPU cache chunk");
            CUDA_CHECK(cudaHostRegister(address, committed, cudaHostRegisterPortable));
            ++registered_;
          }
          {
            std::lock_guard lock(mutex);
            ++available;
          }
          ready.notify_one();
          offset += payload;
        }
      } catch (...) {
        {
          std::lock_guard lock(mutex);
          worker_error = std::current_exception();
        }
        ready.notify_one();
      }
    });

    try {
      for (size_t offset = 0, index = 0; offset < bytes; ++index) {
        {
          std::unique_lock lock(mutex);
          ready.wait(lock, [&] { return worker_error || available > index; });
          if (worker_error)
            std::rethrow_exception(worker_error);
          ++claimed;
        }
        ready.notify_one();
        const auto  payload = std::min(chunk_, bytes - offset);
        ScopedTrace timing("Holofile Read Frame Chunk", "detail");
        reader.read_frames(reinterpret_cast<uint8_t *>(base_ + offset), payload / frame_bytes);
        offset += payload;
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex);
        stopped = true;
      }
      ready.notify_one();
      worker.join();
      throw;
    }
    worker.join();
  }

private:
  std::byte *base_;
  size_t     chunk_, page_;
  size_t     registered_ = 0;
};

class Holofile : public holoflow::core::ISyncTask {
public:
  using Clock = std::chrono::steady_clock;

  // -- Configuration ------------------------------------------------------------------------------
  HolofileSettings                  settings;
  std::unique_ptr<holofile::Reader> reader;
  holofile::Header                  header;
  int                               frame_idx;
  holoflow::core::TDesc             odesc;
  std::optional<Clock::time_point>  next_batch_start;

  // -- Buffers ------------------------------------------------------------------------------------
  std::byte                         *buf; // Non-owning view of the active buffer
  std::unique_ptr<RegisteredPreload> registered_buf;
  HostPtr<std::byte>                 h_buf; // Owned CPU buffer (if any)
  DevPtr<std::byte>                  d_buf; // Owned GPU buffer (if any)

  cudaStream_t stream; // Stream for GPU transfers

  [[nodiscard]] bool pace_before_batch(holoflow::core::SyncCtx &ctx) {
    if (!settings.max_fps.has_value()) {
      next_batch_start.reset();
      return true;
    }

    auto batch_period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
        static_cast<double>(settings.batch_size) / *settings.max_fps));

    if (batch_period <= Clock::duration::zero()) {
      batch_period = Clock::duration{1};
    }

    if (!next_batch_start.has_value()) {
      next_batch_start = Clock::now();
    }

    while (Clock::now() < *next_batch_start) {
      if (ctx.cancelled != nullptr && ctx.cancelled->load(std::memory_order_relaxed)) {
        return false;
      }

      std::this_thread::yield();
    }

    do {
      *next_batch_start += batch_period;
    } while (*next_batch_start < Clock::now());

    return true;
  }

  // -- ISyncTask interface ------------------------------------------------------------------------
  bool supports_prequeued_execution() const noexcept override {
    return settings.load_kind != HolofileSettings::LoadKind::Live && !settings.max_fps.has_value();
  }

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
    if (!settings.loop && frame_idx + settings.batch_size > settings.end_frame)
      return holoflow::core::OpResult::Eof;
    if (!pace_before_batch(ctx)) {
      return holoflow::core::OpResult::Cancelled;
    }

    size_t pixels_per_frame = static_cast<size_t>(header.frame_width) * header.frame_height;
    size_t bits_per_frame   = pixels_per_frame * header.bits_per_pixel;
    size_t bytes_per_frame  = bits_per_frame / 8;

    // Loop back to start when EOF would be bypassed when reading next batch
    if (frame_idx + settings.batch_size > settings.end_frame) {
      if (settings.load_kind == HolofileSettings::LoadKind::Live) {
        reader->seek(settings.start_frame);
      }
      frame_idx = settings.start_frame;
      if (ctx.stream_epoch != nullptr) {
        ctx.stream_epoch->fetch_add(1, std::memory_order_acq_rel);
      }
    }

    // Read frames into buffer
    if (settings.load_kind == HolofileSettings::LoadKind::Live) {
      auto *odata = reinterpret_cast<uint8_t *>(ctx.outputs[0].data());
      reader->read_frames(odata, settings.batch_size);
    } else {
      // Cached storage starts at start_frame, not at frame zero of the original file.
      std::byte *data    = buf + (frame_idx - settings.start_frame) * bytes_per_frame;
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
  check(settings.start_frame >= 0 && (settings.loop ? settings.start_frame < settings.end_frame : settings.start_frame <= settings.end_frame), "Invalid frame range");
  check(settings.tail == "error" || settings.tail == "drop", "tail must be error or drop");
  check(settings.batch_size > 0, "Batch size must be positive");
  check(!settings.max_fps.has_value() || *settings.max_fps > 0, "max_fps must be positive when provided");
  check(settings.end_frame <= static_cast<int>(header.frame_count), "end_frame exceeds total frames in file");
  check(!settings.loop || settings.batch_size <= (settings.end_frame - settings.start_frame), "Batch size exceeds available frames");
  check(bpp_to_dtype.contains(header.bits_per_pixel), "Unsupported bits_per_pixel: " + std::to_string(header.bits_per_pixel));
  // clang-format on

  holoflow::core::TDesc odesc(
      {static_cast<size_t>(settings.batch_size), header.frame_height, header.frame_width},
      bpp_to_dtype.at(header.bits_per_pixel),
      settings.load_kind == HolofileSettings::LoadKind::GPUCached ? holoflow::core::MemLoc::Device
                                                                  : holoflow::core::MemLoc::Host);

  check(settings.loop || settings.tail == "drop" ||
            (settings.end_frame - settings.start_frame) % settings.batch_size == 0,
        "Incomplete final source batch");

  bool owned_output = settings.load_kind != HolofileSettings::LoadKind::Live;

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {owned_output},
      .kind          = holoflow::core::TaskKind::Sync,
      .constness     = holoflow::core::ConstInferenceState::Mutable,
      .finite_source = !settings.loop,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<HolofileSettings>();

  // Setup reader
  std::unique_ptr<holofile::Reader> reader;
  {
    holoflow::runtime::tracing::ScopedTrace timing("Holofile Open and Seek", "detail");
    reader = std::make_unique<holofile::Reader>(settings.path);
    reader->seek(settings.start_frame);
  }
  auto header = reader->header();

  const auto   pixels          = checked_product(header.frame_width, header.frame_height);
  const auto   bytes_per_frame = checked_product(pixels, header.bits_per_pixel / 8);
  const size_t frames_to_load  = settings.end_frame - settings.start_frame;
  const auto   bytes_to_load   = checked_product(frames_to_load, bytes_per_frame);

  // Setup buffers
  using curaii::make_unique_device_ptr;
  using curaii::make_unique_host_ptr;
  std::byte         *buf   = nullptr;
  HostPtr<std::byte> h_buf = nullptr;
  DevPtr<std::byte>  d_buf = nullptr;

  std::unique_ptr<RegisteredPreload> registered_buf;

  switch (settings.load_kind) {
  case HolofileSettings::LoadKind::Live:
    // No preloading
    break;

  case HolofileSettings::LoadKind::CPUCached: {
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const auto chunk = preload_chunk_size(checked_product(bytes_per_frame, settings.batch_size),
                                          system.dwPageSize);
    if (chunk && bytes_to_load > chunk) {
      ScopedTrace timing("Holofile Pipelined CPU Preload", "detail");
      registered_buf = std::make_unique<RegisteredPreload>(bytes_to_load, chunk, system.dwPageSize);
      registered_buf->load(*reader, bytes_to_load, bytes_per_frame);
      buf = registered_buf->data();
      break;
    }
    holoflow::runtime::tracing::ScopedTrace timing("Holofile Allocate Pinned Cache", "detail");
    h_buf = make_unique_host_ptr<std::byte>(std::max(size_t{1}, bytes_to_load));
    buf   = h_buf.get();
  } break;

  case HolofileSettings::LoadKind::GPUCached: {
    holoflow::runtime::tracing::ScopedTrace timing("Holofile Allocate Device Cache", "detail");
    d_buf = make_unique_device_ptr<std::byte>(std::max(size_t{1}, bytes_to_load));
    buf   = d_buf.get();
  } break;
  }

  // Preload if needed
  switch (settings.load_kind) {
  case HolofileSettings::LoadKind::Live:
    // No preloading
    break;

  case HolofileSettings::LoadKind::CPUCached: {
    if (h_buf) {
      ScopedTrace timing("Holofile Read Frame Data", "detail");
      reader->read_frames(reinterpret_cast<uint8_t *>(h_buf.get()), frames_to_load);
    }
  } break;
  case HolofileSettings::LoadKind::GPUCached: {
    auto temp_buf = [&] {
      holoflow::runtime::tracing::ScopedTrace timing("Holofile Allocate Pinned Staging", "detail");
      return make_unique_host_ptr<std::byte>(std::max(size_t{1}, bytes_to_load));
    }();
    {
      holoflow::runtime::tracing::ScopedTrace timing("Holofile Read Frame Data", "detail");
      reader->read_frames(reinterpret_cast<uint8_t *>(temp_buf.get()), frames_to_load);
    }
    {
      holoflow::runtime::tracing::ScopedTrace timing("Holofile Submit Cache Upload", "detail");
      CUDA_CHECK(cudaMemcpyAsync(d_buf.get(), temp_buf.get(), bytes_to_load, cudaMemcpyHostToDevice,
                                 ctx.stream));
    }
    {
      holoflow::runtime::tracing::ScopedTrace timing("Holofile Release Pinned Staging", "detail");
      temp_buf.reset();
    }
  } break;
  }

  // Construct task directly
  auto task            = std::make_unique<Holofile>();
  task->settings       = settings;
  task->reader         = std::move(reader);
  task->header         = header;
  task->frame_idx      = settings.start_frame;
  task->odesc          = infer.output_descs[0];
  task->buf            = buf;
  task->registered_buf = std::move(registered_buf);
  task->h_buf          = std::move(h_buf);
  task->d_buf          = std::move(d_buf);
  task->stream         = ctx.stream;
  task->next_batch_start.reset();

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

  bool       is_live     = settings.load_kind == HolofileSettings::LoadKind::Live;
  bool       path_same   = (old_holofile->settings.path == settings.path);
  bool       kind_same   = (old_holofile->settings.load_kind == settings.load_kind);
  bool       start_same  = (old_holofile->settings.start_frame == settings.start_frame);
  bool       end_same    = (old_holofile->settings.end_frame == settings.end_frame);
  bool       batch_same  = (old_holofile->settings.batch_size == settings.batch_size);
  bool       fps_same    = (old_holofile->settings.max_fps == settings.max_fps);
  bool       pace_same   = batch_same && fps_same;
  bool       bounds_same = start_same && end_same;
  const bool batch_compatible =
      !old_holofile->registered_buf ||
      old_holofile->registered_buf->supports_batch(infer.output_descs[0].num_bytes());
  bool can_reuse = path_same && kind_same && (is_live || bounds_same) && batch_compatible;

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
  auto task            = std::make_unique<Holofile>();
  task->settings       = settings;
  task->reader         = std::move(reader);
  task->header         = header;
  task->frame_idx      = frame_idx;
  task->odesc          = infer.output_descs[0];
  task->buf            = buf;
  task->registered_buf = std::move(old_holofile->registered_buf);
  task->h_buf          = std::move(h_buf);
  task->d_buf          = std::move(d_buf);
  task->stream         = ctx.stream;
  task->next_batch_start =
      pace_same && settings.keep_cursor ? old_holofile->next_batch_start : std::nullopt;

  return task;
}

} // namespace holotask::sources
