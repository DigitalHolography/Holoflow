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

#include "holotask/sinks/holofile.hh"

// #include <Windows.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <omp.h>
#include <span>
#include <system_error>
#include <unordered_set>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holofile/holofile.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"
#include "recording_geometry.hh"
#include "recording_sink.hh"

template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::sinks {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const HolofileSettings &hs) {
  j = nlohmann::json{
      {"path", hs.path},
      {"count", hs.count},
      {"pipeline_settings", hs.pipeline_settings},
      {"use_buffer", hs.use_buffer},
  };
}

void from_json(const nlohmann::json &j, HolofileSettings &hs) {
  j.at("path").get_to(hs.path);
  j.at("count").get_to(hs.count);
  j.at("pipeline_settings").get_to(hs.pipeline_settings);
  if (j.contains("use_buffer"))
    j.at("use_buffer").get_to(hs.use_buffer);
}

// -------------------------------------------------------------------------------------------------
// Private implementation
// -------------------------------------------------------------------------------------------------

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

using detail::RecordingGeometry;

holofile::Header make_header(int count, const RecordingGeometry &g) {
  const auto frame_count = static_cast<uint32_t>(count);
  const auto data_size =
      static_cast<uint64_t>(frame_count) * g.frame_height * g.frame_width * g.bits_per_pixel / 8;
  return holofile::Header{
      .magic_number       = holofile::Header::MAGIC_NUMBER_LE,
      .version            = holofile::Header::CURRENT_VERSION,
      .bits_per_pixel     = g.bits_per_pixel,
      .frame_width        = g.frame_width,
      .frame_height       = g.frame_height,
      .frame_count        = frame_count,
      .data_size_in_bytes = data_size,
      .endianness         = holofile::Header::LITTLE_ENDIAN,
  };
}

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[HolofileFactory] {}", msg);
    throw std::invalid_argument("HolofileFactory error: " + msg);
  }
}

// -------------------------------------------------------------------------------------------------
// Buffer-then-flush writer task
// -------------------------------------------------------------------------------------------------
//
// Two-phase recording:
//   1. Accumulate: each execute() call memcpys incoming frames into the preallocated ring buffer.
//   2. Flush:      once all frames are collected, write the entire buffer to disk in one shot.
//
// No background thread. The flush happens synchronously inside execute(), so the pipeline will
// stall for one cycle while the file is written. For typical recording sizes this is acceptable
// and avoids all async complexity.

class HolofileWriter final : public detail::Sink<holofile::Writer> {
public:
  HolofileWriter(const HolofileSettings &settings, const RecordingGeometry &geometry,
                 size_t frame_byte_size)
      : settings_(settings), geometry_(geometry), frame_byte_size_(frame_byte_size),
        ring_(curaii::make_unique_host_ptr<uint8_t>(static_cast<size_t>(settings.count) *
                                                    frame_byte_size)) {
    set_recording_count(settings_.count);
    // Touch every page so the OS actually backs the allocation with physical memory.
    // Without this, the first recording pays the page-fault cost.
    const size_t total_bytes = static_cast<size_t>(settings.count) * frame_byte_size_;

    // 2. The Un-Optimizable Demand-Zero Pre-faulter
    // We cast to volatile so the compiler is strictly forbidden from removing this loop.
    volatile uint8_t *v_ptr = static_cast<volatile uint8_t *>(ring_.get());

    // Step through memory one 4KB page at a time (Windows page size)
    for (size_t i = 0; i < total_bytes; i += 4096) {
      // We MUST write a non-zero value to force the OS to physically allocate the page.
      // If we write 0, Windows memory deduplicators might map it to a shared zero-page.
      v_ptr[i] = 1;
    }
  }

  ~HolofileWriter() override = default;


  bool can_reuse(const HolofileSettings &settings, const RecordingGeometry &geometry,
                 size_t frame_byte_size) const {
    return settings.use_buffer == settings_.use_buffer && settings.count == settings_.count &&
           detail::same_geometry(geometry, geometry_) && frame_byte_size == frame_byte_size_;
  }

  void update_settings(const HolofileSettings &settings) {
    const auto active_path = settings_.path;

    settings_ = settings;
    if (recording_) {
      settings_.path = active_path;
    }
    recording_path() = settings_.path;
  }

protected:
  virtual void recording_path_update_callback() override { settings_.path = recording_path(); }
  virtual void recording_stopped_callback() override { reset(); }

private:
  int buffer_input(holoflow::core::SyncCtx &ctx, int max_frames) override {
    const auto batch_size = static_cast<int>(ctx.inputs[0].desc.shape[0]);
    const auto count = std::min(max_frames, batch_size);
    const auto *idata = reinterpret_cast<const uint8_t *>(ctx.inputs[0].data());
    const auto start = std::chrono::steady_clock::now();
    mt_memcpy(ring_.get() + static_cast<size_t>(frames_buffered_) * frame_byte_size_, idata,
              static_cast<size_t>(count) * frame_byte_size_);
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    logger()->trace("[HolofileWriter] Buffered {} frames ({} us)", count, duration);
    return count;
  }
  const std::uint8_t *buffered_data(std::size_t index) const override {
    return ring_.get() + index * frame_byte_size_;
  }
  void make_writer() override {
    emplace_writer(settings_.path, make_header(settings_.count, geometry_),
                   holofile::Footer{.pipeline_settings = settings_.pipeline_settings});
  }
  void on_flush_started() override {
    logger()->info("[HolofileWriter] Frame buffer full ({} frames); flushing to disk...", settings_.count);
  }
  void on_flush_finished() override {
    logger()->info("[HolofileWriter] Recording complete: {} frames written to {}",
                   frames_written(), settings_.path);
  }
  void on_execute_error(holoflow::core::SyncCtx &ctx, const std::exception &error) override {
    logger()->error("[HolofileWriter] Failed to finalize recording at {}: {}", settings_.path,
                    error.what());
    emit_failed_event(ctx, error.what());
    reset();
  }

private:
  HolofileSettings  settings_;
  RecordingGeometry geometry_;
  size_t            frame_byte_size_;
  HostPtr<uint8_t>  ring_; ///< Preallocated at construction; reused across recordings.
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
HolofileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  auto settings = jsettings.get<HolofileSettings>();

  check(settings.use_buffer, "use_buffer must be true (synchronous writer is not supported)");
  check(input_descs.size() == 1, "expected exactly one input tensor");

  auto &idesc = input_descs[0];
  check(idesc.shape.size() == 3, "input tensor must have rank 3 (batch, height, width)");
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "input tensor must be in Host memory");

  static const std::unordered_set<holoflow::core::DType> supported_dtypes = {
      holoflow::core::DType::U8,
      holoflow::core::DType::U16,
  };
  check(supported_dtypes.contains(idesc.dtype),
        "unsupported input dtype: " + std::to_string(static_cast<int>(idesc.dtype)));

  const auto batch_size = static_cast<int>(idesc.shape[0]);
  check(settings.count % batch_size == 0, "frame count (" + std::to_string(settings.count) +
                                              ") must be divisible by batch size (" +
                                              std::to_string(batch_size) + ")");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx &) const {
  auto  settings = jsettings.get<HolofileSettings>();
  auto &idesc    = input_descs[0];

  const auto geometry        = detail::recording_geometry_from_desc(idesc);
  const auto frame_byte_size = detail::recording_frame_byte_size(geometry);

  logger()->info("[HolofileFactory::create] Buffer-then-flush writer ({} frames)", settings.count);
  return std::make_unique<HolofileWriter>(settings, geometry, frame_byte_size);
}

std::unique_ptr<holoflow::core::ISyncTask>
HolofileFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old = dynamic_cast<HolofileWriter *>(old_task.get());
  if (old == nullptr)
    return create(input_descs, jsettings, ctx);

  infer(input_descs, jsettings);

  const auto settings        = jsettings.get<HolofileSettings>();
  const auto geometry        = detail::recording_geometry_from_desc(input_descs[0]);
  const auto frame_byte_size = detail::recording_frame_byte_size(geometry);

  if (!old->can_reuse(settings, geometry, frame_byte_size)) {
    logger()->debug("[HolofileFactory::update] Recreating writer because buffer shape changed");
    return create(input_descs, jsettings, ctx);
  }

  logger()->debug("[HolofileFactory::update] Reusing existing Holofile writer buffer");
  old->update_settings(settings);
  return old_task;
}

} // namespace holotask::sinks
