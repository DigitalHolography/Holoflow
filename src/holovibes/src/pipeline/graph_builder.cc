// Copyright 2026 Digital Holography Foundation
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

#include "graph_builder.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <spdlog/fmt/ranges.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cuda_runtime.h>
#include <windows.h>

#include "bug.hh"
#include "graph_builder_tasks.hh"
#include "logger.hh"
#include "pipeline/settings.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

using PhaseReference = holotask::syncs::STFDPhaseReference;

// -------------------------------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------------------------------

class GraphBuilder::Impl : public GraphBuilderTasks {
public:
  Impl(const Settings &settings, holoflow::core::Registry &registry);

  holoflow::core::GraphSpec build();

  [[nodiscard]] std::string
  failure_graph_dot(const holoflow::core::GraphSpecDumpPreferences &prefs) const {
    return GraphBuilderTracer::failure_graph_dot(prefs);
  }

private:
  struct ShackHartmannGeometry {
    size_t frame_width;
    size_t frame_height;
    size_t nb_subapertures;
    size_t subaperture_width;
    size_t subaperture_height;

    float wavelength;
    float pixel_pitch_x;
    float pixel_pitch_y;
    float propagation_distance;
    float pupil_radius_m;
  };

  struct ShackHartmannSlopeOutput {
    TDesc                slopes;
    std::optional<TDesc> xcorr;
  };

  struct AberrationCorrectionState {
    std::optional<TDesc> cumulative_coeffs_gpu;
    std::optional<TDesc> cumulative_phase_gpu;
  };

  // clang-format off
  TDesc build_acquisition();
  TDesc short_time_fresnel_diffraction(const TDesc &field, size_t win_w, size_t win_h, size_t stride_x, size_t stride_y, float lam, float dx, float dy, float z_prop, PhaseReference phase_ref, bool skip_phase_shift = true);
  void build_raw_record(const TDesc &H);
  bool build_raw_view(const TDesc &H);
  TDesc build_preprocessing(TDesc H);
  TDesc build_time_frequency_analysis(TDesc H);
  TDesc build_aberration_correction(const TDesc &FH_current, const TDesc &FH_delayed);
  TDesc build_aberration_correction_pass(const TDesc &FH_current, const TDesc &FH_delayed, bool is_last_pass, AberrationCorrectionState &state);
  ShackHartmannGeometry shack_hartmann_geometry(const TDesc &FH) const;
  TDesc build_shack_hartmann_sensor(const TDesc &FH, const ShackHartmannGeometry &geometry);
  ShackHartmannSlopeOutput build_shack_hartmann_slopes(const TDesc &sensor_images, const ShackHartmannGeometry &geometry, bool output_xcorr);
  void build_shack_hartmann_view(const TDesc &sensor_images, const ShackHartmannGeometry &geometry);
  void build_shack_hartmann_xcorr_view(const TDesc &xcorr, const ShackHartmannGeometry &geometry);
  TDesc build_zernike_correction(const TDesc &FH, const TDesc &slopes, const ShackHartmannGeometry &geometry, bool is_last_pass, AberrationCorrectionState &state);
  void build_zernike_outputs(const AberrationCorrectionState &state, const ShackHartmannGeometry &geometry);
  TDesc build_spatial_propagation(const TDesc &FH);
  TDesc fresnel_diffraction_numpy(const TDesc &FH);
  TDesc build_spatial_filter(const TDesc &FH_z);
  TDesc build_spatial_contrast_input(TDesc H);
  void build_spatial_contrast_view(const TDesc &frames);
  void build_xy_view(const TDesc &FH_z);
  void build_xy_output(TDesc result, size_t discard_first, bool fixed_contrast_range = false);
  void build_3d_cuts(const TDesc &FH_z);
  TDesc build_freq_weights();
  // clang-format on

  struct QueuePlanEntry {
    std::string name;
    bool        device;
    size_t      depth;
    size_t      slots;
    size_t      bytes_per_slot;
    size_t      extra_bytes;
    size_t      total_bytes;
    size_t      floor;
    size_t      ceiling;
    size_t      input_batch;
    size_t      stride;
    size_t      extra_slots;
    bool        sliding_average;
    bool        adjustable;
    size_t      target_bytes;
    V           vertex;
  };

  void   start_queue_plan(size_t frame_bytes);
  size_t add_queue_plan(std::string name, bool device, size_t requested_depth, size_t minimum_depth,
                        size_t                               bytes_per_slot,
                        const std::function<size_t(size_t)> &allocation_bytes,
                        const std::function<size_t(size_t)> &slot_count, size_t extra_bytes = 0);
  TDesc  planned_batch_queue(const char *name, const TDesc &desc, size_t requested_depth,
                             size_t minimum_depth, int output_size, int output_stride);
  void   finish_queue_plan();
  [[noreturn]] void throw_queue_plan_error(const std::string &reason) const;

  Settings                    s_;
  std::vector<QueuePlanEntry> queue_plan_;
  size_t                      frame_bytes_   = 0;
  size_t                      free_vram_     = 0;
  size_t                      total_vram_    = 0;
  size_t                      free_ram_      = 0;
  size_t                      total_ram_     = 0;
  size_t                      vram_headroom_ = 0;
  size_t                      ram_headroom_  = 0;

  std::map<LoadMethod, holotask::sources::HolofileSettings::LoadKind> load_method_map_{
      {LoadMethod::READ_LIVE, holotask::sources::HolofileSettings::LoadKind::Live},
      {LoadMethod::LOAD_IN_CPU, holotask::sources::HolofileSettings::LoadKind::CPUCached},
      {LoadMethod::LOAD_IN_GPU, holotask::sources::HolofileSettings::LoadKind::GPUCached},
  };
};

// -----------------------------------------------------------------------------
// Helpers declarations
// -----------------------------------------------------------------------------

namespace {

constexpr size_t kMaxVramUsagePercent    = 99;
constexpr size_t kGpuInputFreeVramPercent = 75;
constexpr size_t kRamHeadroomMinimum     = 256ULL * 1024 * 1024;
constexpr size_t kHeadroomPercent        = 10;

// AUTO sizes each queue from a byte budget: large input frames receive shallow queues, while
// smaller frames can use more slots up to a queue-specific ceiling. Floors preserve the
// manual-mode safety capacities and BatchQueue's batch/stride slot-alignment requirements.
// SlidingAverage also accounts for its window ring and running-average scratch buffer.
constexpr size_t kQueueMinimumBytes          = 1ULL * 1024 * 1024;
constexpr size_t kIntermediateTargetBytes   = 16ULL * 1024 * 1024;
constexpr size_t kDisplayTargetBytes         = 8ULL * 1024 * 1024;
constexpr size_t kCpuOutputTargetBytes       = 8ULL * 1024 * 1024;
constexpr size_t kSlidingTargetBytes         = 16ULL * 1024 * 1024;

size_t automatic_queue_depth(size_t bytes_per_frame, size_t target_bytes, size_t floor,
                             size_t ceiling) {
  if (bytes_per_frame == 0)
    return floor;
  const size_t rounded_depth = (target_bytes + bytes_per_frame / 2) / bytes_per_frame;
  return std::clamp(rounded_depth, floor, ceiling);
}

size_t aligned_batch_slots(size_t input_batch, size_t output_stride, size_t target_capacity) {
  const size_t alignment = std::lcm(input_batch, output_stride);
  if (alignment == 0)
    throw std::overflow_error("queue slot alignment overflow");
  const size_t minimum = std::max(target_capacity, output_stride) + input_batch;
  return ((minimum + alignment - 1) / alignment) * alignment;
}

holotask::syncs::FlatfieldSettings flatfield_settings_from_cutoff_period(float cutoff_period_m,
                                                                         float dy_m, float dx_m);

std::pair<float, float> fresnel_1fft_output_pitch(float wavelength_m, float z_m, float dy_in_m,
                                                  float dx_in_m, size_t ny, size_t nx);

std::pair<float, float> post_propagation_pitch(const Settings              &settings,
                                               const holoflow::core::TDesc &desc);

} // namespace

// -------------------------------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------------------------------

GraphBuilder::GraphBuilder(const Settings &settings, holoflow::core::Registry &registry)
    : impl_(std::make_unique<Impl>(settings, registry)) {}

GraphBuilder::~GraphBuilder() = default;

GraphBuilder::GraphBuilder(GraphBuilder &&) noexcept            = default;
GraphBuilder &GraphBuilder::operator=(GraphBuilder &&) noexcept = default;

holoflow::core::GraphSpec GraphBuilder::build() { return impl_->build(); }

std::string
GraphBuilder::failure_graph_dot(const holoflow::core::GraphSpecDumpPreferences &prefs) const {
  return impl_->failure_graph_dot(prefs);
}

GraphBuilder::Impl::Impl(const Settings &settings, holoflow::core::Registry &registry)
    : GraphBuilderTasks(registry), s_(settings) {}

void GraphBuilder::Impl::start_queue_plan(size_t frame_bytes) {
  frame_bytes_ = frame_bytes;
  queue_plan_.clear();
  size_t            free_vram   = 0;
  const cudaError_t cuda_status = cudaMemGetInfo(&free_vram, &total_vram_);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                             cudaGetErrorString(cuda_status));
  }
  free_vram_ = free_vram;
  MEMORYSTATUSEX memory_status{};
  memory_status.dwLength = sizeof(memory_status);
  if (!GlobalMemoryStatusEx(&memory_status)) {
    throw std::runtime_error("GlobalMemoryStatusEx failed while planning pipeline queues");
  }
  free_ram_      = static_cast<size_t>(memory_status.ullAvailPhys);
  total_ram_     = static_cast<size_t>(memory_status.ullTotalPhys);
  const size_t maximum_vram_usage = total_vram_ / 100 * kMaxVramUsagePercent;
  const size_t used_vram = total_vram_ > free_vram_ ? total_vram_ - free_vram_ : 0;
  const size_t remaining_vram_budget =
      maximum_vram_usage > used_vram ? maximum_vram_usage - used_vram : 0;
  vram_headroom_ = free_vram_ - std::min(free_vram_, remaining_vram_budget);
  ram_headroom_  = std::max(kRamHeadroomMinimum, free_ram_ * kHeadroomPercent / 100);
}

size_t GraphBuilder::Impl::add_queue_plan(std::string name, bool device, size_t requested_depth,
                                          size_t minimum_depth, size_t bytes_per_slot,
                                          const std::function<size_t(size_t)> &allocation_bytes,
                                          const std::function<size_t(size_t)> &slot_count,
                                          size_t                               extra_bytes) {
  const size_t depth = requested_depth;
  queue_plan_.push_back({std::move(name),
                         device,
                         depth,
                         slot_count(depth),
                         bytes_per_slot,
                         extra_bytes,
                         allocation_bytes(depth),
                         minimum_depth,
                         requested_depth,
                         1,
                         1,
                         0,
                         true,
                         false,
                         0,
                         {}});
  return depth;
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::planned_batch_queue(const char *name, const TDesc &desc, size_t requested_depth,
                                        size_t minimum_depth, int output_size, int output_stride) {
  const size_t input_batch  = static_cast<size_t>(desc.shape.at(0));
  const size_t element_size = desc.num_bytes() / input_batch;
  const std::string queue_name(name);
  const bool gpu_input = queue_name.find("GPU input") != std::string::npos;
  size_t target_bytes =
      queue_name.find("display") != std::string::npos   ? kDisplayTargetBytes
      : queue_name.find("recording") != std::string::npos ? kCpuOutputTargetBytes
                                                          : kIntermediateTargetBytes;
  if (gpu_input) {
    const size_t available_vram = free_vram_ > vram_headroom_ ? free_vram_ - vram_headroom_ : 0;
    // The queue targets half of free VRAM without exceeding the 99% total-usage limit.
    target_bytes = std::min(free_vram_ / 100 * kGpuInputFreeVramPercent, available_vram);
  }
  const size_t total_vram_depth = element_size == 0
                                      ? 1
                                      : (total_vram_ / 100 * kMaxVramUsagePercent) / element_size;
  const size_t maximum_depth = gpu_input ? std::max<size_t>(1, total_vram_depth)
                               : queue_name.find("display") != std::string::npos ? 4
                               : queue_name.find("recording") != std::string::npos ? 8
                                                                                   : 8;
  // Keep the time-frequency queue's algorithm-determined floor and ceiling unchanged.
  const bool fixed_time_frequency = queue_name == "time-frequency input";
  const size_t floor_bytes = std::min(kQueueMinimumBytes, target_bytes);
  const size_t byte_floor = element_size == 0
                                ? 1
                                : std::max<size_t>(1, (floor_bytes + element_size - 1) / element_size);
  const size_t byte_ceiling = element_size == 0 ? 1 : std::max<size_t>(1, target_bytes / element_size);
  const size_t floor = fixed_time_frequency ? minimum_depth
                       : s_.queue_sizing_auto ? std::min(byte_floor, byte_ceiling)
                                              : requested_depth;
  const size_t ceiling = fixed_time_frequency
                             ? std::max(floor, requested_depth)
                             : s_.queue_sizing_auto
                                   ? std::max(floor, std::min(maximum_depth, byte_ceiling))
                                   : std::max(floor, requested_depth);
  const size_t depth   = s_.queue_sizing_auto
                             ? automatic_queue_depth(element_size, target_bytes, floor, ceiling)
                             : requested_depth;
  const size_t slots  = aligned_batch_slots(input_batch, static_cast<size_t>(output_stride), depth);
  auto         queued = batched_queue(desc, {static_cast<int>(depth), output_size, output_stride});
  const bool   device = desc.mem_loc == holoflow::core::MemLoc::Device;
  queue_plan_.push_back({name, device, depth, slots, element_size, 0, slots * element_size, floor,
                         ceiling, input_batch, static_cast<size_t>(output_stride), 0, false,
                         s_.queue_sizing_auto, target_bytes, queued.producer->vertex});
  return queued;
}

[[noreturn]] void GraphBuilder::Impl::throw_queue_plan_error(const std::string &reason) const {
  std::ostringstream message;
  message << reason << " (frame_bytes=" << frame_bytes_ << ")\n"
          << "name | kind | slots | bytes/slot | extra scratch | total\n";
  size_t vram = 0, ram = 0;
  for (const auto &entry : queue_plan_) {
    message << entry.name << " | " << (entry.device ? "device VRAM" : "pinned host") << " | "
            << entry.slots << " | " << entry.bytes_per_slot << " | " << entry.extra_bytes << " | "
            << entry.total_bytes << "\n";
    (entry.device ? vram : ram) += entry.total_bytes;
  }
  message << "VRAM total=" << vram << " available=" << free_vram_ << " headroom=" << vram_headroom_
          << "; host RAM total=" << ram << " available=" << free_ram_
          << " headroom=" << ram_headroom_;
  throw std::runtime_error(message.str());
}

void GraphBuilder::Impl::finish_queue_plan() {
  // Fit the whole plan by shrinking the largest auto-sized queues first.
  auto totals = [&]() {
    std::pair<size_t, size_t> result{};
    for (const auto &entry : queue_plan_)
      (entry.device ? result.first : result.second) += entry.total_bytes;
    return result;
  };
  const size_t vram_budget = free_vram_ > vram_headroom_ ? free_vram_ - vram_headroom_ : 0;
  const size_t ram_budget  = free_ram_ > ram_headroom_ ? free_ram_ - ram_headroom_ : 0;
  if (s_.queue_sizing_auto) {
    auto used = totals();
    while (used.first > vram_budget || used.second > ram_budget) {
      auto largest = queue_plan_.end();
      for (auto it = queue_plan_.begin(); it != queue_plan_.end(); ++it) {
        const size_t budget   = it->device ? vram_budget : ram_budget;
        const size_t occupied = it->device ? used.first : used.second;
        if (it->depth > it->floor && occupied > budget &&
            (largest == queue_plan_.end() || it->total_bytes > largest->total_bytes))
          largest = it;
      }
      if (largest == queue_plan_.end())
        break;
      --largest->depth;
      largest->slots =
          largest->sliding_average
              ? largest->depth + largest->extra_slots
              : aligned_batch_slots(largest->input_batch, largest->stride, largest->depth);
      largest->total_bytes = largest->slots * largest->bytes_per_slot +
                             (largest->sliding_average ? largest->bytes_per_slot : 0);
      used                 = totals();
    }
    for (auto &entry : queue_plan_) {
      if (!entry.adjustable || entry.depth == 0)
        continue;
      if (entry.sliding_average) {
        g_[entry.vertex].settings["target_capacity"] = entry.depth;
      } else {
        g_[entry.vertex].settings["target_capacity"] = entry.depth;
      }
    }
  }
  size_t             vram = 0, ram = 0;
  std::ostringstream depths;
  for (const auto &entry : queue_plan_) {
    (entry.device ? vram : ram) += entry.total_bytes;
    depths << entry.name << "=" << entry.depth << " (" << entry.slots << " slots, "
           << entry.total_bytes << " bytes, "
           << (entry.target_bytes ? "target=" + std::to_string(entry.target_bytes) + " bytes"
                                  : "algorithm-determined")
           << ", floor=" << entry.floor << ", ceiling=" << entry.ceiling << ") ";
  }
  if (vram > (free_vram_ > vram_headroom_ ? free_vram_ - vram_headroom_ : 0) ||
      ram > (free_ram_ > ram_headroom_ ? free_ram_ - ram_headroom_ : 0)) {
    throw_queue_plan_error("queue plan exceeds available memory");
  }
  logger()->info(
      "[GraphBuilder] Queue plan frame_bytes={} depths=[{}] VRAM planned={} total={} free={} "
      "headroom={} host planned={} total={} free={} headroom={}",
      frame_bytes_, depths.str(), vram, total_vram_, free_vram_, vram_headroom_, ram, total_ram_,
      free_ram_, ram_headroom_);
}

// -------------------------------------------------------------------------------------------------
// Top-level pipeline
// -------------------------------------------------------------------------------------------------

holoflow::core::GraphSpec GraphBuilder::Impl::build() {
  if (!std::isfinite(s_.signal_plot_time_window_seconds) ||
      s_.signal_plot_time_window_seconds <= 0.0) {
    throw std::invalid_argument("signal_plot_time_window_seconds must be positive and finite");
  }
  const double signal_plot_sample_time_seconds = s_.signal_plot_sample_time_seconds();
  if (!std::isfinite(signal_plot_sample_time_seconds) || signal_plot_sample_time_seconds <= 0.0) {
    throw std::invalid_argument("derived signal plot sample time must be positive and finite");
  }

  TDesc        H                  = build_acquisition();
  const size_t source_frame_bytes = H.num_bytes() / std::max<size_t>(1, H.shape.at(0));
  start_queue_plan(source_frame_bytes);

  if (s_.recording_method == RecordingMethod::RAW) {
    build_raw_record(H);
  }

  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    bool should_exit = build_raw_view(H);
    if (should_exit) {
      finish_queue_plan();
      return g_;
    }
  }

  if (s_.pp_spatial_contrast) {
    auto frames = build_spatial_contrast_input(H);
    if (s_.pp_spatial_contrast_dimensions != 2) {
      size_t source_cycle_frames = 0;
      if (s_.import_source == ImportSource::HOLOFILE && s_.load_batch > 0 &&
          s_.load_end > s_.load_begin) {
        const int selected_frames = s_.load_end - s_.load_begin;
        source_cycle_frames = static_cast<size_t>(selected_frames / s_.load_batch * s_.load_batch);
      }
      frames = temporal_window(frames, {.window_size         = static_cast<size_t>(s_.time_window),
                                        .stride_size         = static_cast<size_t>(s_.time_stride),
                                        .source_cycle_frames = source_cycle_frames});
    }
    build_spatial_contrast_view(frames);
    finish_queue_plan();
    return g_;
  }

  H = build_preprocessing(H);

  TDesc FH = build_time_frequency_analysis(H);

  bool filter_2d_standalone = s_.spacial_method != SpacialMethod::ANGULAR_SPECTRUM;
  if (s_.filter_2d && filter_2d_standalone) {
    FH = build_spatial_filter(FH);
  }

  if (s_.pp_accumulation <= 0) {
    throw std::invalid_argument("pp_accumulation must be positive");
  }

  // Batch queue with two readers. FH_delayed is used for Shack-Hartmann correction, FH_current is
  // used for propagation.
  auto         target_capacity   = std::max<size_t>(2, 2 * FH.shape.at(0));
  auto         window_size       = static_cast<size_t>(s_.pp_accumulation);
  const size_t dual_input_batch  = static_cast<size_t>(FH.shape.at(0));
  const size_t dual_element_size = FH.num_bytes() / dual_input_batch;
  const size_t dual_delay        = (window_size - 1) / 2;
  const size_t dual_minimum      = target_capacity + dual_input_batch + dual_delay + 1;
  const size_t dual_slots =
      ((dual_minimum + dual_input_batch - 1) / dual_input_batch) * dual_input_batch;
  const bool   dual_device = FH.mem_loc == holoflow::core::MemLoc::Device;
  const size_t dual_bytes  = dual_slots * dual_element_size + dual_element_size;
  add_queue_plan(
      "timing dual-reader", dual_device, dual_slots, dual_slots, dual_element_size,
      [dual_bytes](size_t) { return dual_bytes; }, [dual_slots](size_t) { return dual_slots; },
      dual_element_size);
  const auto  timing_outputs = dual_reader_batch_queue(FH, {target_capacity, window_size});
  TDesc       FH_current     = timing_outputs.at(0);
  const TDesc FH_delayed     = timing_outputs.at(1);

  if (s_.autofocus_enabled) {
    FH_current = build_aberration_correction(FH_current, FH_delayed);
  }

  TDesc FH_z = build_spatial_propagation(FH_current);

  build_xy_view(FH_z);

  if (s_.view_3d_cuts) {
    build_3d_cuts(FH_z);
  }

  finish_queue_plan();
  return g_;
}

// -------------------------------------------------------------------------------------------------
// Pipeline stages
// -------------------------------------------------------------------------------------------------

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_acquisition() {
  auto cam_path = s_.camera_config_path.string();

  if (s_.import_source == ImportSource::HOLOFILE) {
    return holofile_read({.path        = s_.load_path.string(),
                          .load_kind   = load_method_map_.at(s_.load_method),
                          .start_frame = s_.load_begin,
                          .end_frame   = s_.load_end,
                          .batch_size  = s_.load_batch,
                          .max_fps     = s_.load_fps_limit,
                          .keep_cursor = false});
  }

  std::optional<holocam::RecordSettings> record_settings = std::nullopt;
  if (s_.can_use_camera_record()) {
    record_settings = holocam::RecordSettings{.file_path       = s_.recording_path.string(),
                                              .recording_count = s_.recording_count,
                                              .pipeline_config = settings_to_old_json(s_),
                                              .record_queue_count =
                                                  static_cast<size_t>(s_.recording_queue_count)};
  }

  holotask::sources::CameraSettings settings{.cfg_path        = cam_path,
                                             .name            = to_string(s_.import_source),
                                             .record_settings = record_settings};
  return camera(settings);
}

void GraphBuilder::Impl::build_raw_record(const TDesc &H) {
  if (s_.can_use_camera_record())
    return;

  auto path  = s_.recording_path.string();
  auto count = s_.recording_count;
  if (s_.recording_format == "npy") {
    npyfile_write(H, {path, count, true});
  } else if (s_.recording_format == "holo") {
    holofile_write(H, {path, count, settings_to_old_json(s_), true});
  } else {
    auto video = memcpy(H, {holotask::syncs::MemcpySettings::Target::Host});
    ffmpeg_write(video, {path, count, static_cast<double>(s_.pp_fps), s_.recording_format,
                         s_.recording_codec, true, s_.recording_resize_algorithm});
  }
}

bool GraphBuilder::Impl::build_raw_view(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  auto H_disp = memcpy(H, {Host});
  auto H_view =
      planned_batch_queue("raw display", H_disp, s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                          s_.queue_sizing_auto ? 4 : 1, 1, 1);

  if (s_.raw_view) {
    xy_raw_display(H_view, {});
  }

  if (s_.view_type == ViewType::RAW) {
    xy_processed_display(H_view, {});
    return true;
  }

  return false;
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_preprocessing(TDesc H) {
  using MemLoc = holoflow::core::MemLoc;
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  if (H.mem_loc != MemLoc::Device) {
    H = memcpy(H, {Device});
    H = planned_batch_queue("GPU input", H, s_.queue_sizing_auto ? 64 : s_.gpu_in_size,
                            s_.queue_sizing_auto ? 16 : 1, s_.time_window, s_.time_stride);
  }

  return convert(H, {Target::F32, Strat::Real});
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_time_frequency_analysis(TDesc H) {
  // H enters as [T, Hy, Hx] (F32).
  // We first accumulate N_pre such windows into a batch, producing [N_pre, T, Hy, Hx].
  // Time-frequency analysis then operates along axis 1 (the T dimension).
  // The output is [N_pre, Nz, Hy, Hx], which feeds directly into the post-TFA queue.

  int     N_pre = 1;
  int64_t T     = static_cast<int64_t>(H.shape.at(0));
  int64_t Hy    = static_cast<int64_t>(H.shape.at(1));
  int64_t Hx    = static_cast<int64_t>(H.shape.at(2));

  H = reshape(H, {{1, T, Hy, Hx}, false});
  H = planned_batch_queue("time-frequency input", H, N_pre * 2, N_pre * 2, N_pre,
                          N_pre); // [N_pre, T, Hy, Hx]

  if (s_.time_method == TimeMethod::RFFT) {
    auto FH = rfft(H, {1}); // axis 1 = T dimension

    if (s_.view_3d_cuts) {
      return FH;
    }

    // Optimization: slice relevant components early. The rest is not used therefore this saves
    // further computations.
    FH = slice(FH, {{{}, holonp::SliceRange{s_.time_z_begin, s_.time_z_end}, {}, {}}});
    FH = copy(FH, {});
    return FH;
  }

  if (s_.time_method == TimeMethod::FFT) {
    auto FH = fft(H, {1});

    if (s_.view_3d_cuts) {
      return FH;
    }

    // Optimization: slice relevant components early, including the symmetric negative band. The
    // rest is not used therefore this saves further computations.
    auto pos_range = holonp::SliceRange{s_.time_z_begin, s_.time_z_end};
    auto FH_pos    = slice(FH, {{{}, pos_range, {}, {}}});

    auto neg_range = holonp::SliceRange{T - s_.time_z_end, T - s_.time_z_begin};
    auto FH_neg    = slice(FH, {{{}, neg_range, {}, {}}});

    FH = concatenate(std::array<TDesc, 2>{FH_pos, FH_neg}, {1});
    return FH;
  }

  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // PCA natively supports arbitrary leading batch dimensions (rank >= 3).
    // Input H: [N_pre, T, Hy, Hx] — the feature axis is shape[-3] = T.
    // Output FH: [N_pre, Nz, Hy, Hx] where Nz = z1 - z0.
    int  z0 = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int  z1 = s_.view_3d_cuts ? static_cast<int>(T) : s_.time_z_end;
    auto FH = pca(H, {z0, z1});
    return FH;
  }

  HOLOVIBES_UNREACHABLE();
}

// -------------------------------------------------------------------------------------------------
// Aberration correction
// -------------------------------------------------------------------------------------------------

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_aberration_correction(const TDesc &FH_current,
                                                                          const TDesc &FH_delayed) {
  if (s_.autofocus_nb_iter <= 0) {
    throw std::invalid_argument("autofocus_nb_iter must be positive");
  }
  if (s_.pp_accumulation > 1 && s_.autofocus_nb_iter > 1) {
    throw std::invalid_argument(
        "sliding Shack-Hartmann correction only supports one autofocus iteration");
  }

  auto                      corrected = FH_current;
  AberrationCorrectionState state;

  for (int pass = 0; pass < s_.autofocus_nb_iter; ++pass) {
    const bool   is_last_pass = pass == s_.autofocus_nb_iter - 1;
    const TDesc &delayed      = pass == 0 ? FH_delayed : corrected;
    corrected = build_aberration_correction_pass(corrected, delayed, is_last_pass, state);
  }

  // Decouple the final propagation so it can overlap the next Shack-Hartmann iteration.
  return planned_batch_queue("aberration correction", corrected, 2, 2, 1, 1);
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_aberration_correction_pass(const TDesc &FH_current,
                                                     const TDesc &FH_delayed, bool is_last_pass,
                                                     AberrationCorrectionState &state) {
  auto geometry      = shack_hartmann_geometry(FH_current);
  auto sensor_images = build_shack_hartmann_sensor(FH_current, geometry);

  bool output_xcorr = is_last_pass && s_.view_shack_hartmann_xcorr;
  auto slope_output = build_shack_hartmann_slopes(sensor_images, geometry, output_xcorr);

  if (is_last_pass && s_.view_shack_hartmann) {
    build_shack_hartmann_view(sensor_images, geometry);
  }

  if (slope_output.xcorr.has_value()) {
    build_shack_hartmann_xcorr_view(*slope_output.xcorr, geometry);
  }

  return build_zernike_correction(FH_delayed, slope_output.slopes, geometry, is_last_pass, state);
}

GraphBuilder::Impl::ShackHartmannGeometry
GraphBuilder::Impl::shack_hartmann_geometry(const TDesc &FH) const {
  if (s_.autofocus_nb_subaps <= 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be positive");
  }
  if ((s_.autofocus_nb_subaps % 2) == 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be odd");
  }

  const auto frame_width        = FH.shape.at(3);
  const auto frame_height       = FH.shape.at(2);
  const auto nb_subapertures    = static_cast<size_t>(s_.autofocus_nb_subaps);
  const auto subaperture_width  = frame_width / nb_subapertures;
  const auto subaperture_height = frame_height / nb_subapertures;

  if (subaperture_width == 0 || subaperture_height == 0) {
    throw std::invalid_argument("autofocus_nb_subaps is too large for the current frame size");
  }

  const auto  wavelength           = s_.spacial_lambda;
  const auto  pixel_pitch_x        = s_.spacial_pixel_size;
  const auto  pixel_pitch_y        = s_.spacial_pixel_size;
  const auto  propagation_distance = s_.spacial_z;
  const auto  width_m  = static_cast<float>(subaperture_width * nb_subapertures) * pixel_pitch_x;
  const auto  height_m = static_cast<float>(subaperture_height * nb_subapertures) * pixel_pitch_y;
  const float pupil_radius_m = 0.5f * std::min(width_m, height_m);

  return {.frame_width          = frame_width,
          .frame_height         = frame_height,
          .nb_subapertures      = nb_subapertures,
          .subaperture_width    = subaperture_width,
          .subaperture_height   = subaperture_height,
          .wavelength           = wavelength,
          .pixel_pitch_x        = pixel_pitch_x,
          .pixel_pitch_y        = pixel_pitch_y,
          .propagation_distance = propagation_distance,
          .pupil_radius_m       = pupil_radius_m};
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_shack_hartmann_sensor(const TDesc                 &FH,
                                                const ShackHartmannGeometry &geometry) {
  auto propagated = short_time_fresnel_diffraction(
      FH, geometry.subaperture_width, geometry.subaperture_height, geometry.subaperture_width,
      geometry.subaperture_height, geometry.wavelength, geometry.pixel_pitch_x,
      geometry.pixel_pitch_y, geometry.propagation_distance, PhaseReference::GLOBAL);

  auto sensor_images = mean(propagated, {{1}, false});
  sensor_images      = causal_slide_avg(sensor_images, {static_cast<size_t>(s_.pp_accumulation)});
  sensor_images      = fftshift(sensor_images, {{-2, -1}});

  if (!s_.pp_flatfield) {
    return sensor_images;
  }

  const auto [dy, dx] = fresnel_1fft_output_pitch(
      geometry.wavelength, geometry.propagation_distance, geometry.pixel_pitch_y,
      geometry.pixel_pitch_x, geometry.subaperture_height, geometry.subaperture_width);

  const auto cutoff   = s_.pp_flatfield_cutoff_period_m;
  const auto settings = flatfield_settings_from_cutoff_period(cutoff, dy, dx);
  sensor_images       = flatfield(sensor_images, settings);
  return sensor_images;
}

GraphBuilder::Impl::ShackHartmannSlopeOutput GraphBuilder::Impl::build_shack_hartmann_slopes(
    const TDesc &sensor_images, const ShackHartmannGeometry &geometry, bool output_xcorr) {
  using SlopeMode      = holotask::syncs::ShackHartmannSlopeMode;
  auto FullPairwise    = SlopeMode::FullPairwise;
  auto SingleReference = SlopeMode::SingleReference;

  const auto mode       = s_.autofocus_use_graph_laplacian ? FullPairwise : SingleReference;
  const bool emit_xcorr = output_xcorr && mode == SingleReference;

  auto outputs = shack_hartmann_slopes(
      sensor_images,
      {.mode               = mode,
       .lambda             = geometry.wavelength,
       .dx                 = geometry.pixel_pitch_x,
       .dy                 = geometry.pixel_pitch_y,
       .z                  = geometry.propagation_distance,
       .subaperture_height = geometry.subaperture_height,
       .subaperture_width  = geometry.subaperture_width,
       .stride_y           = geometry.subaperture_height,
       .stride_x           = geometry.subaperture_width,
       .correlation_roi    = {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f},
       .skip_subapertures_outside_pupil = s_.autofocus_skip_subapertures_outside_pupil,
       .output_xcorr_maps               = emit_xcorr});

  ShackHartmannSlopeOutput result{.slopes = outputs.at(0)};
  if (emit_xcorr) {
    result.xcorr = outputs.at(1);
  }

  return result;
}

void GraphBuilder::Impl::build_shack_hartmann_view(const TDesc                 &sensor_images,
                                                   const ShackHartmannGeometry &geometry) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;

  const auto height = static_cast<int64_t>(geometry.subaperture_height * geometry.nb_subapertures);
  const auto width  = static_cast<int64_t>(geometry.subaperture_width * geometry.nb_subapertures);

  auto display = normalize(sensor_images, {{-2, -1}, 0.0f, 255.0f});
  display      = transpose(display, {{0, 1, 3, 2, 4}});
  display      = reshape(display, {{1, height, width}});
  display      = convert(display, {Target::U8, Strat::Scaled});
  display      = planned_batch_queue("Shack-Hartmann display", display,
                                     s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                                     s_.queue_sizing_auto ? 4 : 1, 1, 1);
  shack_hartmann_display(display, {});
}

void GraphBuilder::Impl::build_shack_hartmann_xcorr_view(const TDesc                 &xcorr,
                                                         const ShackHartmannGeometry &geometry) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;

  auto display = fftshift(xcorr, {{-2, -1}});
  display      = normalize(display, {{-2, -1}, 0.0f, 255.0f});

  const auto height = static_cast<int64_t>(display.shape.at(3) * geometry.nb_subapertures);
  const auto width  = static_cast<int64_t>(display.shape.at(4) * geometry.nb_subapertures);

  display = convert(display, {Target::U8, Strat::Scaled});
  display = transpose(display, {{0, 1, 3, 2, 4}});
  display = reshape(display, {{1, height, width}});
  display = planned_batch_queue("Shack-Hartmann xcorr display", display,
                                s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                                s_.queue_sizing_auto ? 4 : 1, 1, 1);
  shack_hartmann_xcorr_display(display, {});
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_zernike_correction(const TDesc &FH, const TDesc &slopes,
                                             const ShackHartmannGeometry &geometry,
                                             bool is_last_pass, AberrationCorrectionState &state) {
  auto F32    = holoflow::core::DType::F32;
  auto Device = holoflow::core::MemLoc::Device;

  if (s_.autofocus_zernike_orders.empty()) {
    auto phase     = zeros({{1, geometry.frame_height, geometry.frame_width}, F32});
    auto corrected = correct_phase(FH, phase, {});

    if (is_last_pass && s_.view_zernike_phase) {
      zernike_phase_display(phase, {});
    }

    return corrected;
  }

  holotask::syncs::ZernikeFromSlopesSettings settings{
      .indexes                         = s_.autofocus_zernike_orders,
      .lambda                          = geometry.wavelength,
      .dx                              = geometry.pixel_pitch_x,
      .dy                              = geometry.pixel_pitch_y,
      .subaperture_height              = geometry.subaperture_height,
      .subaperture_width               = geometry.subaperture_width,
      .stride_y                        = geometry.subaperture_height,
      .stride_x                        = geometry.subaperture_width,
      .ny                              = 1,
      .nx                              = 1,
      .skip_subapertures_outside_pupil = s_.autofocus_skip_subapertures_outside_pupil};

  auto coeffs    = zernike_from_slopes(slopes, settings);
  coeffs         = slice(coeffs, {{0, 0, {}}});
  auto height    = static_cast<int>(geometry.frame_height);
  auto width     = static_cast<int>(geometry.frame_width);
  auto phase     = zernike_phase(coeffs, {s_.autofocus_zernike_orders, height, width, Device});
  auto corrected = correct_phase(FH, phase, {});

  const auto accumulate = [this](auto &cumulative, const auto &value) {
    cumulative = cumulative ? add(*cumulative, value, {}) : value;
  };

  accumulate(state.cumulative_coeffs_gpu, coeffs);
  accumulate(state.cumulative_phase_gpu, phase);

  if (is_last_pass) {
    build_zernike_outputs(state, geometry);
  }

  return corrected;
}

void GraphBuilder::Impl::build_zernike_outputs(const AberrationCorrectionState &state,
                                               const ShackHartmannGeometry     &geometry) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  HOLOVIBES_CHECK(state.cumulative_coeffs_gpu.has_value());
  HOLOVIBES_CHECK(state.cumulative_phase_gpu.has_value());

  const auto &coeffs = *state.cumulative_coeffs_gpu;
  const auto &phase  = *state.cumulative_phase_gpu;

  auto coeffs_reshaped = reshape(coeffs, {{1, static_cast<int64_t>(coeffs.shape.at(0))}});
  auto queue           = planned_batch_queue("Zernike coefficients display", coeffs_reshaped,
                                             s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                                             s_.queue_sizing_auto ? 4 : 1, 1, 1);
  coeffs_reshaped      = reshape(queue, {{static_cast<int64_t>(coeffs.shape.at(0))}});
  zernike_coefficients_display(coeffs_reshaped, {s_.autofocus_zernike_orders});

  if (s_.view_zernike_metrics) {
    auto coeffs_host = memcpy(coeffs_reshaped, {Host});
    zernike_history_display(coeffs_host, {
                                             s_.autofocus_zernike_orders,
                                             s_.signal_plot_time_window_seconds,
                                             s_.signal_plot_sample_time_seconds(),
                                             static_cast<size_t>(s_.pp_accumulation - 1),
                                         });
  }

  const bool defocus_included =
      std::ranges::find(s_.autofocus_zernike_orders, 4) != s_.autofocus_zernike_orders.end();

  if (defocus_included) {
    zernike_defocus_z_prop(coeffs, {s_.autofocus_zernike_orders, geometry.wavelength,
                                    geometry.propagation_distance, geometry.pupil_radius_m});
  }

  if (s_.view_zernike_phase) {
    auto height  = static_cast<int64_t>(geometry.frame_height);
    auto width   = static_cast<int64_t>(geometry.frame_width);
    auto display = copy(phase, {});
    display      = wrap2pi(display, {});
    display      = reshape(display, {{1, height, width}});
    display      = planned_batch_queue("Zernike phase display", display,
                                       s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                                       s_.queue_sizing_auto ? 4 : 1, 1, 1);
    zernike_phase_display(display, {});
  }
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::short_time_fresnel_diffraction(
    const TDesc &field, size_t win_w, size_t win_h, size_t stride_x, size_t stride_y, float lam,
    float dx, float dy, float z_prop, PhaseReference phase_ref, bool skip_phase_shift) {
  return GraphBuilderTasks::short_time_fresnel_diffraction(field,
                                                           {.lambda           = lam,
                                                            .dx               = dx,
                                                            .dy               = dy,
                                                            .z                = z_prop,
                                                            .win_h            = win_h,
                                                            .win_w            = win_w,
                                                            .stride_y         = stride_y,
                                                            .stride_x         = stride_x,
                                                            .phase_ref        = phase_ref,
                                                            .skip_phase_shift = skip_phase_shift,
                                                            .output_magnitude = true});
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_spatial_propagation(const TDesc &FH) {
  using Padding = holotask::syncs::AngularSpectrumSettings::Padding;
  using Filter  = holotask::syncs::AngularSpectrumSettings::Filter;

  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    if (s_.fresnel_use_numpy) {
      return fresnel_diffraction_numpy(FH);
    }
    return fresnel_diffraction(FH, {.lambda           = s_.spacial_lambda,
                                    .dx               = s_.spacial_pixel_size,
                                    .dy               = s_.spacial_pixel_size,
                                    .z                = s_.spacial_z,
                                    .axes             = {-2, -1},
                                    .output_magnitude = true});
  }

  if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    if (s_.autofocus_enabled) {
      throw std::logic_error{"Angular Spectrum is not supported with Shack-Hartmann autofocus"};
    }

    std::optional<Padding> padding;
    if (s_.asp_padding_enabled) {
      padding = Padding{.width = s_.asp_padded_width, .height = s_.asp_padded_height};
    }

    std::optional<Filter> filter;
    if (s_.filter_2d) {
      filter = Filter{.r_inner = s_.filter_r_inner,
                      .r_outer = s_.filter_r_outer,
                      .s_inner = s_.filter_smooth_inner,
                      .s_outer = s_.filter_smooth_outer};
    }

    return angular_spectrum(FH, {.lambda  = s_.spacial_lambda,
                                 .dx      = s_.spacial_pixel_size,
                                 .dy      = s_.spacial_pixel_size,
                                 .z       = s_.spacial_z,
                                 .filter  = filter,
                                 .padding = padding});
  }

  HOLOVIBES_UNREACHABLE();
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::fresnel_diffraction_numpy(const TDesc &FH) {
  if (FH.shape.size() < 2) {
    throw std::invalid_argument("NumPy Fresnel diffraction requires at least two dimensions");
  }
  if (s_.spacial_lambda <= 0.0f || s_.spacial_z == 0.0f || s_.spacial_pixel_size <= 0.0f) {
    throw std::invalid_argument("NumPy Fresnel diffraction requires valid optical parameters");
  }

  const size_t height   = FH.shape[FH.shape.size() - 2];
  const size_t width    = FH.shape[FH.shape.size() - 1];
  const size_t size     = std::max(height, width);
  const auto   offset_x = static_cast<double>((size - width) / 2);
  const auto   offset_y = static_cast<double>((size - height) / 2);
  const auto   half     = static_cast<double>(size) / 2.0;
  const auto   dx       = static_cast<double>(s_.spacial_pixel_size);

  const auto coordinate = [&](size_t extent, double offset) {
    const double start = (offset - half) * dx;
    const double stop  = start + (static_cast<double>(extent) - 0.5) * dx;
    return arange({.start  = start,
                   .stop   = stop,
                   .step   = dx,
                   .dtype  = holoflow::core::DType::F32,
                   .device = holoflow::core::MemLoc::Device});
  };

  const auto                 x = coordinate(width, offset_x);
  const auto                 y = coordinate(height, offset_y);
  const std::array<TDesc, 2> axes{x, y};
  const auto                 grids = meshgrid(axes, {.indexing = holonp::MeshgridIndexing::XY});
  const auto                 radius_squared = add(square(grids[0], {}), square(grids[1], {}), {});

  const auto phase_scale =
      asarray({.value  = static_cast<double>(std::numbers::pi_v<float>) /
                         (static_cast<double>(s_.spacial_lambda) * s_.spacial_z),
               .dtype  = holoflow::core::DType::F32,
               .device = holoflow::core::MemLoc::Device});
  const auto imaginary = asarray({.value  = 0.0,
                                  .imag   = 1.0,
                                  .dtype  = holoflow::core::DType::CF32,
                                  .device = holoflow::core::MemLoc::Device});
  const auto lens = exp(multiply(multiply(radius_squared, phase_scale, {}), imaginary, {}), {});
  const auto propagated = fft2(multiply(FH, lens, {}), {});
  return abs(propagated, {});
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_spatial_filter(const TDesc &FH_z) {
  return filter_2d(FH_z, {
                             s_.filter_r_inner,
                             s_.filter_r_outer,
                             s_.filter_smooth_inner,
                             s_.filter_smooth_outer,
                         });
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_freq_weights() {
  auto N  = static_cast<double>(s_.time_window);
  auto fs = s_.input_sampling_frequency_hz;
  auto df = fs / N;
  auto f0 = s_.time_z_begin * df;
  auto f1 = s_.time_z_end * df;

  if (s_.view_3d_cuts) {
    throw std::logic_error{"Frequency weights are not supported when 3D cuts are enabled"};
  }

  if (s_.time_method == TimeMethod::FFT) {
    auto freqs_pos = arange({f0, f1, df, holoflow::core::DType::F32});
    auto freqs_neg = arange({f0 - fs, f1 - fs, df, holoflow::core::DType::F32});
    auto freqs     = concatenate(std::array<TDesc, 2>{freqs_pos, freqs_neg}, {0});
    return freqs;
  }

  if (s_.time_method == TimeMethod::RFFT ||
      s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    auto freqs = arange({f0, f1, df, holoflow::core::DType::F32});
    return freqs;
  }

  HOLOVIBES_UNREACHABLE();
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_spatial_contrast_input(TDesc H) {
  using MemLoc = holoflow::core::MemLoc;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  if (H.mem_loc != MemLoc::Device) {
    H = memcpy(H, {Device});
  }
  H = planned_batch_queue("spatial contrast GPU input", H,
                          s_.queue_sizing_auto ? 64 : s_.gpu_in_size, s_.queue_sizing_auto ? 16 : 1,
                          1, 1);
  return ascontiguousarray(H, {});
}

void GraphBuilder::Impl::build_spatial_contrast_view(const TDesc &frames) {
  const size_t temporal_window_size =
      s_.pp_spatial_contrast_dimensions != 2 ? static_cast<size_t>(s_.time_window) : size_t{1};
  if (frames.rank() != 3 || frames.shape.at(0) != temporal_window_size) {
    throw std::logic_error{"Spatial LSCI input has an unexpected temporal window shape"};
  }

  auto result = spatial_contrast(
      frames, {.window_size =
                   s_.pp_spatial_contrast_dimensions == 1 ? 1 : s_.pp_spatial_contrast_window_size,
               .temporal_window_size = static_cast<int>(temporal_window_size),
               .method               = s_.pp_spatial_contrast_method == 1
                                           ? holotask::syncs::SpatialContrastMethod::FFT
                                           : holotask::syncs::SpatialContrastMethod::Kernel});
  if (s_.pp_flatfield) {
    // Spatial contrast uses the camera grid, so derive the filter from the source pixel pitch.
    const auto pitch = s_.spacial_pixel_size;
    result           = flatfield(result, flatfield_settings_from_cutoff_period(
                                             s_.pp_flatfield_cutoff_period_m, pitch, pitch));
  }
  build_xy_output(result, 0, true);
}

void GraphBuilder::Impl::build_xy_view(const TDesc &FH_z) {
  TDesc      result;
  const bool is_magnitude = FH_z.dtype == holoflow::core::DType::F32;

  if (s_.moment_type == MomentType::M0) {
    result = is_magnitude ? mean(FH_z, {{-3}, true}) : mean_abs(FH_z, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M1) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = is_magnitude ? FH_z : abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M2) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = is_magnitude ? FH_z : abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    freqs         = multiply(freqs, freqs, {});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  else {
    HOLOVIBES_UNREACHABLE();
  }

  if (s_.pp_fft_shift) {
    result = fftshift(result, {{-2, -1}});
  }

  if (s_.pp_flatfield) {
    auto cutoff         = s_.pp_flatfield_cutoff_period_m;
    const auto [dy, dx] = post_propagation_pitch(s_, FH_z);
    result              = flatfield(result, flatfield_settings_from_cutoff_period(cutoff, dy, dx));
  }

  // Average the small temporal batch before registration and preview accumulation.
  result = mean(result, {{0}, false}); // [1, H, W]

  if (s_.pp_registration) {
    using Target        = holotask::syncs::ConversionSettings::Target;
    using Strat         = holotask::syncs::ConversionSettings::Strategy;
    const auto outputs  = registration_outputs(result, {.radius = s_.pp_registration_radius});
    result              = outputs.at(0);
    auto output_display = fftshift(outputs.at(1), {{-2, -1}});
    output_display      = normalize(output_display, {{-2, -1}, 0.0f, 255.0f});
    const auto output_converted = convert(output_display, {Target::U8, Strat::Scaled});
    registration_xcorr_display(output_converted, {});
  }

  auto discard_first = s_.autofocus_enabled ? static_cast<size_t>(s_.pp_accumulation - 1) : 0;
  build_xy_output(result, discard_first);
}

void GraphBuilder::Impl::build_xy_output(TDesc result, size_t discard_first,
                                         bool fixed_contrast_range) {
  using Target                 = holotask::syncs::ConversionSettings::Target;
  using Strat                  = holotask::syncs::ConversionSettings::Strategy;
  using SlidingAverageSettings = holotask::asyncs::SlidingAverageSettings;
  using PctClipSettings        = holotask::syncs::PctClipSettings;
  using Ellipse                = PctClipSettings::Ellipse;
  using holotask::sinks::HolofileSettings;
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  if (s_.pp_accumulation <= 0) {
    throw std::invalid_argument{"pp_accumulation must be positive"};
  }

  auto         window_size   = static_cast<size_t>(s_.pp_accumulation);
  const size_t result_bytes  = result.num_bytes() / static_cast<size_t>(result.shape.at(0));
  const size_t sliding_floor = 8;
  const size_t target_capacity =
      s_.queue_sizing_auto
          ? automatic_queue_depth(result_bytes, kSlidingTargetBytes, sliding_floor, 32)
          : size_t{8};
  auto slide_settings = SlidingAverageSettings{target_capacity, window_size, discard_first};
  result              = slide_avg(result, slide_settings);
  queue_plan_.push_back(
      {"sliding average", true, target_capacity, target_capacity + window_size, result_bytes,
       result_bytes, (target_capacity + window_size + 1) * result_bytes, sliding_floor, 32, 1, 1,
       window_size, true, s_.queue_sizing_auto, kSlidingTargetBytes, result.producer->vertex});

  if (s_.pp_convolution) {
    throw std::logic_error{"Convolution is currently not supported"};
  }

  // Clip the shared float result before it branches to display and processed recording.
  if (s_.pp_pctclip) {
    Ellipse         roi{0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f};
    PctClipSettings pct_clip_settings{s_.pp_pctclip_lower, s_.pp_pctclip_upper, roi};
    result = pct_clip(result, pct_clip_settings);
  }

  // Display and recording use independent queues so neither consumer drains the other.
  auto convert_for_output = [&](const TDesc &input) {
    if (fixed_contrast_range) {
      return convert(input, {Target::U8, Strat::FixedRange, s_.pp_spatial_contrast_display_max,
                             s_.pp_spatial_contrast_invert_display});
    }
    return convert(input, {Target::U8, Strat::Scaled});
  };

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    auto result_rec = convert_for_output(result);
    result_rec      = memcpy(result_rec, {Host});
    result_rec      = planned_batch_queue("processed recording", result_rec,
                                          s_.queue_sizing_auto ? 8 : s_.cpu_out_size,
                                          s_.queue_sizing_auto ? 4 : 1, 1, 1);

    auto path  = s_.recording_path.string();
    auto count = s_.recording_count;
    if (s_.recording_format == "npy") {
      npyfile_write(result_rec, {path, count, true});
    } else if (s_.recording_format == "holo") {
      holofile_write(result_rec, {path, count, settings_to_old_json(s_), true});
    } else {
      const bool resize_to_square = s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION;
      ffmpeg_write(result_rec,
                   {path, count, static_cast<double>(s_.pp_fps), s_.recording_format,
                    s_.recording_codec, resize_to_square, s_.recording_resize_algorithm});
    }
  }

  auto display_result = convert_for_output(result);
  display_result      = planned_batch_queue("processed display", display_result,
                                            s_.queue_sizing_auto ? 4 : s_.gpu_out_size,
                                            s_.queue_sizing_auto ? 2 : 1, 1, 1);
  xy_processed_display(display_result, {});
}

void GraphBuilder::Impl::build_3d_cuts(const TDesc &FH_z) {
  (void)FH_z;
  throw std::logic_error{"3D cuts are currently not supported in GraphBuilder"};
}

// -----------------------------------------------------------------------------
// Helpers definitions
// -----------------------------------------------------------------------------

namespace {

constexpr float kFlatfieldCutoffConstant = 0.187f;

holotask::syncs::FlatfieldSettings flatfield_settings_from_cutoff_period(float cutoff_period_m,
                                                                         float dy_m, float dx_m) {
  if (cutoff_period_m <= 0.0f || dy_m <= 0.0f || dx_m <= 0.0f) {
    throw std::invalid_argument("flatfield cutoff period and image pitches must be positive");
  }

  // The UI exposes a physical cutoff period, not the Gaussian sigma. The 0.187 factor follows
  // the 50% amplitude transition convention for the Gaussian high-pass
  // H_hp(f) = 1 - exp(-2*pi^2*sigma^2*f^2), so f50 = 0.187 / sigma_px and
  // sigma_px = 0.187 * period_px. This is a convention, not a hard cutoff.
  // Axis order follows image layout: y uses dy on axis -2, x uses dx on axis -1.
  return {
      .sigma_y = kFlatfieldCutoffConstant * cutoff_period_m / dy_m,
      .sigma_x = kFlatfieldCutoffConstant * cutoff_period_m / dx_m,
  };
}

std::pair<float, float> fresnel_1fft_output_pitch(float wavelength_m, float z_m, float dy_in_m,
                                                  float dx_in_m, size_t ny, size_t nx) {
  if (wavelength_m <= 0.0f || dy_in_m <= 0.0f || dx_in_m <= 0.0f || ny == 0 || nx == 0) {
    throw std::invalid_argument(
        "Fresnel output pitch requires positive wavelength, pitch, and shape");
  }

  const float z_abs = std::abs(z_m);
  if (z_abs <= 0.0f) {
    throw std::invalid_argument("Fresnel output pitch requires non-zero propagation distance");
  }
  return {
      wavelength_m * z_abs / (static_cast<float>(ny) * dy_in_m),
      wavelength_m * z_abs / (static_cast<float>(nx) * dx_in_m),
  };
}

std::pair<float, float> post_propagation_pitch(const Settings              &settings,
                                               const holoflow::core::TDesc &desc) {
  if (settings.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    const auto rank = desc.shape.size();
    return fresnel_1fft_output_pitch(settings.spacial_lambda, settings.spacial_z,
                                     settings.spacial_pixel_size, settings.spacial_pixel_size,
                                     desc.shape.at(rank - 2), desc.shape.at(rank - 1));
  }

  return {settings.spacial_pixel_size, settings.spacial_pixel_size};
}

} // namespace

} // namespace holovibes::pipeline
