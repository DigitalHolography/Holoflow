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

#include "holotask/syncs/shack_hartmann_slopes.hh"

#include <array>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "logger.hh"
#include "syncs/phase_correlation.cuh"
#include "syncs/shack_hartmann_geometry.hh"
#include "syncs/shack_hartmann_slopes_full_pairwise.cuh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, ShackHartmannSlopeMode mode) {
  switch (mode) {
  case ShackHartmannSlopeMode::SingleReference:
    j = "single_reference";
    break;
  case ShackHartmannSlopeMode::FullPairwise:
    j = "full_pairwise";
    break;
  }
}

void from_json(const nlohmann::json &j, ShackHartmannSlopeMode &mode) {
  const auto value = j.get<std::string>();
  if (value == "single_reference") {
    mode = ShackHartmannSlopeMode::SingleReference;
  } else if (value == "full_pairwise") {
    mode = ShackHartmannSlopeMode::FullPairwise;
  } else {
    throw std::invalid_argument("Unknown Shack-Hartmann slope mode: " + value);
  }
}

void to_json(nlohmann::json &j, const ShackHartmannSlopeSettings &s) {
  j = nlohmann::json{
      {"mode", s.mode},
      {"lambda", s.lambda},
      {"dx", s.dx},
      {"dy", s.dy},
      {"z", s.z},
      {"subaperture_height", s.subaperture_height},
      {"subaperture_width", s.subaperture_width},
      {"stride_y", s.stride_y},
      {"stride_x", s.stride_x},
      {"correlation_roi", s.correlation_roi},
      {"skip_subapertures_outside_pupil", s.skip_subapertures_outside_pupil},
      {"output_xcorr_maps", s.output_xcorr_maps},
      {"pair_batch_size", s.pair_batch_size},
  };
}

void from_json(const nlohmann::json &j, ShackHartmannSlopeSettings &s) {
  s.mode = j.value("mode", ShackHartmannSlopeMode::SingleReference);
  j.at("lambda").get_to(s.lambda);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("z").get_to(s.z);
  j.at("subaperture_height").get_to(s.subaperture_height);
  j.at("subaperture_width").get_to(s.subaperture_width);
  j.at("stride_y").get_to(s.stride_y);
  j.at("stride_x").get_to(s.stride_x);
  if (j.contains("correlation_roi")) {
    j.at("correlation_roi").get_to(s.correlation_roi);
  }
  s.skip_subapertures_outside_pupil = j.value("skip_subapertures_outside_pupil", true);
  s.output_xcorr_maps               = j.value("output_xcorr_maps", false);
  s.pair_batch_size                 = j.value("pair_batch_size", size_t{256});
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void check(bool condition, const std::string &message) {
  if (!condition) {
    logger()->error("[ShackHartmannSlopesFactory::infer] error: {}", message);
    throw std::invalid_argument("ShackHartmannSlopesFactory inference error: " + message);
  }
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

holoflow::core::TDesc center_reference_desc(const holoflow::core::TDesc &input) {
  const size_t reference_sy = input.shape[1] / 2;
  const size_t reference_sx = input.shape[2] / 2;
  const size_t offset =
      input.offset + reference_sy * input.strides[1] + reference_sx * input.strides[2];

  return holoflow::core::TDesc({input.shape[0], input.shape[3], input.shape[4]}, input.dtype,
                               input.mem_loc,
                               {input.strides[0], input.strides[3], input.strides[4]}, offset);
}

detail::ShackHartmannGeometrySettings
geometry_settings(const holoflow::core::TDesc &input, const ShackHartmannSlopeSettings &settings) {
  return {
      .sy                              = input.shape[1],
      .sx                              = input.shape[2],
      .subaperture_height              = settings.subaperture_height,
      .subaperture_width               = settings.subaperture_width,
      .stride_y                        = settings.stride_y,
      .stride_x                        = settings.stride_x,
      .dy                              = settings.dy,
      .dx                              = settings.dx,
      .skip_subapertures_outside_pupil = settings.skip_subapertures_outside_pupil,
  };
}

// measurement(i, j) = g[i] - g[j]. The CrossCorrelation2 convention places a moving image
// translated by (+dx, +dy) from its reference at the corresponding positive circular offset.
__global__ void recover_phase_correlation_peaks(const float *__restrict__ maps,
                                                float2 *__restrict__ shifts, size_t map_count,
                                                size_t height, size_t width) {
  const size_t map_index = blockIdx.x;
  if (map_index >= map_count) {
    return;
  }

  const size_t pixels_per_map = height * width;
  const float *map            = maps + map_index * pixels_per_map;
  detail::PhaseCorrelationPeak local_peak{-FLT_MAX, 0};
  for (size_t pixel = threadIdx.x; pixel < pixels_per_map; pixel += blockDim.x) {
    local_peak = detail::select_phase_correlation_peak(local_peak, {map[pixel], pixel});
  }

  __shared__ detail::PhaseCorrelationPeak
      shared_peaks[detail::kPhaseCorrelationPeakBlockSize];
  const auto peak = detail::reduce_phase_correlation_peak(local_peak, shared_peaks);
  if (threadIdx.x != 0) {
    return;
  }

  const size_t peak_y  = peak.index / width;
  const size_t peak_x  = peak.index % width;
  const size_t x_minus = (peak_x + width - 1) % width;
  const size_t x_plus  = (peak_x + 1) % width;
  const size_t y_minus = (peak_y + height - 1) % height;
  const size_t y_plus  = (peak_y + 1) % height;

  const float dx_subpixel = detail::parabolic_peak_offset(
      map[peak_y * width + x_minus], map[peak_y * width + peak_x], map[peak_y * width + x_plus]);
  const float dy_subpixel = detail::parabolic_peak_offset(
      map[y_minus * width + peak_x], map[peak_y * width + peak_x], map[y_plus * width + peak_x]);

  shifts[map_index] = {
      detail::circular_signed_coordinate(peak_x, width) + dx_subpixel,
      detail::circular_signed_coordinate(peak_y, height) + dy_subpixel,
  };
}

__global__ void recover_zero_mean_slopes(const float2 *__restrict__ measured_shifts,
                                         const std::uint8_t *__restrict__ active,
                                         float *__restrict__ slopes, size_t sample_count,
                                         size_t center_index, float slope_per_pixel_x,
                                         float slope_per_pixel_y) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const float2 reference_bias = measured_shifts[center_index];
  float        sum_x          = 0.0f;
  float        sum_y          = 0.0f;
  size_t       active_count   = 0;

  for (size_t i = 0; i < sample_count; ++i) {
    if (active[i] == 0) {
      continue;
    }
    sum_x += (measured_shifts[i].x - reference_bias.x) * slope_per_pixel_x;
    sum_y += (measured_shifts[i].y - reference_bias.y) * slope_per_pixel_y;
    ++active_count;
  }

  const float mean_x = sum_x / static_cast<float>(active_count);
  const float mean_y = sum_y / static_cast<float>(active_count);

  for (size_t i = 0; i < sample_count; ++i) {
    if (active[i] != 0) {
      slopes[2 * i]     = (measured_shifts[i].x - reference_bias.x) * slope_per_pixel_x - mean_x;
      slopes[2 * i + 1] = (measured_shifts[i].y - reference_bias.y) * slope_per_pixel_y - mean_y;
    } else {
      slopes[2 * i]     = 0.0f;
      slopes[2 * i + 1] = 0.0f;
    }
  }
}

// -------------------------------------------------------------------------------------------------
// ShackHartmannSlopes task implementation
// -------------------------------------------------------------------------------------------------

class ShackHartmannSlopes : public detail::ShackHartmannSlopesTaskBase {
public:
  ShackHartmannSlopes(ShackHartmannSlopeSettings settings, holoflow::core::TDesc input_desc,
                      holoflow::core::TDesc                      reference_desc,
                      std::unique_ptr<holoflow::core::ISyncTask> cross_correlation,
                      std::unique_ptr<holoflow::core::Tensor>    xcorr_scratch,
                      DevPtr<float2> measured_shifts, DevPtr<std::uint8_t> active,
                      cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)),
        reference_desc_(std::move(reference_desc)),
        cross_correlation_(std::move(cross_correlation)), xcorr_scratch_(std::move(xcorr_scratch)),
        measured_shifts_(std::move(measured_shifts)), active_(std::move(active)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto reference_view = ctx.inputs[0];
    reference_view.desc = reference_desc_;

    auto xcorr_view = settings_.output_xcorr_maps ? ctx.outputs[1] : xcorr_scratch_->view();

    std::array<holoflow::core::TView, 2> xcorr_inputs{ctx.inputs[0], reference_view};
    std::array<holoflow::core::TView, 1> xcorr_outputs{xcorr_view};
    holoflow::core::SyncCtx              xcorr_ctx{
        .inputs       = xcorr_inputs,
        .outputs      = xcorr_outputs,
        .cancelled    = ctx.cancelled,
        .event_writer = ctx.event_writer,
        .event_reader = ctx.event_reader,
    };

    const auto result = cross_correlation_->execute(xcorr_ctx);
    if (result != holoflow::core::OpResult::Ok) {
      return result;
    }

    const size_t sy           = input_desc_.shape[1];
    const size_t sx           = input_desc_.shape[2];
    const size_t height       = input_desc_.shape[3];
    const size_t width        = input_desc_.shape[4];
    const size_t sample_count = sy * sx;

    constexpr unsigned int block = detail::kPhaseCorrelationPeakBlockSize;
    recover_phase_correlation_peaks<<<static_cast<unsigned int>(sample_count), block, 0, stream_>>>(
        reinterpret_cast<const float *>(xcorr_view.data()), measured_shifts_.get(), sample_count,
        height, width);

    const float delta_out_x =
        settings_.lambda * settings_.z / (static_cast<float>(width) * settings_.dx);
    const float delta_out_y =
        settings_.lambda * settings_.z / (static_cast<float>(height) * settings_.dy);

    const size_t center_index = (sy / 2) * sx + sx / 2;
    recover_zero_mean_slopes<<<1, 1, 0, stream_>>>(
        measured_shifts_.get(), active_.get(), reinterpret_cast<float *>(ctx.outputs[0].data()),
        sample_count, center_index, delta_out_x / settings_.z, delta_out_y / settings_.z);

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const ShackHartmannSlopeSettings &settings() const override { return settings_; }
  const holoflow::core::TDesc      &input_desc() const override { return input_desc_; }

  void update_stream(cudaStream_t stream) override {
    if (stream_ == stream) {
      return;
    }

    const CrossCorrelation2Settings xcorr_settings{
        .axes = {-2, -1},
        .norm = FftNorm::Backward,
        .roi  = settings_.correlation_roi,
    };
    const std::array input_descs{input_desc_, reference_desc_};
    cross_correlation_ = CrossCorrelation2Factory{}.update(
        std::move(cross_correlation_), input_descs, xcorr_settings, {.stream = stream});
    stream_ = stream;
  }

private:
  ShackHartmannSlopeSettings                 settings_;
  holoflow::core::TDesc                      input_desc_;
  holoflow::core::TDesc                      reference_desc_;
  std::unique_ptr<holoflow::core::ISyncTask> cross_correlation_;
  std::unique_ptr<holoflow::core::Tensor>    xcorr_scratch_;
  DevPtr<float2>                             measured_shifts_;
  DevPtr<std::uint8_t>                       active_;
  cudaStream_t                               stream_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// ShackHartmannSlopesFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ShackHartmannSlopesFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                  const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ShackHartmannSlopeSettings>();

  check(input_descs.size() == 1, "task must have exactly one input");
  const auto &input = input_descs[0];
  check(input.mem_loc == holoflow::core::MemLoc::Device, "input memory location must be Device");
  check(input.dtype == holoflow::core::DType::F32, "input dtype must be F32");
  check(input.rank() == 5, "input rank must be 5");
  check(input.shape[0] == 1, "only batch size 1 is supported");
  check(input.shape[1] > 0 && input.shape[2] > 0, "subaperture grid dimensions must be positive");
  if (settings.mode == ShackHartmannSlopeMode::SingleReference) {
    check((input.shape[1] % 2) == 1 && (input.shape[2] % 2) == 1,
          "subaperture grid dimensions must be odd for an unambiguous center reference");
  }
  check(input.shape[3] >= 2 && input.shape[4] >= 2, "reconstruction dimensions must be at least 2");

  check(settings.lambda > 0.0f, "wavelength must be > 0");
  check(settings.dx > 0.0f && settings.dy > 0.0f, "pixel pitches must be > 0");
  check(settings.z > 0.0f, "propagation distance must be > 0");
  check(settings.subaperture_height > 0 && settings.subaperture_width > 0,
        "subaperture dimensions must be positive");
  check(settings.stride_y > 0 && settings.stride_x > 0, "strides must be positive");
  check(settings.correlation_roi.rx > 0.0f && settings.correlation_roi.ry > 0.0f,
        "correlation ROI radii must be positive");
  check(settings.pair_batch_size > 0, "pair_batch_size must be positive");
  check(!(settings.mode == ShackHartmannSlopeMode::FullPairwise && settings.output_xcorr_maps),
        "FullPairwise correlation-map output is not supported");

  const auto geometry = detail::make_shack_hartmann_geometry(geometry_settings(input, settings));
  check(geometry.pupil_radius_m > 0.0f, "pupil radius must be positive");
  const size_t active_count = static_cast<size_t>(
      std::ranges::count_if(geometry.samples, &detail::ShackHartmannSample::active));
  check(active_count > 0, "at least one subaperture must be active");
  if (settings.mode == ShackHartmannSlopeMode::FullPairwise) {
    check(active_count >= 2, "FullPairwise mode requires at least two active subapertures");
    check(active_count - 1 <= std::numeric_limits<size_t>::max() / active_count,
          "FullPairwise edge count overflows size_t");
  }

  holoflow::core::TDesc slopes_desc({1, input.shape[1], input.shape[2], 2},
                                    holoflow::core::DType::F32, holoflow::core::MemLoc::Device);
  std::vector<holoflow::core::TDesc> outputs{slopes_desc};
  if (settings.output_xcorr_maps) {
    outputs.emplace_back(input.shape, holoflow::core::DType::F32, holoflow::core::MemLoc::Device);
  }

  return {
      .input_descs   = {input},
      .output_descs  = std::move(outputs),
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = std::vector<bool>(settings.output_xcorr_maps ? 2 : 1, false),
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ShackHartmannSlopesFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json                  &jsettings,
                                   const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto  settings = jsettings.get<ShackHartmannSlopeSettings>();
  const auto &input    = input_descs[0];
  const auto  geometry = detail::make_shack_hartmann_geometry(geometry_settings(input, settings));

  if (settings.mode == ShackHartmannSlopeMode::FullPairwise) {
    return detail::make_full_pairwise_shack_hartmann_slopes(settings, input, geometry, ctx);
  }

  const auto reference = center_reference_desc(input);

  const CrossCorrelation2Settings xcorr_settings{
      .axes = {-2, -1},
      .norm = FftNorm::Backward,
      .roi  = settings.correlation_roi,
  };
  const std::array xcorr_inputs{input, reference};
  auto cross_correlation = CrossCorrelation2Factory{}.create(xcorr_inputs, xcorr_settings, ctx);

  std::unique_ptr<holoflow::core::Tensor> xcorr_scratch;
  if (!settings.output_xcorr_maps) {
    xcorr_scratch = std::make_unique<holoflow::core::Tensor>(holoflow::core::TDesc(
        input.shape, holoflow::core::DType::F32, holoflow::core::MemLoc::Device));
  }

  const size_t sample_count    = input.shape[1] * input.shape[2];
  auto         measured_shifts = curaii::make_unique_device_ptr<float2>(sample_count, ctx.stream);
  auto         active = curaii::make_unique_device_ptr<std::uint8_t>(sample_count, ctx.stream);

  std::vector<std::uint8_t> active_host;
  active_host.reserve(sample_count);
  for (const auto &sample : geometry.samples) {
    active_host.push_back(sample.active ? std::uint8_t{1} : std::uint8_t{0});
  }
  CUDA_CHECK(cudaMemcpyAsync(active.get(), active_host.data(), active_host.size(),
                             cudaMemcpyHostToDevice, ctx.stream));

  return std::make_unique<ShackHartmannSlopes>(
      settings, input, reference, std::move(cross_correlation), std::move(xcorr_scratch),
      std::move(measured_shifts), std::move(active), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ShackHartmannSlopesFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                   std::span<const holoflow::core::TDesc>     input_descs,
                                   const nlohmann::json                      &jsettings,
                                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_slopes = dynamic_cast<detail::ShackHartmannSlopesTaskBase *>(old_task.get());
  if (old_slopes != nullptr && input_descs.size() == 1) {
    const auto settings = jsettings.get<ShackHartmannSlopeSettings>();
    if (settings == old_slopes->settings() && same_desc(input_descs[0], old_slopes->input_desc())) {
      old_slopes->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
