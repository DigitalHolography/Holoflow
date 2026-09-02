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

#include "holotask/syncs/zernike_from_slopes.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "logger.hh"
#include "syncs/shack_hartmann_geometry.hh"
#include "syncs/zernike_from_slopes_gpu.cuh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ZernikeFromSlopesSettings &s) {
  j = nlohmann::json{
      {"indexes", s.indexes},
      {"lambda", s.lambda},
      {"dx", s.dx},
      {"dy", s.dy},
      {"subaperture_height", s.subaperture_height},
      {"subaperture_width", s.subaperture_width},
      {"stride_y", s.stride_y},
      {"stride_x", s.stride_x},
      {"ny", s.ny},
      {"nx", s.nx},
      {"skip_subapertures_outside_pupil", s.skip_subapertures_outside_pupil},
  };
}

void from_json(const nlohmann::json &j, ZernikeFromSlopesSettings &s) {
  s.indexes.clear();
  if (j.contains("indexes")) {
    j.at("indexes").get_to(s.indexes);
  } else if (j.contains("indices")) {
    j.at("indices").get_to(s.indexes);
  }

  j.at("lambda").get_to(s.lambda);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("subaperture_height").get_to(s.subaperture_height);
  j.at("subaperture_width").get_to(s.subaperture_width);
  j.at("stride_y").get_to(s.stride_y);
  j.at("stride_x").get_to(s.stride_x);
  s.ny                              = j.value("ny", size_t{1});
  s.nx                              = j.value("nx", size_t{1});
  s.skip_subapertures_outside_pupil = j.value("skip_subapertures_outside_pupil", true);
}

namespace {

constexpr size_t kMaxSupportedModes = 9; // Noll indices 2..10

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void check(bool condition, const std::string &message) {
  if (!condition) {
    logger()->error("[ZernikeFromSlopesFactory::infer] error: {}", message);
    throw std::invalid_argument("ZernikeFromSlopesFactory inference error: " + message);
  }
}

struct ZernikeDerivative {
  float dx_n = 0.0f;
  float dy_n = 0.0f;
};

ZernikeDerivative eval_zernike_noll_derivative(int noll_index, float x_n, float y_n) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);
  const float sqrt8 = std::sqrt(8.0f);

  switch (noll_index) {
  case 2:
    return {2.0f, 0.0f};
  case 3:
    return {0.0f, 2.0f};
  case 4:
    return {4.0f * sqrt3 * x_n, 4.0f * sqrt3 * y_n};
  case 5:
    return {2.0f * sqrt6 * y_n, 2.0f * sqrt6 * x_n};
  case 6:
    return {2.0f * sqrt6 * x_n, -2.0f * sqrt6 * y_n};
  case 7:
    return {6.0f * sqrt8 * x_n * y_n, 3.0f * sqrt8 * (x_n * x_n - y_n * y_n)};
  case 8:
    return {6.0f * sqrt8 * x_n * y_n, sqrt8 * (3.0f * x_n * x_n + 9.0f * y_n * y_n - 2.0f)};
  case 9:
    return {sqrt8 * (9.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f), 6.0f * sqrt8 * x_n * y_n};
  case 10:
    return {3.0f * sqrt8 * (x_n * x_n - y_n * y_n), -6.0f * sqrt8 * x_n * y_n};
  default:
    throw std::invalid_argument("Unsupported Noll index");
  }
}

std::array<float, kMaxSupportedModes>
solve_linear_system(std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> matrix,
                    std::array<float, kMaxSupportedModes> rhs, size_t size) {
  std::array<float, kMaxSupportedModes> solution{};
  constexpr float                       singular_epsilon = 1e-12f;

  for (size_t column = 0; column < size; ++column) {
    size_t pivot      = column;
    float  best_value = std::abs(matrix[column][column]);
    for (size_t row = column + 1; row < size; ++row) {
      const float candidate = std::abs(matrix[row][column]);
      if (candidate > best_value) {
        best_value = candidate;
        pivot      = row;
      }
    }

    if (best_value < singular_epsilon) {
      return solution;
    }

    if (pivot != column) {
      std::swap(matrix[pivot], matrix[column]);
      std::swap(rhs[pivot], rhs[column]);
    }

    const float pivot_value = matrix[column][column];
    for (size_t j = column; j < size; ++j) {
      matrix[column][j] /= pivot_value;
    }
    rhs[column] /= pivot_value;

    for (size_t row = column + 1; row < size; ++row) {
      const float factor = matrix[row][column];
      for (size_t j = column; j < size; ++j) {
        matrix[row][j] -= factor * matrix[column][j];
      }
      rhs[row] -= factor * rhs[column];
    }
  }

  for (int row = static_cast<int>(size) - 1; row >= 0; --row) {
    float value = rhs[static_cast<size_t>(row)];
    for (size_t column = static_cast<size_t>(row) + 1; column < size; ++column) {
      value -= matrix[static_cast<size_t>(row)][column] * solution[column];
    }
    solution[static_cast<size_t>(row)] = value;
  }

  return solution;
}

float load_slope(const holoflow::core::TView &view, size_t sy, size_t sx, size_t component) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(view.storage->ptr + view.desc.offset);
  const auto  offset =
      sy * view.desc.strides[1] + sx * view.desc.strides[2] + component * view.desc.strides[3];
  return *reinterpret_cast<const float *>(bytes + offset);
}

detail::ShackHartmannGeometrySettings geometry_settings(const holoflow::core::TDesc     &input,
                                                        const ZernikeFromSlopesSettings &settings) {
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

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

struct GpuFitData {
  std::vector<size_t>                  active_samples;
  std::vector<float>                   derivatives_x;
  std::vector<float>                   derivatives_y;
  std::vector<float>                   regularized_gram;
  detail::ZernikeFromSlopesGpuSettings kernel_settings;
};

GpuFitData make_gpu_fit_data(const holoflow::core::TDesc     &input,
                             const ZernikeFromSlopesSettings &settings) {
  const auto geometry = detail::make_shack_hartmann_geometry(geometry_settings(input, settings));

  std::vector<size_t> observable_positions;
  observable_positions.reserve(settings.indexes.size());
  for (size_t position = 0; position < settings.indexes.size(); ++position) {
    if (settings.indexes[position] != 2 && settings.indexes[position] != 3) {
      observable_positions.push_back(position);
    }
  }

  GpuFitData data;
  data.active_samples.reserve(geometry.samples.size());
  for (size_t sample = 0; sample < geometry.samples.size(); ++sample) {
    if (geometry.samples[sample].active) {
      data.active_samples.push_back(sample);
    }
  }

  const size_t observable_count = observable_positions.size();
  const size_t active_count     = data.active_samples.size();
  data.derivatives_x.resize(active_count * observable_count);
  data.derivatives_y.resize(active_count * observable_count);
  std::array<float, kMaxSupportedModes> means_x{};
  std::array<float, kMaxSupportedModes> means_y{};

  for (size_t active = 0; active < active_count; ++active) {
    const auto &sample = geometry.samples[data.active_samples[active]];
    for (size_t mode = 0; mode < observable_count; ++mode) {
      const auto derivative = eval_zernike_noll_derivative(
          settings.indexes[observable_positions[mode]], sample.x_n, sample.y_n);
      const size_t offset        = active * observable_count + mode;
      data.derivatives_x[offset] = derivative.dx_n / geometry.pupil_radius_m;
      data.derivatives_y[offset] = derivative.dy_n / geometry.pupil_radius_m;
      means_x[mode] += data.derivatives_x[offset];
      means_y[mode] += data.derivatives_y[offset];
    }
  }

  for (size_t mode = 0; mode < observable_count; ++mode) {
    means_x[mode] /= static_cast<float>(active_count);
    means_y[mode] /= static_cast<float>(active_count);
  }
  for (size_t active = 0; active < active_count; ++active) {
    for (size_t mode = 0; mode < observable_count; ++mode) {
      const size_t offset = active * observable_count + mode;
      data.derivatives_x[offset] -= means_x[mode];
      data.derivatives_y[offset] -= means_y[mode];
    }
  }

  data.regularized_gram.assign(observable_count * observable_count, 0.0f);
  for (size_t active = 0; active < active_count; ++active) {
    const size_t derivative_offset = active * observable_count;
    for (size_t i = 0; i < observable_count; ++i) {
      for (size_t j = 0; j < observable_count; ++j) {
        data.regularized_gram[i * observable_count + j] +=
            data.derivatives_x[derivative_offset + i] * data.derivatives_x[derivative_offset + j] +
            data.derivatives_y[derivative_offset + i] * data.derivatives_y[derivative_offset + j];
      }
    }
  }
  constexpr float ridge = 1e-9f;
  for (size_t mode = 0; mode < observable_count; ++mode) {
    data.regularized_gram[mode * observable_count + mode] += ridge;
  }

  data.kernel_settings = {
      .active_count      = active_count,
      .observable_count  = observable_count,
      .output_count      = settings.indexes.size(),
      .sx_count          = input.shape[2],
      .stride_y          = input.strides[1],
      .stride_x          = input.strides[2],
      .stride_component  = input.strides[3],
      .radians_per_meter = 2.0f * std::acos(-1.0f) / settings.lambda,
  };
  for (size_t mode = 0; mode < observable_count; ++mode) {
    data.kernel_settings.observable_positions[mode] = static_cast<int>(observable_positions[mode]);
  }
  return data;
}

// -------------------------------------------------------------------------------------------------
// ZernikeFromSlopes task implementation
// -------------------------------------------------------------------------------------------------

class CpuZernikeFromSlopes : public holoflow::core::ISyncTask {
public:
  CpuZernikeFromSlopes(ZernikeFromSlopesSettings settings, holoflow::core::TDesc input_desc,
                       cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    const auto  &input    = ctx.inputs[0].desc;
    const size_t sy_count = input.shape[1];
    const size_t sx_count = input.shape[2];
    const auto geometry = detail::make_shack_hartmann_geometry(geometry_settings(input, settings_));

    std::vector<size_t> observable_positions;
    observable_positions.reserve(settings_.indexes.size());
    for (size_t position = 0; position < settings_.indexes.size(); ++position) {
      if (settings_.indexes[position] != 2 && settings_.indexes[position] != 3) {
        observable_positions.push_back(position);
      }
    }

    const size_t observable_count = observable_positions.size();
    std::vector<std::array<float, kMaxSupportedModes>> derivatives_x(geometry.samples.size());
    std::vector<std::array<float, kMaxSupportedModes>> derivatives_y(geometry.samples.size());
    std::array<float, kMaxSupportedModes>              means_x{};
    std::array<float, kMaxSupportedModes>              means_y{};
    size_t                                             active_count = 0;

    for (size_t sample_index = 0; sample_index < geometry.samples.size(); ++sample_index) {
      const auto &sample = geometry.samples[sample_index];
      if (!sample.active) {
        continue;
      }

      ++active_count;
      for (size_t mode = 0; mode < observable_count; ++mode) {
        const auto derivative = eval_zernike_noll_derivative(
            settings_.indexes[observable_positions[mode]], sample.x_n, sample.y_n);
        derivatives_x[sample_index][mode] = derivative.dx_n / geometry.pupil_radius_m;
        derivatives_y[sample_index][mode] = derivative.dy_n / geometry.pupil_radius_m;
        means_x[mode] += derivatives_x[sample_index][mode];
        means_y[mode] += derivatives_y[sample_index][mode];
      }
    }

    for (size_t mode = 0; mode < observable_count; ++mode) {
      means_x[mode] /= static_cast<float>(active_count);
      means_y[mode] /= static_cast<float>(active_count);
    }

    std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> gtg{};
    std::array<float, kMaxSupportedModes>                                 gts{};

    for (size_t sy = 0; sy < sy_count; ++sy) {
      for (size_t sx = 0; sx < sx_count; ++sx) {
        const size_t sample_index = sy * sx_count + sx;
        if (!geometry.samples[sample_index].active) {
          continue;
        }

        const float slope_x = load_slope(ctx.inputs[0], sy, sx, 0);
        const float slope_y = load_slope(ctx.inputs[0], sy, sx, 1);

        for (size_t i = 0; i < observable_count; ++i) {
          const float gx_i = derivatives_x[sample_index][i] - means_x[i];
          const float gy_i = derivatives_y[sample_index][i] - means_y[i];
          for (size_t j = 0; j < observable_count; ++j) {
            const float gx_j = derivatives_x[sample_index][j] - means_x[j];
            const float gy_j = derivatives_y[sample_index][j] - means_y[j];
            gtg[i][j] += gx_i * gx_j + gy_i * gy_j;
          }
          gts[i] += gx_i * slope_x + gy_i * slope_y;
        }
      }
    }

    constexpr float ridge = 1e-9f;
    for (size_t mode = 0; mode < observable_count; ++mode) {
      gtg[mode][mode] += ridge;
    }
    const auto coefficients_m = solve_linear_system(gtg, gts, observable_count);

    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
    std::fill_n(output, settings_.indexes.size(), 0.0f);
    const float radians_per_meter = 2.0f * std::acos(-1.0f) / settings_.lambda;
    for (size_t mode = 0; mode < observable_count; ++mode) {
      output[observable_positions[mode]] = coefficients_m[mode] * radians_per_meter;
    }

    return holoflow::core::OpResult::Ok;
  }

  void                             update_stream(cudaStream_t stream) { stream_ = stream; }
  const ZernikeFromSlopesSettings &settings() const { return settings_; }
  const holoflow::core::TDesc     &input_desc() const { return input_desc_; }

private:
  ZernikeFromSlopesSettings settings_;
  holoflow::core::TDesc     input_desc_;
  cudaStream_t              stream_;
};

class GpuZernikeFromSlopes : public holoflow::core::ISyncTask {
public:
  GpuZernikeFromSlopes(ZernikeFromSlopesSettings settings, holoflow::core::TDesc input_desc,
                       GpuFitData fit_data, cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)),
        kernel_settings_(fit_data.kernel_settings), stream_(stream),
        active_samples_(curaii::make_unique_device_ptr<size_t>(
            std::max<size_t>(1, fit_data.active_samples.size()), stream)),
        derivatives_x_(curaii::make_unique_device_ptr<float>(
            std::max<size_t>(1, fit_data.derivatives_x.size()), stream)),
        derivatives_y_(curaii::make_unique_device_ptr<float>(
            std::max<size_t>(1, fit_data.derivatives_y.size()), stream)),
        regularized_gram_(curaii::make_unique_device_ptr<float>(
            std::max<size_t>(1, fit_data.regularized_gram.size()), stream)) {
    if (!fit_data.active_samples.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(active_samples_.get(), fit_data.active_samples.data(),
                                 fit_data.active_samples.size() * sizeof(size_t),
                                 cudaMemcpyHostToDevice, stream_));
    }
    if (!fit_data.derivatives_x.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(derivatives_x_.get(), fit_data.derivatives_x.data(),
                                 fit_data.derivatives_x.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(derivatives_y_.get(), fit_data.derivatives_y.data(),
                                 fit_data.derivatives_y.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(regularized_gram_.get(), fit_data.regularized_gram.data(),
                                 fit_data.regularized_gram.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUDA_CHECK(detail::launch_zernike_from_slopes_gpu(
        reinterpret_cast<const float *>(ctx.inputs[0].data()),
        reinterpret_cast<float *>(ctx.outputs[0].data()), active_samples_.get(),
        derivatives_x_.get(), derivatives_y_.get(), regularized_gram_.get(), kernel_settings_,
        stream_));
    return holoflow::core::OpResult::Ok;
  }

  void                             update_stream(cudaStream_t stream) { stream_ = stream; }
  const ZernikeFromSlopesSettings &settings() const { return settings_; }
  const holoflow::core::TDesc     &input_desc() const { return input_desc_; }

private:
  ZernikeFromSlopesSettings            settings_;
  holoflow::core::TDesc                input_desc_;
  detail::ZernikeFromSlopesGpuSettings kernel_settings_;
  cudaStream_t                         stream_;
  DevPtr<size_t>                       active_samples_;
  DevPtr<float>                        derivatives_x_;
  DevPtr<float>                        derivatives_y_;
  DevPtr<float>                        regularized_gram_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// ZernikeFromSlopesFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ZernikeFromSlopesFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikeFromSlopesSettings>();

  check(input_descs.size() == 1, "task must have exactly one input");
  const auto &input = input_descs[0];
  check(input.mem_loc == holoflow::core::MemLoc::Host ||
            input.mem_loc == holoflow::core::MemLoc::Device,
        "input memory location must be Host or Device");
  check(input.dtype == holoflow::core::DType::F32, "input dtype must be F32");
  check(input.rank() == 4, "input rank must be 4");
  check(input.shape[0] == 1, "only batch size 1 is supported");
  check(input.shape[1] > 0 && input.shape[2] > 0, "subaperture grid dimensions must be positive");
  check(input.shape[3] == 2, "last input dimension must contain [dW/dx, dW/dy]");

  check(settings.lambda > 0.0f, "wavelength must be > 0");
  check(settings.dx > 0.0f && settings.dy > 0.0f, "pixel pitches must be > 0");
  check(settings.subaperture_height > 0 && settings.subaperture_width > 0,
        "subaperture dimensions must be positive");
  check(settings.stride_y > 0 && settings.stride_x > 0, "strides must be positive");
  check(settings.ny == 1 && settings.nx == 1,
        "only global fitting with ny = nx = 1 is supported for zero-mean slopes");

  check(!settings.indexes.empty(), "indexes must not be empty");
  check(settings.indexes.size() <= kMaxSupportedModes, "too many requested Zernike modes");
  for (int index : settings.indexes) {
    check(index >= 2 && index <= 10, "only Noll indexes 2..10 are supported");
  }
  auto unique_indexes = settings.indexes;
  std::sort(unique_indexes.begin(), unique_indexes.end());
  check(std::adjacent_find(unique_indexes.begin(), unique_indexes.end()) == unique_indexes.end(),
        "indexes must be unique");

  const auto geometry = detail::make_shack_hartmann_geometry(geometry_settings(input, settings));
  check(geometry.pupil_radius_m > 0.0f, "pupil radius must be positive");
  check(std::ranges::any_of(geometry.samples, &detail::ShackHartmannSample::active),
        "at least one subaperture must be active");

  holoflow::core::TDesc output({1, 1, settings.indexes.size()}, holoflow::core::DType::F32,
                               input.mem_loc);
  return {
      .input_descs   = {input},
      .output_descs  = {output},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeFromSlopesFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings,
                                 const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  auto settings = jsettings.get<ZernikeFromSlopesSettings>();
  auto input    = input_descs[0];
  if (input.mem_loc == holoflow::core::MemLoc::Device) {
    return std::make_unique<GpuZernikeFromSlopes>(settings, input,
                                                  make_gpu_fit_data(input, settings), ctx.stream);
  }
  return std::make_unique<CpuZernikeFromSlopes>(settings, input, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeFromSlopesFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                                 std::span<const holoflow::core::TDesc>     input_descs,
                                 const nlohmann::json                      &jsettings,
                                 const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  const auto  settings = jsettings.get<ZernikeFromSlopesSettings>();
  const auto &input    = input_descs[0];
  if (auto *old_cpu = dynamic_cast<CpuZernikeFromSlopes *>(old_task.get());
      old_cpu != nullptr && input.mem_loc == holoflow::core::MemLoc::Host &&
      settings == old_cpu->settings() && same_desc(input, old_cpu->input_desc())) {
    old_cpu->update_stream(ctx.stream);
    return old_task;
  }
  if (auto *old_gpu = dynamic_cast<GpuZernikeFromSlopes *>(old_task.get());
      old_gpu != nullptr && input.mem_loc == holoflow::core::MemLoc::Device &&
      settings == old_gpu->settings() && same_desc(input, old_gpu->input_desc())) {
    old_gpu->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
