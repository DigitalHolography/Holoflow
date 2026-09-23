// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/gradient.hh"
#include "utils/tensor_common.hh"

#include <cuComplex.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const GradientSettings &s) {
  j = {{"spacing", s.spacing}, {"axis", s.axis}, {"edge_order", s.edge_order}};
}

void from_json(const nlohmann::json &j, GradientSettings &s) {
  s.spacing.clear();
  if (j.contains("spacing") && !j.at("spacing").is_null()) {
    if (j.at("spacing").is_array())
      s.spacing = j.at("spacing").get<std::vector<float>>();
    else
      s.spacing = {j.at("spacing").get<float>()};
  }

  s.axis.clear();
  if (j.contains("axis") && !j.at("axis").is_null()) {
    if (j.at("axis").is_array())
      s.axis = j.at("axis").get<std::vector<int>>();
    else
      s.axis = {j.at("axis").get<int>()};
  }
  s.edge_order = j.value("edge_order", 1);
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Gradient: " + msg);
}

std::vector<int> normalize_axes(std::span<const int> axes, int rank) {
  std::vector<int> result;
  if (axes.empty()) {
    result.resize(static_cast<size_t>(rank));
    for (int axis = 0; axis < rank; ++axis)
      result[static_cast<size_t>(axis)] = axis;
    return result;
  }

  result.reserve(axes.size());
  for (int axis : axes) {
    if (axis < 0)
      axis += rank;
    check(axis >= 0 && axis < rank, "axis out of range");
    check(std::find(result.begin(), result.end(), axis) == result.end(), "axes must be unique");
    result.push_back(axis);
  }
  return result;
}

template <typename T>
__device__ T load_gradient_value(const T *input, std::int64_t base, std::int64_t offset) {
  return input[base + offset];
}

template <typename T>
__global__ void gradient_kernel(const T *__restrict__ input, T *__restrict__ output,
                                std::int64_t output_elements, int rank, int axis,
                                std::int64_t axis_size, const size_t *__restrict__ input_strides,
                                const size_t *__restrict__ output_strides, float spacing,
                                int edge_order) {
  const auto output_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_elements)
    return;

  auto         coordinate = output_index;
  std::int64_t input_base = 0;
  for (int dim = 0; dim < rank; ++dim) {
    const auto value = coordinate / output_strides[dim];
    coordinate -= value * output_strides[dim];
    input_base += value * input_strides[dim];
  }

  const auto axis_coordinate = (output_index / output_strides[axis]) % axis_size;
  input_base -= axis_coordinate * input_strides[axis];
  const auto stride = static_cast<std::int64_t>(input_strides[axis]);

  T result{};
  if (edge_order == 2 && axis_coordinate == 0) {
    result = (-1.5f * load_gradient_value(input, input_base, 0) +
              2.0f * load_gradient_value(input, input_base, stride) -
              0.5f * load_gradient_value(input, input_base, 2 * stride)) /
             spacing;
  } else if (edge_order == 2 && axis_coordinate == axis_size - 1) {
    result = (0.5f * load_gradient_value(input, input_base, (axis_size - 3) * stride) -
              2.0f * load_gradient_value(input, input_base, (axis_size - 2) * stride) +
              1.5f * load_gradient_value(input, input_base, (axis_size - 1) * stride)) /
             spacing;
  } else if (axis_coordinate == 0) {
    result = (load_gradient_value(input, input_base, stride) -
              load_gradient_value(input, input_base, 0)) /
             spacing;
  } else if (axis_coordinate == axis_size - 1) {
    result = (load_gradient_value(input, input_base, (axis_size - 1) * stride) -
              load_gradient_value(input, input_base, (axis_size - 2) * stride)) /
             spacing;
  } else {
    result = (load_gradient_value(input, input_base, (axis_coordinate + 1) * stride) -
              load_gradient_value(input, input_base, (axis_coordinate - 1) * stride)) /
             (2.0f * spacing);
  }
  output[output_index] = result;
}

template <>
__global__ void gradient_kernel<cuFloatComplex>(const cuFloatComplex *__restrict__ input,
                                                cuFloatComplex *__restrict__ output,
                                                std::int64_t output_elements, int rank, int axis,
                                                std::int64_t axis_size,
                                                const size_t *__restrict__ input_strides,
                                                const size_t *__restrict__ output_strides,
                                                float spacing, int edge_order) {
  const auto output_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_elements)
    return;

  auto         coordinate = output_index;
  std::int64_t input_base = 0;
  for (int dim = 0; dim < rank; ++dim) {
    const auto value = coordinate / output_strides[dim];
    coordinate -= value * output_strides[dim];
    input_base += value * input_strides[dim];
  }
  const auto axis_coordinate = (output_index / output_strides[axis]) % axis_size;
  input_base -= axis_coordinate * input_strides[axis];
  const auto stride = static_cast<std::int64_t>(input_strides[axis]);

  const auto scaled = [spacing](cuFloatComplex value) {
    return make_cuFloatComplex(value.x / spacing, value.y / spacing);
  };
  const auto add_scaled = [](cuFloatComplex a, cuFloatComplex b, float factor) {
    return make_cuFloatComplex(a.x + factor * b.x, a.y + factor * b.y);
  };

  cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);
  if (edge_order == 2 && axis_coordinate == 0) {
    result = add_scaled(result, input[input_base], -1.5f);
    result = add_scaled(result, input[input_base + stride], 2.0f);
    result = add_scaled(result, input[input_base + 2 * stride], -0.5f);
  } else if (edge_order == 2 && axis_coordinate == axis_size - 1) {
    result = add_scaled(result, input[input_base + (axis_size - 3) * stride], 0.5f);
    result = add_scaled(result, input[input_base + (axis_size - 2) * stride], -2.0f);
    result = add_scaled(result, input[input_base + (axis_size - 1) * stride], 1.5f);
  } else if (axis_coordinate == 0) {
    result = add_scaled(result, input[input_base + stride], 1.0f);
    result = add_scaled(result, input[input_base], -1.0f);
  } else if (axis_coordinate == axis_size - 1) {
    result = add_scaled(result, input[input_base + (axis_size - 1) * stride], 1.0f);
    result = add_scaled(result, input[input_base + (axis_size - 2) * stride], -1.0f);
  } else {
    result = add_scaled(result, input[input_base + (axis_coordinate + 1) * stride], 0.5f);
    result = add_scaled(result, input[input_base + (axis_coordinate - 1) * stride], -0.5f);
  }
  output[output_index] = scaled(result);
}

class Gradient : public holoflow::core::ISyncTask {
public:
  Gradient(GradientSettings settings, holoflow::core::TDesc idesc,
           std::vector<holoflow::core::TDesc> odescs, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), odescs_(std::move(odescs)),
        axes_(normalize_axes(settings_.axis, static_cast<int>(idesc_.shape.size()))),
        stream_(stream) {
    if (settings_.spacing.empty())
      spacings_.assign(axes_.size(), 1.0f);
    else if (settings_.spacing.size() == 1)
      spacings_.assign(axes_.size(), settings_.spacing[0]);
    else
      spacings_ = settings_.spacing;
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto    input_strides   = upload(utils::get_elem_strides(idesc_));
    const auto    output_strides  = upload(utils::compact_strides(idesc_.shape));
    const auto    output_elements = static_cast<std::int64_t>(idesc_.num_elements());
    constexpr int block           = 256;
    const int     grid            = static_cast<int>((output_elements + block - 1) / block);

    for (size_t output_index = 0; output_index < axes_.size(); ++output_index) {
      const int axis = axes_[output_index];
      if (idesc_.dtype == holoflow::core::DType::F32) {
        gradient_kernel<<<grid, block, 0, stream_>>>(
            reinterpret_cast<const float *>(ctx.inputs[0].data()),
            reinterpret_cast<float *>(ctx.outputs[output_index].data()), output_elements,
            static_cast<int>(idesc_.shape.size()), axis,
            static_cast<std::int64_t>(idesc_.shape[static_cast<size_t>(axis)]), input_strides.get(),
            output_strides.get(), spacings_[output_index], settings_.edge_order);
      } else {
        gradient_kernel<<<grid, block, 0, stream_>>>(
            reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data()),
            reinterpret_cast<cuFloatComplex *>(ctx.outputs[output_index].data()), output_elements,
            static_cast<int>(idesc_.shape.size()), axis,
            static_cast<std::int64_t>(idesc_.shape[static_cast<size_t>(axis)]), input_strides.get(),
            output_strides.get(), spacings_[output_index], settings_.edge_order);
      }
      CUDA_CHECK(cudaGetLastError());
    }
    return holoflow::core::OpResult::Ok;
  }

  const GradientSettings      &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  template <typename T> auto upload(const std::vector<T> &values) {
    auto result = curaii::make_unique_device_ptr<T>(values.size());
    if (!values.empty())
      CUDA_CHECK(cudaMemcpyAsync(result.get(), values.data(), values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_));
    return result;
  }

  GradientSettings                   settings_;
  holoflow::core::TDesc              idesc_;
  std::vector<holoflow::core::TDesc> odescs_;
  std::vector<int>                   axes_;
  std::vector<float>                 spacings_;
  cudaStream_t                       stream_;
};

} // namespace

holoflow::core::InferResult
GradientFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "supported dtypes are F32 and CF32");
  check(!idesc.shape.empty(), "input must have rank >= 1");
  check(idesc.num_elements() > 0, "input tensor has zero elements");

  const auto settings = jsettings.get<GradientSettings>();
  check(settings.edge_order == 1 || settings.edge_order == 2, "edge_order must be 1 or 2");
  const auto axes = normalize_axes(settings.axis, static_cast<int>(idesc.shape.size()));
  check(!axes.empty(), "at least one axis is required");
  check(settings.spacing.empty() || settings.spacing.size() == 1 ||
            settings.spacing.size() == axes.size(),
        "spacing must be empty, scalar, or match the number of axes");
  for (const float spacing : settings.spacing)
    check(std::isfinite(spacing) && spacing != 0.0f, "spacing must be finite and nonzero");
  for (const int axis : axes)
    check(idesc.shape[static_cast<size_t>(axis)] >= static_cast<size_t>(settings.edge_order + 1),
          "selected axis is too short for edge_order");

  std::vector<holoflow::core::TDesc> output_descs;
  output_descs.reserve(axes.size());
  for (size_t i = 0; i < axes.size(); ++i)
    output_descs.emplace_back(idesc.shape, idesc.dtype, holoflow::core::MemLoc::Device);

  return {.input_descs   = {idesc},
          .output_descs  = std::move(output_descs),
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = std::vector<bool>(axes.size(), false),
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
GradientFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto inferred = infer(input_descs, jsettings);
  return std::make_unique<Gradient>(jsettings.get<GradientSettings>(), input_descs[0],
                                    inferred.output_descs, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
GradientFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old_gradient = dynamic_cast<Gradient *>(old_task.get());
  const auto settings     = jsettings.get<GradientSettings>();
  if (old_gradient != nullptr && old_gradient->settings() == settings && input_descs.size() == 1 &&
      utils::same_desc(input_descs[0], old_gradient->idesc())) {
    old_gradient->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
