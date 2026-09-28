// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/diff.hh"
#include "utils/tensor_common.hh"

#include <cuComplex.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const DiffSettings &s) { j = {{"n", s.n}, {"axis", s.axis}}; }

void from_json(const nlohmann::json &j, DiffSettings &s) {
  s.n    = j.value("n", 1);
  s.axis = j.value("axis", -1);
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Diff: " + msg);
}

__device__ inline std::uint64_t binomial(int n, int k) {
  std::uint64_t result = 1;
  for (int i = 1; i <= k; ++i)
    result = result * static_cast<std::uint64_t>(n - k + i) / static_cast<std::uint64_t>(i);
  return result;
}

template <typename T>
__global__ void diff_kernel(const T *__restrict__ input, T *__restrict__ output,
                            std::int64_t output_elements, int rank, int axis, int n,
                            std::int64_t output_axis_size, const size_t *__restrict__ input_strides,
                            const size_t *__restrict__ output_strides) {
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

  const auto axis_coordinate = (output_index / output_strides[axis]) % output_axis_size;
  input_base -= axis_coordinate * input_strides[axis];

  T result{};
  for (int k = 0; k <= n; ++k) {
    const auto coefficient = static_cast<float>(binomial(n, k));
    const auto offset      = input_base + (axis_coordinate + k) * input_strides[axis];
    const auto value       = input[offset];
    if ((n - k) & 1)
      result -= value * coefficient;
    else
      result += value * coefficient;
  }
  output[output_index] = result;
}

template <>
__global__ void diff_kernel<cuFloatComplex>(const cuFloatComplex *__restrict__ input,
                                            cuFloatComplex *__restrict__ output,
                                            std::int64_t output_elements, int rank, int axis, int n,
                                            std::int64_t output_axis_size,
                                            const size_t *__restrict__ input_strides,
                                            const size_t *__restrict__ output_strides) {
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
  const auto axis_coordinate = (output_index / output_strides[axis]) % output_axis_size;
  input_base -= axis_coordinate * input_strides[axis];

  cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);
  for (int k = 0; k <= n; ++k) {
    const float coefficient = static_cast<float>(binomial(n, k));
    auto        value       = input[input_base + (axis_coordinate + k) * input_strides[axis]];
    value.x *= coefficient;
    value.y *= coefficient;
    if ((n - k) & 1) {
      result.x -= value.x;
      result.y -= value.y;
    } else {
      result.x += value.x;
      result.y += value.y;
    }
  }
  output[output_index] = result;
}

class Diff : public holoflow::core::ISyncTask {
public:
  Diff(DiffSettings settings, holoflow::core::TDesc idesc, holoflow::core::TDesc odesc,
       cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), odesc_(std::move(odesc)),
        stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto    input_strides   = upload(utils::get_elem_strides(idesc_));
    const auto    output_strides  = upload(utils::compact_strides(odesc_.shape));
    const auto    output_elements = static_cast<std::int64_t>(odesc_.num_elements());
    constexpr int block           = 256;
    const int     grid            = static_cast<int>((output_elements + block - 1) / block);
    const int  axis = utils::normalize_axis(settings_.axis, static_cast<int>(idesc_.shape.size()));
    const auto output_axis_size = static_cast<std::int64_t>(odesc_.shape[axis]);

    if (idesc_.dtype == holoflow::core::DType::F32) {
      diff_kernel<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const float *>(ctx.inputs[0].data()),
          reinterpret_cast<float *>(ctx.outputs[0].data()), output_elements,
          static_cast<int>(idesc_.shape.size()), axis, settings_.n, output_axis_size,
          input_strides.get(), output_strides.get());
    } else {
      diff_kernel<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data()),
          reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data()), output_elements,
          static_cast<int>(idesc_.shape.size()), axis, settings_.n, output_axis_size,
          input_strides.get(), output_strides.get());
    }
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const DiffSettings          &settings() const { return settings_; }
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

  DiffSettings          settings_;
  holoflow::core::TDesc idesc_;
  holoflow::core::TDesc odesc_;
  cudaStream_t          stream_;
};

} // namespace

holoflow::core::InferResult DiffFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "supported dtypes are F32 and CF32");
  check(idesc.shape.size() > 0, "input must have rank >= 1");
  check(idesc.num_elements() > 0, "input tensor has zero elements");
  const auto settings = jsettings.get<DiffSettings>();
  check(settings.n >= 0, "n must be nonnegative");
  check(settings.n <= 32, "n is too large");
  const int axis = utils::normalize_axis(settings.axis, static_cast<int>(idesc.shape.size()));
  check(axis >= 0 && axis < static_cast<int>(idesc.shape.size()), "axis out of range");
  check(static_cast<size_t>(settings.n) < idesc.shape[static_cast<size_t>(axis)],
        "n must be smaller than the selected axis length");

  auto output_shape = idesc.shape;
  output_shape[static_cast<size_t>(axis)] -= static_cast<size_t>(settings.n);
  const holoflow::core::TDesc odesc(output_shape, idesc.dtype, holoflow::core::MemLoc::Device);
  return {.input_descs   = {idesc},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
DiffFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto inferred = infer(input_descs, jsettings);
  return std::make_unique<Diff>(jsettings.get<DiffSettings>(), input_descs[0],
                                inferred.output_descs[0], ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
DiffFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old_diff = dynamic_cast<Diff *>(old_task.get());
  const auto settings = jsettings.get<DiffSettings>();
  if (old_diff != nullptr && input_descs.size() == 1 && old_diff->settings() == settings &&
      utils::same_desc(input_descs[0], old_diff->idesc())) {
    old_diff->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
