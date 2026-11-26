// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/convolve.hh"
#include "holonp/correlate.hh"
#include "utils/tensor_common.hh"

#include <cuComplex.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const ConvolveSettings &s) { j = {{"mode", s.mode}}; }
void from_json(const nlohmann::json &j, ConvolveSettings &s) { s.mode = j.value("mode", "full"); }
void to_json(nlohmann::json &j, const CorrelateSettings &s) { j = {{"mode", s.mode}}; }
void from_json(const nlohmann::json &j, CorrelateSettings &s) { s.mode = j.value("mode", "full"); }

namespace {

enum class Operation { Convolve, Correlate };

inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("Convolution: " + message);
}

struct Layout {
  size_t output_length;
  size_t start;
};

Layout make_layout(size_t a_length, size_t b_length, const std::string &mode) {
  check(mode == "full" || mode == "same" || mode == "valid",
        "mode must be 'full', 'same', or 'valid'");
  const size_t full_length = a_length + b_length - 1;
  if (mode == "full")
    return {.output_length = full_length, .start = 0};
  if (mode == "valid")
    return {.output_length = std::max(a_length, b_length) - std::min(a_length, b_length) + 1,
            .start         = std::min(a_length, b_length) - 1};

  const size_t output_length = std::max(a_length, b_length);
  const size_t start         = (full_length - output_length) / 2;
  return {.output_length = output_length, .start = start};
}

template <typename T>
__global__ void convolution_kernel(const T *__restrict__ a, const T *__restrict__ b,
                                   T *__restrict__ output, std::int64_t output_length,
                                   std::int64_t a_length, std::int64_t b_length, std::int64_t start,
                                   size_t a_stride, size_t b_stride, bool correlate) {
  const auto output_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_length)
    return;

  const auto full_index = output_index + start;
  T          result{};
  if (!correlate) {
    for (std::int64_t b_index = 0; b_index < b_length; ++b_index) {
      const auto a_index = full_index - b_index;
      if (a_index >= 0 && a_index < a_length)
        result += a[a_index * a_stride] * b[b_index * b_stride];
    }
  } else {
    const auto lag = full_index - (b_length - 1);
    for (std::int64_t b_index = 0; b_index < b_length; ++b_index) {
      const auto a_index = lag + b_index;
      if (a_index >= 0 && a_index < a_length)
        result += a[a_index * a_stride] * b[b_index * b_stride];
    }
  }
  output[output_index] = result;
}

template <>
__global__ void convolution_kernel<cuFloatComplex>(
    const cuFloatComplex *__restrict__ a, const cuFloatComplex *__restrict__ b,
    cuFloatComplex *__restrict__ output, std::int64_t output_length, std::int64_t a_length,
    std::int64_t b_length, std::int64_t start, size_t a_stride, size_t b_stride, bool correlate) {
  const auto output_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_length)
    return;

  const auto     full_index = output_index + start;
  cuFloatComplex result     = make_cuFloatComplex(0.0f, 0.0f);
  if (!correlate) {
    for (std::int64_t b_index = 0; b_index < b_length; ++b_index) {
      const auto a_index = full_index - b_index;
      if (a_index >= 0 && a_index < a_length) {
        const auto value = cuCmulf(a[a_index * a_stride], b[b_index * b_stride]);
        result           = cuCaddf(result, value);
      }
    }
  } else {
    const auto lag = full_index - (b_length - 1);
    for (std::int64_t b_index = 0; b_index < b_length; ++b_index) {
      const auto a_index = lag + b_index;
      if (a_index >= 0 && a_index < a_length) {
        const auto value = cuCmulf(a[a_index * a_stride], cuConjf(b[b_index * b_stride]));
        result           = cuCaddf(result, value);
      }
    }
  }
  output[output_index] = result;
}

class ConvolutionTask : public holoflow::core::ISyncTask {
public:
  ConvolutionTask(Operation operation, std::string mode, holoflow::core::TDesc a_desc,
                  holoflow::core::TDesc b_desc, cudaStream_t stream)
      : operation_(operation), mode_(std::move(mode)), a_desc_(std::move(a_desc)),
        b_desc_(std::move(b_desc)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto    a_strides = utils::get_elem_strides(a_desc_);
    const auto    b_strides = utils::get_elem_strides(b_desc_);
    const auto    layout    = make_layout(a_desc_.shape[0], b_desc_.shape[0], mode_);
    constexpr int block     = 256;
    const int     grid      = static_cast<int>((layout.output_length + block - 1) / block);
    const bool    correlate = operation_ == Operation::Correlate;

    if (a_desc_.dtype == holoflow::core::DType::F32) {
      convolution_kernel<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const float *>(ctx.inputs[0].data()),
          reinterpret_cast<const float *>(ctx.inputs[1].data()),
          reinterpret_cast<float *>(ctx.outputs[0].data()),
          static_cast<std::int64_t>(layout.output_length),
          static_cast<std::int64_t>(a_desc_.shape[0]), static_cast<std::int64_t>(b_desc_.shape[0]),
          static_cast<std::int64_t>(layout.start), a_strides[0], b_strides[0], correlate);
    } else {
      convolution_kernel<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data()),
          reinterpret_cast<const cuFloatComplex *>(ctx.inputs[1].data()),
          reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data()),
          static_cast<std::int64_t>(layout.output_length),
          static_cast<std::int64_t>(a_desc_.shape[0]), static_cast<std::int64_t>(b_desc_.shape[0]),
          static_cast<std::int64_t>(layout.start), a_strides[0], b_strides[0], correlate);
    }
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  Operation                    operation() const { return operation_; }
  const std::string           &mode() const { return mode_; }
  const holoflow::core::TDesc &a_desc() const { return a_desc_; }
  const holoflow::core::TDesc &b_desc() const { return b_desc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  Operation             operation_;
  std::string           mode_;
  holoflow::core::TDesc a_desc_;
  holoflow::core::TDesc b_desc_;
  cudaStream_t          stream_;
};

holoflow::core::InferResult infer_common(std::span<const holoflow::core::TDesc> input_descs,
                                         const nlohmann::json &jsettings, Operation operation,
                                         const std::string &operation_name) {
  check(input_descs.size() == 2, "expected exactly 2 inputs");
  const auto &a = input_descs[0];
  const auto &b = input_descs[1];
  check(a.mem_loc == holoflow::core::MemLoc::Device && b.mem_loc == holoflow::core::MemLoc::Device,
        "only Device tensors are supported");
  check(a.dtype == b.dtype, "inputs must have the same dtype");
  check(a.dtype == holoflow::core::DType::F32 || a.dtype == holoflow::core::DType::CF32,
        "supported dtypes are F32 and CF32");
  check(a.shape.size() == 1 && b.shape.size() == 1, "inputs must be 1-D");
  check(a.shape[0] > 0 && b.shape[0] > 0, "inputs must be nonempty");
  const auto settings = operation == Operation::Convolve ? jsettings.get<ConvolveSettings>().mode
                                                         : jsettings.get<CorrelateSettings>().mode;
  const auto layout   = make_layout(a.shape[0], b.shape[0], settings);
  (void)operation_name;
  return {.input_descs   = {a, b},
          .output_descs  = {holoflow::core::TDesc({layout.output_length}, a.dtype,
                                                  holoflow::core::MemLoc::Device)},
          .in_place      = {},
          .owned_inputs  = {false, false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

template <Operation operation, typename Settings, typename Factory>
std::unique_ptr<holoflow::core::ISyncTask>
create_task(const Factory &factory, std::span<const holoflow::core::TDesc> input_descs,
            const nlohmann::json &jsettings, const holoflow::core::SyncCreateCtx &ctx) {
  (void)factory;
  const auto inferred = infer_common(input_descs, jsettings, operation, "Convolution");
  return std::make_unique<ConvolutionTask>(operation, jsettings.get<Settings>().mode,
                                           input_descs[0], input_descs[1], ctx.stream);
}

template <Operation operation, typename Settings>
std::unique_ptr<holoflow::core::ISyncTask>
update_task(std::unique_ptr<holoflow::core::ISyncTask> old_task,
            std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
            const holoflow::core::SyncCreateCtx &ctx) {
  const auto settings = jsettings.get<Settings>();
  auto      *old      = dynamic_cast<ConvolutionTask *>(old_task.get());
  if (old != nullptr && old->operation() == operation && old->mode() == settings.mode &&
      input_descs.size() == 2 && utils::same_desc(input_descs[0], old->a_desc()) &&
      utils::same_desc(input_descs[1], old->b_desc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  if constexpr (operation == Operation::Convolve) {
    return create_task<operation, ConvolveSettings>(ConvolveFactory{}, input_descs, jsettings, ctx);
  } else {
    return create_task<operation, CorrelateSettings>(CorrelateFactory{}, input_descs, jsettings,
                                                     ctx);
  }
}

} // namespace

holoflow::core::InferResult ConvolveFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                   const nlohmann::json &settings) const {
  return infer_common(inputs, settings, Operation::Convolve, "Convolve");
}

std::unique_ptr<holoflow::core::ISyncTask>
ConvolveFactory::create(std::span<const holoflow::core::TDesc> inputs,
                        const nlohmann::json                  &settings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  return create_task<Operation::Convolve, ConvolveSettings>(*this, inputs, settings, ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
ConvolveFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     inputs,
                        const nlohmann::json                      &settings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(inputs, settings);
  return update_task<Operation::Convolve, ConvolveSettings>(std::move(old_task), inputs, settings,
                                                            ctx);
}

holoflow::core::InferResult CorrelateFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                    const nlohmann::json &settings) const {
  return infer_common(inputs, settings, Operation::Correlate, "Correlate");
}

std::unique_ptr<holoflow::core::ISyncTask>
CorrelateFactory::create(std::span<const holoflow::core::TDesc> inputs,
                         const nlohmann::json                  &settings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  return create_task<Operation::Correlate, CorrelateSettings>(*this, inputs, settings, ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
CorrelateFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                         std::span<const holoflow::core::TDesc>     inputs,
                         const nlohmann::json                      &settings,
                         const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(inputs, settings);
  return update_task<Operation::Correlate, CorrelateSettings>(std::move(old_task), inputs, settings,
                                                              ctx);
}

} // namespace holonp
