// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace holoflow::python {

inline size_t checked_product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::overflow_error("Array collection size overflow");
  return a * b;
}

class ArrayCollector final : public core::ISyncTask {
public:
  ArrayCollector(core::TDesc input, const nlohmann::json &settings, cudaStream_t stream)
      : input_(std::move(input)), axis_(settings.at("axis").get<size_t>()),
        length_(settings.at("length").get<size_t>()), direct_(settings.value("direct", false)),
        stream_(stream) {
    auto shape   = input_.shape;
    shape[axis_] = length_;
    result_      = core::TDesc(shape, input_.dtype, core::MemLoc::Host);
    size_t bytes = core::TDesc({1}, input_.dtype, core::MemLoc::Host).num_bytes();
    for (auto dim : shape)
      bytes = checked_product(bytes, dim);
    buffer_ = std::make_shared<std::vector<std::byte>>(std::max(size_t{1}, bytes));
    scratch_.resize(input_.num_bytes());
  }

  std::optional<core::TView> acquire_input(int index) override {
    if (!direct_ || index != 0)
      throw std::out_of_range("ArrayCollector input");
    auto &storage = storage_access().owned_input_storage(0);
    // A final acquisition may precede discovering upstream EOF. Never write beyond the result.
    storage.ptr = received_ + input_.shape[axis_] <= length_
                      ? buffer_->data() + received_ * result_.strides[axis_]
                      : scratch_.data();
    return core::TView{input_, &storage};
  }

  core::OpResult execute(core::SyncCtx &ctx) override {
    const auto count = input_.shape[axis_];
    if (count > length_ - received_)
      throw std::runtime_error("NumPy collector length overflow");
    if (!direct_) {
      auto             *src  = ctx.inputs[0].data();
      auto             *dst  = buffer_->data();
      const size_t      item = core::TDesc({1}, input_.dtype, core::MemLoc::Host).num_bytes();
      const core::TDesc packed(input_.shape, input_.dtype, input_.mem_loc);
      const auto        copy = [&](std::byte *out, const std::byte *in, size_t bytes) {
        if (input_.mem_loc == core::MemLoc::Host)
          std::memcpy(out, in, bytes);
        else
          CUDA_CHECK(cudaMemcpyAsync(out, in, bytes, cudaMemcpyDeviceToHost, stream_));
      };
      if (input_.strides == packed.strides) {
        size_t outer = 1, inner = item;
        for (size_t d = 0; d < axis_; ++d)
          outer *= input_.shape[d];
        for (size_t d = axis_ + 1; d < input_.rank(); ++d)
          inner *= input_.shape[d];
        for (size_t i = 0; i < outer; ++i)
          copy(dst + (i * length_ + received_) * inner, src + i * count * inner, count * inner);
      } else {
        for (size_t i = 0; i < input_.num_elements(); ++i) {
          size_t rest = i, in_offset = 0, out_offset = received_ * result_.strides[axis_];
          for (size_t d = input_.rank(); d-- > 0;) {
            const auto coordinate = rest % input_.shape[d];
            rest /= input_.shape[d];
            in_offset += coordinate * input_.strides[d];
            out_offset += coordinate * result_.strides[d];
          }
          copy(dst + out_offset, src + in_offset, item);
        }
      }
    }
    received_ += count;
    return core::OpResult::Ok;
  }

  void end_of_stream(core::SyncCtx &) override {
    if (received_ != length_)
      throw std::runtime_error("NumPy collector length underflow at EOF");
    ready_ = true;
  }

  pybind11::array array() const {
    if (!ready_)
      throw std::runtime_error("Array collector has not received EOF");
    namespace py = pybind11;
    py::dtype dtype;
    switch (input_.dtype) {
    case core::DType::U8:
      dtype = py::dtype("uint8");
      break;
    case core::DType::U16:
      dtype = py::dtype("uint16");
      break;
    case core::DType::F32:
      dtype = py::dtype("float32");
      break;
    case core::DType::CF32:
      dtype = py::dtype("complex64");
      break;
    default:
      throw std::runtime_error("Unsupported NumPy dtype");
    }
    auto       *owner = new std::shared_ptr<std::vector<std::byte>>(buffer_);
    py::capsule capsule(
        owner, [](void *p) { delete static_cast<std::shared_ptr<std::vector<std::byte>> *>(p); });
    return py::array(dtype, result_.shape, result_.strides, buffer_->data(), capsule);
  }

private:
  core::TDesc                             input_, result_;
  size_t                                  axis_, length_, received_ = 0;
  bool                                    direct_, ready_ = false;
  cudaStream_t                            stream_;
  std::shared_ptr<std::vector<std::byte>> buffer_;
  std::vector<std::byte>                  scratch_;
};

class ArrayCollectorFactory final : public core::ISyncTaskFactory {
public:
  core::InferResult infer(std::span<const core::TDesc> inputs,
                          const nlohmann::json        &s) const override {
    if (inputs.size() != 1 || inputs[0].rank() == 0)
      throw std::invalid_argument("NumPy collector requires one nonscalar tensor");
    const auto axis = s.at("axis").get<int>();
    if (axis < 0 || axis >= static_cast<int>(inputs[0].rank()))
      throw std::invalid_argument("Invalid concatenation axis");
    if (s.contains("length") && !s["length"].is_null() &&
        (!s["length"].is_number_unsigned() &&
         (!s["length"].is_number_integer() || s["length"].get<int64_t>() < 0)))
      throw std::invalid_argument("length must be a nonnegative integer");
    const bool direct = s.value("direct", false);
    return {.input_descs   = {inputs[0]},
            .output_descs  = {},
            .in_place      = {},
            .owned_inputs  = {direct},
            .owned_outputs = {},
            .kind          = core::TaskKind::Sync};
  }
  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc> inputs,
                                          const nlohmann::json        &s,
                                          const core::SyncCreateCtx   &ctx) const override {
    (void)infer(inputs, s);
    return std::make_unique<ArrayCollector>(inputs[0], s, ctx.stream);
  }
};

} // namespace holoflow::python
