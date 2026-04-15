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

#include "holonp/concatenate.hh"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ConcatenateSettings &s) {
  if (s.axis.has_value()) {
    j = nlohmann::json{{"axis", *s.axis}};
  } else {
    j = nlohmann::json{{"axis", nullptr}};
  }
}

void from_json(const nlohmann::json &j, ConcatenateSettings &s) {
  if (j.contains("axis")) {
    const auto &axis = j.at("axis");
    if (axis.is_null()) {
      s.axis = std::nullopt;
    } else if (axis.is_number_integer()) {
      s.axis = axis.get<int>();
    } else {
      throw std::invalid_argument("ConcatenateSettings: axis must be int or null");
    }
  } else {
    s.axis = 0;
  }
}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

constexpr std::int64_t kMaxI64 = std::numeric_limits<std::int64_t>::max();

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Concatenate: " + msg);
  }
}

inline std::int64_t to_i64(size_t v, const std::string &msg) {
  check(v <= static_cast<size_t>(kMaxI64), msg);
  return static_cast<std::int64_t>(v);
}

inline std::int64_t checked_add(std::int64_t a, std::int64_t b, const std::string &msg) {
  check(b <= 0 || a <= (kMaxI64 - b), msg);
  return a + b;
}

inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const std::string &msg) {
  if (a == 0 || b == 0) {
    return 0;
  }
  check(a <= (kMaxI64 / b), msg);
  return a * b;
}

inline int normalize_axis(int axis, int ndim) {
  if (axis < 0) {
    axis += ndim;
  }
  check(axis >= 0 && axis < ndim, "axis out of range");
  return axis;
}

struct ConcatPlan {
  std::vector<size_t>               out_shape;
  std::int64_t                      inner        = 1;
  std::int64_t                      out_axis_dim = 0;
  std::vector<ConcatenateInputPlan> inputs;
};

ConcatPlan build_plan(const ConcatenateSettings             &settings,
                      std::span<const holoflow::core::TDesc> input_descs) {
  check(!input_descs.empty(), "expected at least 1 input");

  const auto dtype   = input_descs[0].dtype;
  const auto mem_loc = input_descs[0].mem_loc;
  check(mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  (void)holoflow::core::size_of(dtype);

  for (const auto &desc : input_descs) {
    check(desc.mem_loc == mem_loc, "all inputs must share the same mem_loc");
    check(desc.dtype == dtype, "all inputs must share the same dtype");
    holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
    check(desc.strides == contiguous.strides, "all inputs must be contiguous");
  }

  ConcatPlan plan;

  if (!settings.axis.has_value()) {
    std::int64_t offset = 0;
    for (const auto &desc : input_descs) {
      const auto total_in = to_i64(desc.num_elements(), "input too large");
      plan.inputs.push_back(ConcatenateInputPlan{total_in, total_in, offset});
      offset = checked_add(offset, total_in, "output too large");
    }
    plan.inner        = 1;
    plan.out_axis_dim = offset;
    plan.out_shape    = {static_cast<size_t>(plan.out_axis_dim)};
    return plan;
  }

  const int ndim = static_cast<int>(input_descs[0].shape.size());
  check(ndim > 0, "input ndim must be > 0");
  for (const auto &desc : input_descs) {
    check(static_cast<int>(desc.shape.size()) == ndim, "all inputs must share the same rank");
  }

  const int axis = normalize_axis(*settings.axis, ndim);

  plan.out_shape = input_descs[0].shape;

  plan.inner = 1;
  for (int i = axis + 1; i < ndim; ++i) {
    plan.inner = checked_mul(
        plan.inner, to_i64(plan.out_shape[static_cast<size_t>(i)], "input shape too large"),
        "inner size too large");
  }

  std::int64_t axis_offset = 0;
  for (const auto &desc : input_descs) {
    for (int i = 0; i < ndim; ++i) {
      if (i == axis) {
        continue;
      }
      check(desc.shape[static_cast<size_t>(i)] == plan.out_shape[static_cast<size_t>(i)],
            "all input shapes must match except along axis");
    }

    const auto axis_dim = to_i64(desc.shape[static_cast<size_t>(axis)], "axis dimension too large");
    const auto total_in = to_i64(desc.num_elements(), "input too large");

    plan.inputs.push_back(ConcatenateInputPlan{total_in, axis_dim, axis_offset});
    axis_offset = checked_add(axis_offset, axis_dim, "output axis too large");
  }

  plan.out_axis_dim                         = axis_offset;
  plan.out_shape[static_cast<size_t>(axis)] = static_cast<size_t>(plan.out_axis_dim);

  return plan;
}

template <typename T>
__global__ void concat_kernel(const T *__restrict__ in, T *__restrict__ out, std::int64_t total_in,
                              std::int64_t inner, std::int64_t axis_dim, std::int64_t out_axis_dim,
                              std::int64_t axis_offset) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_in) {
    return;
  }

  const std::int64_t axis_block = axis_dim * inner;
  const std::int64_t outer_idx  = tid / axis_block;
  const std::int64_t rem        = tid - outer_idx * axis_block;
  const std::int64_t axis_idx   = rem / inner;
  const std::int64_t inner_idx  = rem - axis_idx * inner;

  const std::int64_t out_idx =
      (outer_idx * out_axis_dim + (axis_offset + axis_idx)) * inner + inner_idx;

  out[out_idx] = in[tid];
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

// -------------------------------------------------------------------------------------------------
// Concatenate task implementation
// -------------------------------------------------------------------------------------------------

class Concatenate : public holoflow::core::ISyncTask {
public:
  Concatenate(ConcatenateSettings settings, cudaStream_t stream, std::int64_t inner,
              std::int64_t out_axis_dim, std::vector<ConcatenateInputPlan> inputs,
              std::vector<holoflow::core::TDesc> input_descs)
      : settings_(std::move(settings)), stream_(stream), inner_(inner), out_axis_dim_(out_axis_dim),
        inputs_(std::move(inputs)), input_descs_(std::move(input_descs)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const ConcatenateSettings                &settings() const { return settings_; }
  const std::vector<holoflow::core::TDesc> &input_descs() const { return input_descs_; }
  void                                      update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ConcatenateSettings                settings_;
  cudaStream_t                       stream_;
  std::int64_t                       inner_;
  std::int64_t                       out_axis_dim_;
  std::vector<ConcatenateInputPlan>  inputs_;
  std::vector<holoflow::core::TDesc> input_descs_;
};

} // namespace

holoflow::core::OpResult Concatenate::execute(holoflow::core::SyncCtx &ctx) {
  if (ctx.outputs.size() != 1 || ctx.inputs.size() != inputs_.size()) {
    logger()->error("[Concatenate::execute] expected N inputs and 1 output");
    std::abort();
  }

  auto      *odata     = ctx.outputs[0].data();
  const auto odesc     = ctx.outputs[0].desc;
  const auto total_out = static_cast<std::int64_t>(odesc.num_elements());
  if (total_out <= 0) {
    return holoflow::core::OpResult::Ok;
  }

  constexpr int block_size = 256;
  const auto    dtype      = odesc.dtype;

  for (size_t i = 0; i < inputs_.size(); ++i) {
    const auto &plan     = inputs_[i];
    const auto  total_in = plan.total_in;
    if (total_in <= 0) {
      continue;
    }

    auto *idata = ctx.inputs[i].data();

    const int grid_size = static_cast<int>((total_in + block_size - 1) / block_size);

    switch (dtype) {
    case holoflow::core::DType::U8: {
      auto *out = reinterpret_cast<std::uint8_t *>(odata);
      auto *in  = reinterpret_cast<const std::uint8_t *>(idata);
      concat_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, total_in, inner_, plan.axis_dim,
                                                           out_axis_dim_, plan.axis_offset);
      break;
    }
    case holoflow::core::DType::U16: {
      auto *out = reinterpret_cast<std::uint16_t *>(odata);
      auto *in  = reinterpret_cast<const std::uint16_t *>(idata);
      concat_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, total_in, inner_, plan.axis_dim,
                                                           out_axis_dim_, plan.axis_offset);
      break;
    }
    case holoflow::core::DType::F32: {
      auto *out = reinterpret_cast<float *>(odata);
      auto *in  = reinterpret_cast<const float *>(idata);
      concat_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, total_in, inner_, plan.axis_dim,
                                                           out_axis_dim_, plan.axis_offset);
      break;
    }
    case holoflow::core::DType::CF32: {
      auto *out = reinterpret_cast<cuFloatComplex *>(odata);
      auto *in  = reinterpret_cast<const cuFloatComplex *>(idata);
      concat_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, total_in, inner_, plan.axis_dim,
                                                           out_axis_dim_, plan.axis_offset);
      break;
    }
    default:
      logger()->error("[Concatenate::execute] unsupported dtype");
      std::abort();
    }

    CUDA_CHECK(cudaGetLastError());
  }

  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// ConcatenateFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ConcatenateFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ConcatenateSettings>();
  const auto plan     = build_plan(settings, input_descs);

  auto                  oshape = plan.out_shape;
  holoflow::core::TDesc odesc(oshape, input_descs[0].dtype, input_descs[0].mem_loc);

  return holoflow::core::InferResult{
      .input_descs   = {input_descs.begin(), input_descs.end()},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = std::vector<bool>(input_descs.size(), false),
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ConcatenateFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings,
                           const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto settings = jsettings.get<ConcatenateSettings>();
  auto       plan     = build_plan(settings, input_descs);

  return std::make_unique<Concatenate>(
      settings, ctx.stream, plan.inner, plan.out_axis_dim, std::move(plan.inputs),
      std::vector<holoflow::core::TDesc>(input_descs.begin(), input_descs.end()));
}

std::unique_ptr<holoflow::core::ISyncTask>
ConcatenateFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                           std::span<const holoflow::core::TDesc>     input_descs,
                           const nlohmann::json                      &jsettings,
                           const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_concat = dynamic_cast<Concatenate *>(old_task.get());
  if (old_concat == nullptr || input_descs.size() != old_concat->input_descs().size()) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings    = jsettings.get<ConcatenateSettings>();
  bool       same_inputs = settings == old_concat->settings();
  for (size_t i = 0; same_inputs && i < input_descs.size(); ++i) {
    same_inputs = same_desc(input_descs[i], old_concat->input_descs()[i]);
  }

  if (same_inputs) {
    old_concat->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
