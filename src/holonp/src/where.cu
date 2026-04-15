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

#include "holonp/where.hh"

#include <algorithm>
#include <cuComplex.h>

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const WhereSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, WhereSettings &) {}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

template <typename T>
__global__ void
where_kernel(const uint8_t *__restrict__ cond, const T *__restrict__ x, const T *__restrict__ y,
             T *__restrict__ out, size_t total_out, size_t ndim,
             const size_t *__restrict__ out_shape, const size_t *__restrict__ cond_strides,
             const size_t *__restrict__ x_strides, const size_t *__restrict__ y_strides) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_out)
    return;

  size_t cond_off = 0, x_off = 0, y_off = 0, rem = idx;

  for (int i = int(ndim) - 1; i >= 0; --i) {
    size_t coord = rem % out_shape[i];
    rem /= out_shape[i];
    cond_off += coord * cond_strides[i];
    x_off += coord * x_strides[i];
    y_off += coord * y_strides[i];
  }

  const bool take_x = cond[cond_off] != 0;
  out[idx]          = take_x ? x[x_off] : y[y_off];
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Where: " + msg);
  }
}

std::vector<size_t> get_elem_strides(const holoflow::core::TDesc &d) {
  size_t esize = holoflow::core::size_of(d.dtype);
  if (!d.strides.empty()) {
    std::vector<size_t> s;
    for (auto val : d.strides)
      s.push_back(val / esize);
    return s;
  }
  std::vector<size_t> s(d.shape.size());
  size_t              acc = 1;
  for (int i = int(d.shape.size()) - 1; i >= 0; --i) {
    s[i] = acc;
    acc *= d.shape[i];
  }
  return s;
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

// -------------------------------------------------------------------------------------------------
// Where task implementation
// -------------------------------------------------------------------------------------------------

class Where : public holoflow::core::ISyncTask {
public:
  Where(cudaStream_t stream, holoflow::core::DType out_dtype, size_t total_out, size_t ndim,
        DevPtr<size_t> d_out_shape, DevPtr<size_t> d_cond_strides, DevPtr<size_t> d_x_strides,
        DevPtr<size_t> d_y_strides, std::vector<holoflow::core::TDesc> input_descs)
      : stream_(stream), out_dtype_(out_dtype), total_out_(total_out), ndim_(ndim),
        d_out_shape_(std::move(d_out_shape)), d_cond_strides_(std::move(d_cond_strides)),
        d_x_strides_(std::move(d_x_strides)), d_y_strides_(std::move(d_y_strides)),
        input_descs_(std::move(input_descs)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const std::vector<holoflow::core::TDesc> &input_descs() const { return input_descs_; }
  void                                      update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  cudaStream_t                       stream_;
  holoflow::core::DType              out_dtype_;
  size_t                             total_out_;
  size_t                             ndim_;
  DevPtr<size_t>                     d_out_shape_;
  DevPtr<size_t>                     d_cond_strides_;
  DevPtr<size_t>                     d_x_strides_;
  DevPtr<size_t>                     d_y_strides_;
  std::vector<holoflow::core::TDesc> input_descs_;
};

} // namespace

holoflow::core::OpResult Where::execute(holoflow::core::SyncCtx &ctx) {
  if (total_out_ == 0)
    return holoflow::core::OpResult::Ok;

  int block = 256;
  int grid  = static_cast<int>((total_out_ + block - 1) / block);

#define LAUNCH_TYPE(T)                                                                             \
  where_kernel<T><<<grid, block, 0, stream_>>>(                                                    \
      (uint8_t *)ctx.inputs[0].data(), (T *)ctx.inputs[1].data(), (T *)ctx.inputs[2].data(),       \
      (T *)ctx.outputs[0].data(), total_out_, ndim_, d_out_shape_.get(), d_cond_strides_.get(),    \
      d_x_strides_.get(), d_y_strides_.get())

  switch (out_dtype_) {
  case holoflow::core::DType::F32:
    LAUNCH_TYPE(float);
    break;
  case holoflow::core::DType::CF32:
    LAUNCH_TYPE(cuFloatComplex);
    break;
  case holoflow::core::DType::U16:
    LAUNCH_TYPE(uint16_t);
    break;
  case holoflow::core::DType::U8:
    LAUNCH_TYPE(uint8_t);
    break;
  default:
    std::abort();
  }
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// WhereFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult WhereFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                const nlohmann::json &) const {
  check(inputs.size() == 3, "expected exactly 3 input tensors");
  check(inputs[0].mem_loc == holoflow::core::MemLoc::Device,
        "condition tensor must be in device memory");
  check(inputs[1].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 1 must be in device memory");
  check(inputs[2].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 2 must be in device memory");
  check(inputs[0].dtype == holoflow::core::DType::U8, "condition tensor must be U8");
  check(inputs[1].dtype == inputs[2].dtype, "input tensor data types must match");

  const auto &cond = inputs[0];
  const auto &x    = inputs[1];
  const auto &y    = inputs[2];

  size_t ndim = std::max({cond.shape.size(), x.shape.size(), y.shape.size()});

  std::vector<size_t> out_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    size_t cd    = (i < ndim - cond.shape.size()) ? 1 : cond.shape[i - (ndim - cond.shape.size())];
    size_t xd    = (i < ndim - x.shape.size()) ? 1 : x.shape[i - (ndim - x.shape.size())];
    size_t yd    = (i < ndim - y.shape.size()) ? 1 : y.shape[i - (ndim - y.shape.size())];
    out_shape[i] = std::max({cd, xd, yd});
  }

  holoflow::core::TDesc o(out_shape, x.dtype, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {cond, x, y},
      .output_descs  = {o},
      .in_place      = {},
      .owned_inputs  = {false, false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
WhereFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &j,
                     const holoflow::core::SyncCreateCtx &ctx) const {
  const auto &cond  = inputs[0];
  const auto &x     = inputs[1];
  const auto &y     = inputs[2];
  auto        res   = infer(inputs, j);
  const auto &odesc = res.output_descs[0];
  size_t      ndim  = odesc.shape.size();
  size_t      total = 1;

  std::vector<size_t> cond_strides_h(ndim), x_strides_h(ndim), y_strides_h(ndim);
  auto                cs_raw = get_elem_strides(cond);
  auto                xs_raw = get_elem_strides(x);
  auto                ys_raw = get_elem_strides(y);

  for (size_t i = 0; i < ndim; ++i) {
    total *= odesc.shape[i];
    auto map = [&](const auto &shape, const auto &strides) {
      int axis = int(i) - (int(ndim) - int(shape.size()));
      return (axis < 0 || shape[axis] == 1) ? 0 : strides[axis];
    };
    cond_strides_h[i] = map(cond.shape, cs_raw);
    x_strides_h[i]    = map(x.shape, xs_raw);
    y_strides_h[i]    = map(y.shape, ys_raw);
  }

  auto d_shape = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_c_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_x_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_y_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto bytes   = ndim * sizeof(size_t);
  auto h2d     = cudaMemcpyHostToDevice;

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), odesc.shape.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_c_str.get(), cond_strides_h.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_x_str.get(), x_strides_h.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_y_str.get(), y_strides_h.data(), bytes, h2d, ctx.stream));

  return std::make_unique<Where>(ctx.stream, odesc.dtype, total, ndim, std::move(d_shape),
                                 std::move(d_c_str), std::move(d_x_str), std::move(d_y_str),
                                 std::vector<holoflow::core::TDesc>(inputs.begin(), inputs.end()));
}

std::unique_ptr<holoflow::core::ISyncTask>
WhereFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                     std::span<const holoflow::core::TDesc>     input_descs,
                     const nlohmann::json                      &jsettings,
                     const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_where = dynamic_cast<Where *>(old_task.get());
  if (old_where == nullptr || input_descs.size() != 3 || old_where->input_descs().size() != 3) {
    return create(input_descs, jsettings, ctx);
  }

  if (same_desc(input_descs[0], old_where->input_descs()[0]) &&
      same_desc(input_descs[1], old_where->input_descs()[1]) &&
      same_desc(input_descs[2], old_where->input_descs()[2])) {
    old_where->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
