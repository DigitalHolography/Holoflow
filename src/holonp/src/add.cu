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

#include "holonp/add.hh"

#include <cuComplex.h>

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const AddSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, AddSettings &) {}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

template <typename T>
__global__ void add_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out,
                           size_t total_out, size_t ndim, const size_t *__restrict__ out_shape,
                           const size_t *__restrict__ a_strides,
                           const size_t *__restrict__ b_strides) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_out)
    return;

  size_t a_off = 0, b_off = 0, rem = idx;

  for (int i = int(ndim) - 1; i >= 0; --i) {
    size_t coord = rem % out_shape[i];
    rem /= out_shape[i];
    a_off += coord * a_strides[i];
    b_off += coord * b_strides[i];
  }

  if constexpr (std::is_same_v<T, cuFloatComplex>) {
    out[idx] = cuCaddf(a[a_off], b[b_off]);
  } else {
    out[idx] = a[a_off] + b[b_off];
  }
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Add: " + msg);
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
// Add task implementation
// -------------------------------------------------------------------------------------------------

class Add : public holoflow::core::ISyncTask {
public:
  Add(cudaStream_t stream, holoflow::core::DType dtype, size_t total_out, size_t ndim,
      DevPtr<size_t> d_out_shape, DevPtr<size_t> d_a_strides, DevPtr<size_t> d_b_strides,
      std::vector<holoflow::core::TDesc> input_descs)
      : stream_(stream), dtype_(dtype), total_out_(total_out), ndim_(ndim),
        d_out_shape_(std::move(d_out_shape)), d_a_strides_(std::move(d_a_strides)),
        d_b_strides_(std::move(d_b_strides)), input_descs_(std::move(input_descs)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const std::vector<holoflow::core::TDesc> &input_descs() const { return input_descs_; }
  void                                      update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  cudaStream_t          stream_;
  holoflow::core::DType dtype_;
  size_t                total_out_;
  size_t                ndim_;
  DevPtr<size_t>        d_out_shape_;
  DevPtr<size_t>        d_a_strides_;
  DevPtr<size_t>        d_b_strides_;
  std::vector<holoflow::core::TDesc> input_descs_;
};

} // namespace

holoflow::core::OpResult Add::execute(holoflow::core::SyncCtx &ctx) {
  if (total_out_ == 0)
    return holoflow::core::OpResult::Ok;

  int block = 256;
  int grid  = static_cast<int>((total_out_ + block - 1) / block);

#define LAUNCH_TYPE(T)                                                                             \
  add_kernel<T><<<grid, block, 0, stream_>>>(                                                      \
      (T *)ctx.inputs[0].data(), (T *)ctx.inputs[1].data(), (T *)ctx.outputs[0].data(),            \
      total_out_, ndim_, d_out_shape_.get(), d_a_strides_.get(), d_b_strides_.get())

  switch (dtype_) {
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
// AddFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult AddFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                              const nlohmann::json &) const {
  check(inputs.size() == 2, "expected exactly 2 input tensors");
  check(inputs[0].dtype == inputs[1].dtype, "input tensor data types must match");
  check(inputs[0].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 0 must be in device memory");
  check(inputs[1].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 1 must be in device memory");

  const auto &a = inputs[0], &b = inputs[1];
  size_t      ndim = std::max(a.shape.size(), b.shape.size());

  std::vector<size_t> out_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    size_t ad    = (i < ndim - a.shape.size()) ? 1 : a.shape[i - (ndim - a.shape.size())];
    size_t bd    = (i < ndim - b.shape.size()) ? 1 : b.shape[i - (ndim - b.shape.size())];
    out_shape[i] = std::max(ad, bd);
  }

  holoflow::core::TDesc o(out_shape, a.dtype, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {a, b},
      .output_descs  = {o},
      .in_place      = {},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AddFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &j,
                   const holoflow::core::SyncCreateCtx &ctx) const {
  const auto &a = inputs[0], &b = inputs[1];
  auto        res   = infer(inputs, j);
  const auto &odesc = res.output_descs[0];
  size_t      ndim  = odesc.shape.size();

  size_t              total = 1;
  std::vector<size_t> a_strides_h(ndim), b_strides_h(ndim);
  auto                as_raw = get_elem_strides(a);
  auto                bs_raw = get_elem_strides(b);

  for (size_t i = 0; i < ndim; ++i) {
    total *= odesc.shape[i];
    auto map = [&](const auto &shape, const auto &strides) {
      int axis = int(i) - (int(ndim) - int(shape.size()));
      return (axis < 0 || shape[axis] == 1) ? 0 : strides[axis];
    };
    a_strides_h[i] = map(a.shape, as_raw);
    b_strides_h[i] = map(b.shape, bs_raw);
  }

  auto d_shape = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_a_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto d_b_str = curaii::make_unique_device_ptr<size_t>(ndim);
  auto bytes   = ndim * sizeof(size_t);
  auto h2d     = cudaMemcpyHostToDevice;

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), odesc.shape.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_a_str.get(), a_strides_h.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_b_str.get(), b_strides_h.data(), bytes, h2d, ctx.stream));

  return std::make_unique<Add>(ctx.stream, inputs[0].dtype, total, ndim, std::move(d_shape),
                               std::move(d_a_str), std::move(d_b_str),
                               std::vector<holoflow::core::TDesc>(inputs.begin(), inputs.end()));
}

std::unique_ptr<holoflow::core::ISyncTask>
AddFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_add = dynamic_cast<Add *>(old_task.get());
  if (old_add == nullptr || input_descs.size() != 2 || old_add->input_descs().size() != 2) {
    return create(input_descs, jsettings, ctx);
  }

  if (same_desc(input_descs[0], old_add->input_descs()[0]) &&
      same_desc(input_descs[1], old_add->input_descs()[1])) {
    old_add->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
