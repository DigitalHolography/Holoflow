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

#include "holonp/remainder.hh"
#include "utils/tensor_common.hh"

#include <cmath>
#include <vector>

namespace holonp {

void to_json(nlohmann::json &j, const RemainderSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, RemainderSettings &) {}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

__global__ void remainder_kernel(const float *a, const float *b, float *out, size_t total,
                                 size_t ndim, const size_t *shape, const size_t *a_strides,
                                 const size_t *b_strides) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  size_t a_offset = 0, b_offset = 0, remaining = idx;
  for (size_t axis = ndim; axis-- > 0;) {
    const auto coordinate = remaining % shape[axis];
    remaining /= shape[axis];
    a_offset += coordinate * a_strides[axis];
    b_offset += coordinate * b_strides[axis];
  }

  const float divisor = b[b_offset];
  float value = fmodf(a[a_offset], divisor);
  if (value == 0.0f)
    value = copysignf(0.0f, divisor);
  else if (!isnan(value) && signbit(value) != signbit(divisor))
    value += divisor;
  out[idx] = value;
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Remainder: " + msg);
  }
}

class Remainder : public holoflow::core::ISyncTask {
public:
  Remainder(cudaStream_t stream, std::vector<holoflow::core::TDesc> idescs,
         size_t total_out, size_t ndim, DevPtr<size_t> d_out_shape,
         DevPtr<size_t> d_a_strides, DevPtr<size_t> d_b_strides)
      : stream_(stream), idescs_(std::move(idescs)), total_out_(total_out),
        ndim_(ndim), d_out_shape_(std::move(d_out_shape)), d_a_strides_(std::move(d_a_strides)),
        d_b_strides_(std::move(d_b_strides)) {}

  bool supports_cuda_graph() const noexcept override { return true; }
  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs, nullptr, nullptr, nullptr};
    (void)execute(execution);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const std::vector<holoflow::core::TDesc> &idescs() const { return idescs_; }
  void                                      update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  cudaStream_t                       stream_;
  std::vector<holoflow::core::TDesc> idescs_;
  size_t                             total_out_;
  size_t                             ndim_;
  DevPtr<size_t>                     d_out_shape_;
  DevPtr<size_t>                     d_a_strides_;
  DevPtr<size_t>                     d_b_strides_;
};

} // namespace

holoflow::core::OpResult Remainder::execute(holoflow::core::SyncCtx &ctx) {
  if (total_out_ != 0) {
    constexpr int block = 256;
    const int grid = static_cast<int>((total_out_ + block - 1) / block);
    remainder_kernel<<<grid, block, 0, stream_>>>(
        reinterpret_cast<const float *>(ctx.inputs[0].data()),
        reinterpret_cast<const float *>(ctx.inputs[1].data()),
        reinterpret_cast<float *>(ctx.outputs[0].data()), total_out_, ndim_,
        d_out_shape_.get(), d_a_strides_.get(), d_b_strides_.get());
    CUDA_CHECK(cudaGetLastError());
  }
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult RemainderFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                 const nlohmann::json &) const {
  check(inputs.size() == 2, "expected exactly 2 input tensors");
  check(inputs[0].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 0 must be in device memory");
  check(inputs[1].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 1 must be in device memory");

  const auto &a = inputs[0];
  const auto &b = inputs[1];
  check(a.dtype == holoflow::core::DType::F32 && b.dtype == holoflow::core::DType::F32,
        "inputs must be float32");

  size_t ndim = std::max(a.shape.size(), b.shape.size());

  std::vector<size_t> out_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    size_t ad    = (i < ndim - a.shape.size()) ? 1 : a.shape[i - (ndim - a.shape.size())];
    size_t bd    = (i < ndim - b.shape.size()) ? 1 : b.shape[i - (ndim - b.shape.size())];
    check(ad == bd || ad == 1 || bd == 1, "input shapes are not broadcastable");
    out_shape[i] = ad == 1 ? bd : ad;
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
RemainderFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &j,
                      const holoflow::core::SyncCreateCtx &ctx) const {

  const auto &a = inputs[0], &b = inputs[1];
  auto        res   = infer(inputs, j);
  const auto &odesc = res.output_descs[0];
  size_t      ndim  = odesc.shape.size();
  size_t      total = odesc.num_elements();

  std::vector<size_t> a_strides_h(ndim), b_strides_h(ndim);
  auto                as_raw = utils::get_elem_strides(a);
  auto                bs_raw = utils::get_elem_strides(b);

  for (size_t i = 0; i < ndim; ++i) {
    auto map = [&](const auto &shape, const auto &strides) {
      int axis = int(i) - (int(ndim) - int(shape.size()));
      return (axis < 0 || shape[axis] == 1) ? 0 : strides[axis];
    };
    a_strides_h[i] = map(a.shape, as_raw);
    b_strides_h[i] = map(b.shape, bs_raw);
  }

  auto d_shape = curaii::make_unique_device_ptr<size_t>(ndim, ctx.stream);
  auto d_a_str = curaii::make_unique_device_ptr<size_t>(ndim, ctx.stream);
  auto d_b_str = curaii::make_unique_device_ptr<size_t>(ndim, ctx.stream);
  auto bytes   = ndim * sizeof(size_t);

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), odesc.shape.data(), bytes, cudaMemcpyHostToDevice,
                             ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_a_str.get(), a_strides_h.data(), bytes, cudaMemcpyHostToDevice,
                             ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_b_str.get(), b_strides_h.data(), bytes, cudaMemcpyHostToDevice,
                             ctx.stream));

  return std::make_unique<Remainder>(
      ctx.stream, std::vector<holoflow::core::TDesc>{inputs[0], inputs[1]}, total,
      ndim, std::move(d_shape), std::move(d_a_str), std::move(d_b_str));
}

std::unique_ptr<holoflow::core::ISyncTask>
RemainderFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {

  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  auto *old_remainder = dynamic_cast<Remainder *>(old_task.get());
  if (old_remainder && input_descs.size() == 2) {
    const auto &old_idescs = old_remainder->idescs();
    if (utils::same_desc(input_descs[0], old_idescs[0]) &&
        utils::same_desc(input_descs[1], old_idescs[1])) {
      old_remainder->update_stream(ctx.stream);
      return old_task;
    }
  }
  ctx.invalidate_execution();
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
