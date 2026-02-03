// Copyright 2025 Digital Holography Foundation
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

#include "holonp/assign.hh"

namespace holonp {

namespace {

__global__ void assign_nd_kernel(const std::byte *__restrict__ src, std::byte *__restrict__ dst,
                                 const int64_t *__restrict__ src_strides,
                                 const int64_t *__restrict__ dst_strides,
                                 const int64_t *__restrict__ shape, int ndim, int elem_size,
                                 int64_t total_elems) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elems)
    return;

  int64_t rem     = tid;
  int64_t src_off = 0;
  int64_t dst_off = 0;

  // Compute multi-index from linear index and map to respective byte offsets
  for (int i = ndim - 1; i >= 0; --i) {
    int64_t idx = rem % shape[i];
    rem /= shape[i];
    src_off += idx * src_strides[i];
    dst_off += idx * dst_strides[i];
  }

  const std::byte *s_ptr = src + src_off;
  std::byte       *d_ptr = dst + dst_off;

  // Byte-wise copy for arbitrary element sizes
  for (int b = 0; b < elem_size; ++b) {
    d_ptr[b] = s_ptr[b];
  }
}

} // namespace

void to_json(nlohmann::json &j, const AssignSettings &s) {
  (void)s;
  j = nlohmann::json::object();
}

void from_json(const nlohmann::json &j, AssignSettings &s) {
  (void)j;
  (void)s;
}

Assign::Assign(size_t ndim, size_t total_elems, size_t elem_size,
               curaii::unique_device_ptr<int64_t> d_src_strides,
               curaii::unique_device_ptr<int64_t> d_dst_strides,
               curaii::unique_device_ptr<int64_t> d_shape, cudaStream_t stream)
    : ndim_(ndim), total_elems_(total_elems), elem_size_(elem_size), stream_(stream),
      d_src_strides_(std::move(d_src_strides)), d_dst_strides_(std::move(d_dst_strides)),
      d_shape_(std::move(d_shape)) {}

holoflow::core::InferResult AssignFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json & /*jsettings*/) const {
  if (input_descs.size() != 2)
    throw std::runtime_error("Assign requires 2 inputs");

  const auto &src = input_descs[0];
  const auto &dst = input_descs[1];

  // Enforce No-Broadcasting: Shapes must be identical
  if (src.shape != dst.shape) {
    throw std::invalid_argument("Shapes must match exactly for strided assignment");
  }

  if (src.dtype != dst.dtype) {
    throw std::invalid_argument("DTypes must match");
  }

  return holoflow::core::InferResult{
      .input_descs   = {src, dst},
      .output_descs  = {dst},
      .in_place      = {{1, 0}},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AssignFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json & /*jsettings*/,
                      const holoflow::core::SyncCreateCtx &ctx) const {
  const auto  &src  = input_descs[0];
  const auto  &dst  = input_descs[1];
  const size_t ndim = src.shape.size();

  std::vector<int64_t> h_src_strides(ndim);
  std::vector<int64_t> h_dst_strides(ndim);
  std::vector<int64_t> h_shape(ndim);

  for (size_t i = 0; i < ndim; ++i) {
    h_src_strides[i] = static_cast<int64_t>(src.strides[i]);
    h_dst_strides[i] = static_cast<int64_t>(dst.strides[i]);
    h_shape[i]       = static_cast<int64_t>(src.shape[i]);
  }

  auto d_src_strides = curaii::make_unique_device_ptr<int64_t>(ndim);
  auto d_dst_strides = curaii::make_unique_device_ptr<int64_t>(ndim);
  auto d_shape       = curaii::make_unique_device_ptr<int64_t>(ndim);

  size_t bytes = ndim * sizeof(int64_t);
  CUDA_CHECK(cudaMemcpyAsync(d_src_strides.get(), h_src_strides.data(), bytes,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_dst_strides.get(), h_dst_strides.data(), bytes,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(
      cudaMemcpyAsync(d_shape.get(), h_shape.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));

  return std::make_unique<Assign>(ndim, src.num_elements(), holoflow::core::size_of(src.dtype),
                                  std::move(d_src_strides), std::move(d_dst_strides),
                                  std::move(d_shape), ctx.stream);
}

holoflow::core::OpResult Assign::execute(holoflow::core::SyncCtx &ctx) {
  if (total_elems_ == 0)
    return holoflow::core::OpResult::Ok;

  // TView::data() already accounts for TDesc::offset
  const std::byte *src_ptr = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  std::byte       *dst_ptr = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block_size = 256;
  int           grid_size  = static_cast<int>((total_elems_ + block_size - 1) / block_size);

  assign_nd_kernel<<<grid_size, block_size, 0, stream_>>>(
      src_ptr, dst_ptr, d_src_strides_.get(), d_dst_strides_.get(), d_shape_.get(),
      static_cast<int>(ndim_), static_cast<int>(elem_size_), static_cast<int64_t>(total_elems_));

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp