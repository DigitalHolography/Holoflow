// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holonp/assign.hh"

namespace holonp {

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

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

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

// -------------------------------------------------------------------------------------------------
// Assign task implementation
// -------------------------------------------------------------------------------------------------

class Assign : public holoflow::core::ISyncTask {
public:
  Assign(size_t ndim, size_t total_elems, size_t elem_size,
         curaii::unique_device_ptr<int64_t> d_src_strides,
         curaii::unique_device_ptr<int64_t> d_dst_strides,
         curaii::unique_device_ptr<int64_t> d_shape, cudaStream_t stream,
         std::vector<holoflow::core::TDesc> input_descs)
      : ndim_(ndim), total_elems_(total_elems), elem_size_(elem_size), stream_(stream),
        d_src_strides_(std::move(d_src_strides)), d_dst_strides_(std::move(d_dst_strides)),
        d_shape_(std::move(d_shape)), input_descs_(std::move(input_descs)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const std::vector<holoflow::core::TDesc> &input_descs() const { return input_descs_; }
  void                                      update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  size_t                             ndim_;
  size_t                             total_elems_;
  size_t                             elem_size_;
  cudaStream_t                       stream_;
  curaii::unique_device_ptr<int64_t> d_src_strides_;
  curaii::unique_device_ptr<int64_t> d_dst_strides_;
  curaii::unique_device_ptr<int64_t> d_shape_;
  std::vector<holoflow::core::TDesc> input_descs_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const AssignSettings &s) {
  (void)s;
  j = nlohmann::json::object();
}

void from_json(const nlohmann::json &j, AssignSettings &s) {
  (void)j;
  (void)s;
}

holoflow::core::InferResult AssignFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json & /*jsettings*/) const {
  if (input_descs.size() != 2)
    throw std::runtime_error("Assign requires 2 inputs");

  // Changed: dst is now input[0], src is input[1]
  const auto &dst = input_descs[0];
  const auto &src = input_descs[1];

  // Enforce No-Broadcasting: Shapes must be identical
  if (src.shape != dst.shape) {
    throw std::invalid_argument("Shapes must match exactly for strided assignment");
  }

  if (src.dtype != dst.dtype) {
    throw std::invalid_argument("DTypes must match");
  }

  return holoflow::core::InferResult{
      .input_descs   = {dst, src},
      .output_descs  = {dst},
      .in_place      = {{0, 0}},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AssignFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json & /*jsettings*/,
                      const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(input_descs, nlohmann::json::object());

  const auto  &dst  = input_descs[0];
  const auto  &src  = input_descs[1];
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

  return std::make_unique<Assign>(
      ndim, src.num_elements(), holoflow::core::size_of(src.dtype), std::move(d_src_strides),
      std::move(d_dst_strides), std::move(d_shape), ctx.stream,
      std::vector<holoflow::core::TDesc>(input_descs.begin(), input_descs.end()));
}

std::unique_ptr<holoflow::core::ISyncTask>
AssignFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_assign = dynamic_cast<Assign *>(old_task.get());
  if (old_assign == nullptr || input_descs.size() != 2 || old_assign->input_descs().size() != 2) {
    return create(input_descs, jsettings, ctx);
  }

  if (same_desc(input_descs[0], old_assign->input_descs()[0]) &&
      same_desc(input_descs[1], old_assign->input_descs()[1])) {
    old_assign->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

holoflow::core::OpResult Assign::execute(holoflow::core::SyncCtx &ctx) {
  if (total_elems_ == 0)
    return holoflow::core::OpResult::Ok;

  const std::byte *src_ptr = reinterpret_cast<const std::byte *>(ctx.inputs[1].data());
  std::byte       *dst_ptr = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block_size = 256;
  int           grid_size  = static_cast<int>((total_elems_ + block_size - 1) / block_size);

  assign_nd_kernel<<<grid_size, block_size, 0, stream_>>>(
      src_ptr, dst_ptr, d_src_strides_.get(), d_dst_strides_.get(), d_shape_.get(),
      static_cast<int>(ndim_), static_cast<int>(elem_size_), static_cast<int64_t>(total_elems_));

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp
