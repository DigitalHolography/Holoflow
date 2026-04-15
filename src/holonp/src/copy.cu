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

#include "holonp/copy.hh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace holonp {

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Copy: " + msg);
  }
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

__global__ void copy_kernel(const std::byte *__restrict__ src, std::byte *__restrict__ dst,
                            const std::int64_t *__restrict__ src_strides,
                            const std::int64_t *__restrict__ src_shape, int ndim, int elem_size,
                            std::int64_t total_elems) {
  const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                           static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_elems) {
    return;
  }

  std::int64_t rem     = tid;
  std::int64_t src_off = 0;

  for (int i = ndim - 1; i >= 0; --i) {
    const std::int64_t idx = rem % src_shape[i];
    rem /= src_shape[i];
    src_off += idx * src_strides[i];
  }

  const std::byte *src_ptr = src + src_off;
  std::byte       *dst_ptr = dst + (tid * elem_size);
  for (int b = 0; b < elem_size; ++b) {
    dst_ptr[b] = src_ptr[b];
  }
}

// -------------------------------------------------------------------------------------------------
// Task implementation
// -------------------------------------------------------------------------------------------------

class Copy : public holoflow::core::ISyncTask {
public:
  Copy(size_t ndim, size_t total_elems, size_t elem_size,
       curaii::unique_device_ptr<std::int64_t> d_src_strides,
       curaii::unique_device_ptr<std::int64_t> d_src_shape, cudaStream_t stream,
       holoflow::core::TDesc idesc)
      : ndim_(ndim), total_elems_(total_elems), elem_size_(elem_size), stream_(stream),
        d_src_strides_(std::move(d_src_strides)), d_src_shape_(std::move(d_src_shape)),
        idesc_(std::move(idesc)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  size_t                       ndim_        = 0;
  size_t                       total_elems_ = 0;
  size_t                       elem_size_   = 0;
  cudaStream_t                 stream_      = static_cast<cudaStream_t>(0);
  curaii::unique_device_ptr<std::int64_t> d_src_strides_;
  curaii::unique_device_ptr<std::int64_t> d_src_shape_;
  holoflow::core::TDesc        idesc_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const CopySettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, CopySettings &) {}

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult CopyFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json                  &jsettings) const {
  (void)jsettings.get<CopySettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  holoflow::core::TDesc odesc(idesc.shape, idesc.dtype, idesc.mem_loc);

  return {
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CopyFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto &idesc = input_descs[0];

  const size_t              ndim = idesc.shape.size();
  std::vector<std::int64_t> h_strides(ndim);
  std::vector<std::int64_t> h_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_strides[i] = static_cast<std::int64_t>(idesc.strides[i]);
    h_shape[i]   = static_cast<std::int64_t>(idesc.shape[i]);
  }

  auto d_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_shape   = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_strides.get(), h_strides.data(), ndim * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), h_shape.data(), ndim * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, ctx.stream));

  return std::make_unique<Copy>(ndim, idesc.num_elements(), holoflow::core::size_of(idesc.dtype),
                                std::move(d_strides), std::move(d_shape), ctx.stream, idesc);
}

std::unique_ptr<holoflow::core::ISyncTask>
CopyFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_copy = dynamic_cast<Copy *>(old_task.get());
  if (old_copy == nullptr || input_descs.size() != 1) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &idesc = input_descs[0];
  if (same_desc(idesc, old_copy->idesc())) {
    old_copy->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

holoflow::core::OpResult Copy::execute(holoflow::core::SyncCtx &ctx) {
  if (total_elems_ == 0) {
    return holoflow::core::OpResult::Ok;
  }

  const std::byte *src = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  std::byte       *dst = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block = 256;
  const int     grid =
      static_cast<int>((static_cast<std::int64_t>(total_elems_) + block - 1) / block);

  copy_kernel<<<grid, block, 0, stream_>>>(src, dst, d_src_strides_.get(), d_src_shape_.get(),
                                           static_cast<int>(ndim_), static_cast<int>(elem_size_),
                                           static_cast<std::int64_t>(total_elems_));
  CUDA_CHECK(cudaGetLastError());

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp
