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
//
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holonp/slice_copy.hh"

namespace holonp {

struct SliceAssignSettings {
  // Must have length == write tensor ndim.
  std::vector<SliceItem> slices;
};

void to_json(nlohmann::json &j, const SliceAssignSettings &s) {
  j = nlohmann::json{{"slices", s.slices}};
}

void from_json(const nlohmann::json &j, SliceAssignSettings &s) { j.at("slices").get_to(s.slices); }

namespace {

constexpr int kMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("SliceAssign: " + msg);
  }
}

inline size_t product_shape(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
}

struct NormalizedSlice {
  std::int64_t start;
  std::int64_t stop; // exclusive
  std::int64_t step; // > 0
};

// NumPy-like normalization for basic positive-step slicing.
// Supports negative start/stop by wrapping with dim.
// Defaults: start=0, stop=dim, step=1.
// Clamps to [0, dim].
inline NormalizedSlice normalize_slice(const SliceItem &s, std::int64_t dim) {
  check(dim >= 0, "invalid dimension");

  const auto step = s.step;
  check(step > 0, "only positive step is supported for now");

  std::int64_t start = s.start.value_or(0);
  std::int64_t stop  = s.stop.value_or(dim);

  if (start < 0)
    start += dim;
  if (stop < 0)
    stop += dim;

  start = std::clamp<std::int64_t>(start, 0, dim);
  stop  = std::clamp<std::int64_t>(stop, 0, dim);

  return NormalizedSlice{start, stop, step};
}

inline std::int64_t out_len(const NormalizedSlice &ns) {
  if (ns.start >= ns.stop) {
    return 0;
  }
  const auto span = ns.stop - ns.start;
  return (span + ns.step - 1) / ns.step; // ceil_div
}

inline std::vector<NormalizedSlice>
normalize_and_validate_slices(std::span<const SliceItem> slices,
                              std::span<const size_t>    dst_shape) {
  const int ndim = static_cast<int>(dst_shape.size());
  check(static_cast<int>(slices.size()) == ndim, "slices length must match write ndim");
  check(ndim > 0, "write ndim must be > 0");
  check(ndim <= kMaxNDim, "write ndim too large");

  std::vector<NormalizedSlice> out;
  out.reserve(static_cast<size_t>(ndim));
  for (int i = 0; i < ndim; ++i) {
    out.push_back(normalize_slice(slices[static_cast<size_t>(i)],
                                  static_cast<std::int64_t>(dst_shape[static_cast<size_t>(i)])));
  }
  return out;
}

inline std::vector<std::int64_t> make_contig_strides(std::span<const size_t> shape) {
  const int                 ndim = static_cast<int>(shape.size());
  std::vector<std::int64_t> strides(static_cast<size_t>(ndim), 1);
  std::int64_t              acc = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = acc;
    acc *= static_cast<std::int64_t>(shape[static_cast<size_t>(i)]);
  }
  return strides;
}

// Data-moving basic slice assign for contiguous output.
// Writes the source tensor into the destination slice in row-major order (linear tid).
__global__ void
slice_assign_nd_contig_kernel_u8(const std::uint8_t *__restrict__ src,
                                 std::uint8_t *__restrict__ dst, int elem_size, int ndim,
                                 const std::int64_t *__restrict__ dst_strides, // [ndim] (elements)
                                 const std::int64_t *__restrict__ start,       // [ndim]
                                 const std::int64_t *__restrict__ step,        // [ndim]
                                 const std::int64_t *__restrict__ slice_shape, // [ndim]
                                 std::int64_t total_slice_elems) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_slice_elems) {
    return;
  }

  // Decode tid (slice linear) -> slice multi-index, then map to destination linear offset.
  std::int64_t rem        = tid;
  std::int64_t dst_off_el = 0;

  for (int k = 0; k < ndim; ++k) {
    std::int64_t out_suffix = 1;
    for (int j = k + 1; j < ndim; ++j) {
      out_suffix *= slice_shape[j];
    }

    const std::int64_t out_idx_k = (out_suffix > 0) ? (rem / out_suffix) : 0;
    rem -= out_idx_k * out_suffix;

    const std::int64_t dst_idx_k = start[k] + out_idx_k * step[k];
    dst_off_el += dst_idx_k * dst_strides[k];
  }

  const auto *src_ptr = src + static_cast<size_t>(tid) * static_cast<size_t>(elem_size);
  auto       *dst_ptr = dst + static_cast<size_t>(dst_off_el) * static_cast<size_t>(elem_size);

  for (int b = 0; b < elem_size; ++b) {
    dst_ptr[b] = src_ptr[b];
  }
}

} // namespace

class SliceAssign : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  SliceAssign(const SliceAssignSettings &settings, cudaStream_t stream, size_t ndim,
              size_t total_out, HostPtr<std::int64_t> h_dst_strides,
              DevPtr<std::int64_t> d_dst_strides, HostPtr<std::int64_t> h_start,
              DevPtr<std::int64_t> d_start, HostPtr<std::int64_t> h_step,
              DevPtr<std::int64_t> d_step, HostPtr<std::int64_t> h_slice_shape,
              DevPtr<std::int64_t> d_slice_shape);

  friend class SliceAssignFactory;

  SliceAssignSettings settings_;
  cudaStream_t        stream_;

  size_t ndim_;
  size_t total_out_;

  HostPtr<std::int64_t> h_dst_strides_;
  DevPtr<std::int64_t>  d_dst_strides_;

  HostPtr<std::int64_t> h_start_;
  DevPtr<std::int64_t>  d_start_;

  HostPtr<std::int64_t> h_step_;
  DevPtr<std::int64_t>  d_step_;

  HostPtr<std::int64_t> h_slice_shape_;
  DevPtr<std::int64_t>  d_slice_shape_;
};

class SliceAssignFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

// --------------------- impl ---------------------

SliceAssign::SliceAssign(const SliceAssignSettings &settings, cudaStream_t stream, size_t ndim,
                         size_t total_out, HostPtr<std::int64_t> h_dst_strides,
                         DevPtr<std::int64_t> d_dst_strides, HostPtr<std::int64_t> h_start,
                         DevPtr<std::int64_t> d_start, HostPtr<std::int64_t> h_step,
                         DevPtr<std::int64_t> d_step, HostPtr<std::int64_t> h_slice_shape,
                         DevPtr<std::int64_t> d_slice_shape)
    : settings_(settings), stream_(stream), ndim_(ndim), total_out_(total_out),
      h_dst_strides_(std::move(h_dst_strides)), d_dst_strides_(std::move(d_dst_strides)),
      h_start_(std::move(h_start)), d_start_(std::move(d_start)), h_step_(std::move(h_step)),
      d_step_(std::move(d_step)), h_slice_shape_(std::move(h_slice_shape)),
      d_slice_shape_(std::move(d_slice_shape)) {}

holoflow::core::OpResult SliceAssign::execute(holoflow::core::SyncCtx &ctx) {
  auto *src_data = ctx.inputs[0].data();
  auto *dst_data = ctx.outputs[0].data();
  const auto &src_desc = ctx.inputs[0].desc;

  constexpr int block_size = 256;
  const auto    total_i64  = static_cast<std::int64_t>(total_out_);
  const int     grid_size  = static_cast<int>((total_i64 + block_size - 1) / block_size);

  const int esz = static_cast<int>(size_of(src_desc.dtype));

  slice_assign_nd_contig_kernel_u8<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<const std::uint8_t *>(src_data), reinterpret_cast<std::uint8_t *>(dst_data),
      esz, static_cast<int>(ndim_), d_dst_strides_.get(), d_start_.get(), d_step_.get(),
      d_slice_shape_.get(), total_i64);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
SliceAssignFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 2, "expected exactly 2 inputs");
  const auto &src_desc = input_descs[0];
  const auto &dst_desc = input_descs[1];

  check(src_desc.mem_loc == holoflow::core::MemLoc::Device, "read tensor must be on Device");
  check(dst_desc.mem_loc == holoflow::core::MemLoc::Device, "write tensor must be on Device");
  check(src_desc.dtype == dst_desc.dtype, "read/write dtypes must match");

  const int dst_ndim = static_cast<int>(dst_desc.shape.size());
  const int src_ndim = static_cast<int>(src_desc.shape.size());
  check(dst_ndim > 0, "write ndim must be > 0");
  check(dst_ndim <= kMaxNDim, "write ndim too large");
  check(src_ndim > 0, "read ndim must be > 0");
  check(src_ndim <= kMaxNDim, "read ndim too large");

  // Ensure dtype supported (throws if unknown).
  (void)size_of(src_desc.dtype);

  check(product_shape(src_desc.shape) > 0, "read tensor has zero elements");
  check(product_shape(dst_desc.shape) > 0, "write tensor has zero elements");

  const auto settings = jsettings.get<SliceAssignSettings>();
  const auto nslices  = normalize_and_validate_slices(settings.slices, dst_desc.shape);

  // Compute slice shape (the region being written), always in dst rank.
  std::vector<size_t> slice_shape(static_cast<size_t>(dst_ndim));
  for (int i = 0; i < dst_ndim; ++i) {
    slice_shape[static_cast<size_t>(i)] =
        static_cast<size_t>(out_len(nslices[static_cast<size_t>(i)]));
  }

  const bool same_rank = (src_ndim == dst_ndim);
  const bool slot_rank = (src_ndim + 2 == dst_ndim);

  check(same_rank || slot_rank,
        "unsupported ranks: expected src.ndim == dst.ndim, or dst.ndim == src.ndim + 2");

  // NumPy slot-assign mode:
  // dst: (sy,sx,...) , src: (...) and slice must select exactly one (iy,ix) slot.
  check(slice_shape[0] == 1 && slice_shape[1] == 1,
        "slot-assign requires first two slice dims to have length 1 (iy:iy+1, ix:ix+1)");

  for (int k = 0; k < src_ndim; ++k) {
    const auto expected = slice_shape[static_cast<size_t>(k)];
    const auto actual   = src_desc.shape[static_cast<size_t>(k)];
    check(actual == expected, "read shape must match slice shape on trailing dims");
  }

  const auto total_out = product_shape(slice_shape);
  check(total_out > 0, "slice produces an empty tensor (not supported for now)");

  return holoflow::core::InferResult{
      .input_descs   = {src_desc, dst_desc},
      .output_descs  = {dst_desc},
      .in_place      = {{1, 0}},
      .owned_inputs  = {false, false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SliceAssignFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                           const nlohmann::json                  &jsettings,
                           const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate and lock semantics via infer.
  (void)this->infer(input_descs, jsettings);

  const auto  settings = jsettings.get<SliceAssignSettings>();
  const auto &dst_desc = input_descs[1];

  const size_t ndim = dst_desc.shape.size();
  const auto   ns   = normalize_and_validate_slices(settings.slices, dst_desc.shape);

  const auto dst_strides_vec = make_contig_strides(dst_desc.shape);
  auto       h_dst_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_dst_strides.get()[i] = dst_strides_vec[i];
  }

  auto h_start = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  auto h_step  = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    h_start.get()[i] = ns[i].start;
    h_step.get()[i]  = ns[i].step;
  }

  // Compute slice_shape in dst rank (always), and total elems to write.
  auto                h_slice_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  std::vector<size_t> slice_shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    const auto dim_len     = static_cast<size_t>(out_len(ns[i]));
    slice_shape[i]         = dim_len;
    h_slice_shape.get()[i] = static_cast<std::int64_t>(dim_len);
  }

  auto d_dst_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_start       = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_step        = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_slice_shape = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_dst_strides.get(), h_dst_strides.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_start.get(), h_start.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_step.get(), h_step.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_slice_shape.get(), h_slice_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));

  const size_t slice_elems = product_shape(slice_shape);

  return std::unique_ptr<holoflow::core::ISyncTask>(new SliceAssign(
      settings, ctx.stream, ndim, slice_elems, std::move(h_dst_strides), std::move(d_dst_strides),
      std::move(h_start), std::move(d_start), std::move(h_step), std::move(d_step),
      std::move(h_slice_shape), std::move(d_slice_shape)));
}

} // namespace holonp