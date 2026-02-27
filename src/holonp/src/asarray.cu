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

#include "holonp/asarray.hh"
#include "holonp/ascontiguousarray.hh"
#include "holonp/copy.hh"

#include <cstdint>
#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const AsArraySettings &s) {
  j = nlohmann::json{{"value", s.value}};
  if (s.device) {
    j["device"] = *s.device;
  }
}

void from_json(const nlohmann::json &j, AsArraySettings &s) {
  j.at("value").get_to(s.value);
  if (j.contains("device")) {
    s.device = j.at("device").get<holoflow::core::MemLoc>();
  } else {
    s.device = std::nullopt;
  }
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("AsArrayFactory inference error: " + msg);
  }
}

} // namespace

AsArray::AsArray(const AsArraySettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult AsArray::execute(holoflow::core::SyncCtx &ctx) {
  auto *odata = ctx.outputs[0].data();
  auto  odesc = ctx.outputs[0].desc;

  if (odesc.dtype != holoflow::core::DType::F32) {
    logger()->error("[AsArray::execute] unsupported dtype");
    std::abort();
  }

  const float value = static_cast<float>(settings_.value);
  CUDA_CHECK(cudaMemcpyAsync(odata, &value, sizeof(float), cudaMemcpyHostToDevice, stream_));
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
AsArrayFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<AsArraySettings>();
  const auto memloc   = settings.device.value_or(holoflow::core::MemLoc::Device);

  check(input_descs.empty(), "expected zero inputs");
  check(memloc == holoflow::core::MemLoc::Device, "only Device output is supported (for now)");

  holoflow::core::TDesc odesc({1}, holoflow::core::DType::F32, memloc);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AsArrayFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<AsArraySettings>();
  (void)infer;

  return std::unique_ptr<holoflow::core::ISyncTask>(new AsArray(settings, ctx.stream));
}

} // namespace holonp

namespace holonp {

namespace {

inline void check_ascontig(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("AsContiguousArray: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

__global__ void ascontiguousarray_copy_kernel(const std::byte *__restrict__ src,
                                              std::byte *__restrict__ dst,
                                              const std::int64_t *__restrict__ src_strides,
                                              const std::int64_t *__restrict__ src_shape, int ndim,
                                              int elem_size, std::int64_t total_elems) {
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

} // namespace

void to_json(nlohmann::json &j, const AsContiguousArraySettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, AsContiguousArraySettings &) {}

holoflow::core::InferResult
AsContiguousArrayFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                const nlohmann::json                  &jsettings) const {
  (void)jsettings.get<AsContiguousArraySettings>();

  check_ascontig(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check_ascontig(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  std::vector<holoflow::core::InPlace> in_place;
  holoflow::core::TDesc                odesc = idesc;
  if (!is_c_contiguous(idesc)) {
    odesc = holoflow::core::TDesc(idesc.shape, idesc.dtype, idesc.mem_loc);
  } else {
    in_place = {{0, 0}};
  }

  return {
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = in_place,
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
AsContiguousArrayFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings,
                                 const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)jsettings.get<AsContiguousArraySettings>();

  const auto &idesc = input_descs[0];
  if (is_c_contiguous(idesc)) {
    return std::unique_ptr<holoflow::core::ISyncTask>(new AsContiguousArray());
  }

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

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new AsContiguousArray(ndim, idesc.num_elements(), holoflow::core::size_of(idesc.dtype),
                            std::move(d_strides), std::move(d_shape), ctx.stream));
}

AsContiguousArray::AsContiguousArray() : is_noop_(true) {}

AsContiguousArray::AsContiguousArray(size_t ndim, size_t total_elems, size_t elem_size,
                                     curaii::unique_device_ptr<std::int64_t> d_src_strides,
                                     curaii::unique_device_ptr<std::int64_t> d_src_shape,
                                     cudaStream_t stream)
    : is_noop_(false), ndim_(ndim), total_elems_(total_elems), elem_size_(elem_size),
      stream_(stream), d_src_strides_(std::move(d_src_strides)),
      d_src_shape_(std::move(d_src_shape)) {}

holoflow::core::OpResult AsContiguousArray::execute(holoflow::core::SyncCtx &ctx) {
  if (is_noop_ || total_elems_ == 0) {
    return holoflow::core::OpResult::Ok;
  }

  const std::byte *src = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  std::byte       *dst = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block = 256;
  const int     grid =
      static_cast<int>((static_cast<std::int64_t>(total_elems_) + block - 1) / block);

  ascontiguousarray_copy_kernel<<<grid, block, 0, stream_>>>(
      src, dst, d_src_strides_.get(), d_src_shape_.get(), static_cast<int>(ndim_),
      static_cast<int>(elem_size_), static_cast<std::int64_t>(total_elems_));
  CUDA_CHECK(cudaGetLastError());

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp

namespace holonp {

namespace {

inline void check_copy(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Copy: " + msg);
  }
}

} // namespace

void to_json(nlohmann::json &j, const CopySettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, CopySettings &) {}

holoflow::core::InferResult CopyFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  (void)jsettings.get<CopySettings>();

  check_copy(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check_copy(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");

  // Always return a fresh dense descriptor (no offset, default compact strides).
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
  (void)jsettings.get<CopySettings>();
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

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new Copy(ndim, idesc.num_elements(), holoflow::core::size_of(idesc.dtype),
               std::move(d_strides), std::move(d_shape), ctx.stream));
}

Copy::Copy(size_t ndim, size_t total_elems, size_t elem_size,
           curaii::unique_device_ptr<std::int64_t> d_src_strides,
           curaii::unique_device_ptr<std::int64_t> d_src_shape, cudaStream_t stream)
    : ndim_(ndim), total_elems_(total_elems), elem_size_(elem_size), stream_(stream),
      d_src_strides_(std::move(d_src_strides)), d_src_shape_(std::move(d_src_shape)) {}

holoflow::core::OpResult Copy::execute(holoflow::core::SyncCtx &ctx) {
  if (total_elems_ == 0) {
    return holoflow::core::OpResult::Ok;
  }

  const std::byte *src = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  std::byte       *dst = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block = 256;
  const int     grid =
      static_cast<int>((static_cast<std::int64_t>(total_elems_) + block - 1) / block);

  ascontiguousarray_copy_kernel<<<grid, block, 0, stream_>>>(
      src, dst, d_src_strides_.get(), d_src_shape_.get(), static_cast<int>(ndim_),
      static_cast<int>(elem_size_), static_cast<std::int64_t>(total_elems_));
  CUDA_CHECK(cudaGetLastError());

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp
