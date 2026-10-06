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

#include "holonp/fftshift.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const FFTShiftSettings &s) {
  j = nlohmann::json{{"axes", s.axes}, {"inverse", s.inverse}};
}

void from_json(const nlohmann::json &j, FFTShiftSettings &s) {
  if (j.contains("axes") && !j.at("axes").is_null()) {
    j.at("axes").get_to(s.axes);
  } else {
    s.axes.clear();
  }
  s.inverse = j.value("inverse", false);
}

namespace {

constexpr int kMaxNDim = 16;

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("FFTShift: " + msg);
  }
}

__global__ void fftshift_nd_contig_kernel_u8(const std::uint8_t *__restrict__ in,
                                             std::uint8_t *__restrict__ out, int elem_size,
                                             int ndim, const std::int64_t *__restrict__ shape,
                                             const std::int64_t *__restrict__ strides,
                                             const std::int64_t *__restrict__ shifts,
                                             std::int64_t total_elems) {
  const std::int64_t tid =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
  if (tid >= total_elems) {
    return;
  }

  std::int64_t rem       = tid;
  std::int64_t in_off_el = 0;

  for (int k = 0; k < ndim; ++k) {
    const std::int64_t stride    = strides[k];
    const std::int64_t out_idx_k = (stride > 0) ? (rem / stride) : 0;
    rem -= out_idx_k * stride;

    const std::int64_t dim      = shape[k];
    const std::int64_t shift    = shifts[k];
    std::int64_t       in_idx_k = out_idx_k + shift;
    if (in_idx_k >= dim) {
      in_idx_k -= dim;
    }
    in_off_el += in_idx_k * stride;
  }

  const auto *src = in + static_cast<size_t>(in_off_el) * static_cast<size_t>(elem_size);
  auto       *dst = out + static_cast<size_t>(tid) * static_cast<size_t>(elem_size);

  for (int b = 0; b < elem_size; ++b) {
    dst[b] = src[b];
  }
}

class FFTShift : public holoflow::core::ISyncTask {
public:
  FFTShift(FFTShiftSettings settings, holoflow::core::TDesc idesc, cudaStream_t stream, size_t ndim,
           size_t total_elems, HostPtr<std::int64_t> h_shape, DevPtr<std::int64_t> d_shape,
           HostPtr<std::int64_t> h_strides, DevPtr<std::int64_t> d_strides,
           HostPtr<std::int64_t> h_shifts, DevPtr<std::int64_t> d_shifts)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream), ndim_(ndim),
        total_elems_(total_elems), h_shape_(std::move(h_shape)), d_shape_(std::move(d_shape)),
        h_strides_(std::move(h_strides)), d_strides_(std::move(d_strides)),
        h_shifts_(std::move(h_shifts)), d_shifts_(std::move(d_shifts)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  const FFTShiftSettings      &settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  FFTShiftSettings      settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                ndim_;
  size_t                total_elems_;
  HostPtr<std::int64_t> h_shape_;
  DevPtr<std::int64_t>  d_shape_;
  HostPtr<std::int64_t> h_strides_;
  DevPtr<std::int64_t>  d_strides_;
  HostPtr<std::int64_t> h_shifts_;
  DevPtr<std::int64_t>  d_shifts_;
};

} // namespace

holoflow::core::OpResult FFTShift::execute(holoflow::core::SyncCtx &ctx) {
  auto       *idata = ctx.inputs[0].data();
  auto       *odata = ctx.outputs[0].data();
  const auto &idesc = ctx.inputs[0].desc;

  constexpr int block_size = 256;
  const auto    total_i64  = static_cast<std::int64_t>(total_elems_);
  const int     grid_size  = static_cast<int>((total_i64 + block_size - 1) / block_size);

  const int esz = static_cast<int>(size_of(idesc.dtype));

  fftshift_nd_contig_kernel_u8<<<grid_size, block_size, 0, stream_>>>(
      reinterpret_cast<const std::uint8_t *>(idata), reinterpret_cast<std::uint8_t *>(odata), esz,
      static_cast<int>(ndim_), d_shape_.get(), d_strides_.get(), d_shifts_.get(), total_i64);

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
FFTShiftFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<FFTShiftSettings>();

  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];

  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(utils::is_c_contiguous(idesc), "input must be C-contiguous");

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0, "input ndim must be > 0");
  check(ndim <= kMaxNDim, "input ndim too large");

  (void)size_of(idesc.dtype);
  (void)utils::normalize_axes(settings.axes, ndim);

  const auto total = utils::product_shape(idesc.shape);
  check(total > 0, "input tensor has zero elements");

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {idesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTShiftFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<FFTShiftSettings>();

  const auto &idesc = input_descs[0];
  const int   ndim  = static_cast<int>(idesc.shape.size());
  const auto  total = utils::product_shape(idesc.shape);

  const auto axes = utils::normalize_axes(settings.axes, ndim);

  auto h_shape = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_shape.get()[i] = static_cast<std::int64_t>(idesc.shape[static_cast<size_t>(i)]);
  }

  const auto strides_vec = utils::compact_strides_i64(idesc.shape);
  auto       h_strides   = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_strides.get()[i] = strides_vec[static_cast<size_t>(i)];
  }

  auto h_shifts = curaii::make_unique_host_ptr<std::int64_t>(ndim);
  for (int i = 0; i < ndim; ++i) {
    h_shifts.get()[i] = 0;
  }
  for (int a : axes) {
    const auto half   = h_shape.get()[a] / 2;
    h_shifts.get()[a] = settings.inverse ? half : h_shape.get()[a] - half;
  }

  auto d_shape   = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_strides = curaii::make_unique_device_ptr<std::int64_t>(ndim);
  auto d_shifts  = curaii::make_unique_device_ptr<std::int64_t>(ndim);

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), h_shape.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_strides.get(), h_strides.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_shifts.get(), h_shifts.get(), sizeof(std::int64_t) * ndim,
                             cudaMemcpyHostToDevice, ctx.stream));

  return std::unique_ptr<holoflow::core::ISyncTask>(
      new FFTShift(settings, idesc, ctx.stream, static_cast<size_t>(ndim), total,
                   std::move(h_shape), std::move(d_shape), std::move(h_strides),
                   std::move(d_strides), std::move(h_shifts), std::move(d_shifts)));
}

std::unique_ptr<holoflow::core::ISyncTask>
FFTShiftFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {

  auto *old_fftshift = dynamic_cast<FFTShift *>(old_task.get());
  if (old_fftshift != nullptr && input_descs.size() == 1) {

    const auto  new_settings = jsettings.get<FFTShiftSettings>();
    const auto &new_idesc    = input_descs[0];
    const auto &old_idesc    = old_fftshift->idesc();

    bool can_reuse =
        (new_settings == old_fftshift->settings()) && utils::same_desc(new_idesc, old_idesc);

    if (can_reuse) {
      old_fftshift->update_stream(ctx.stream);
      return old_task;
    }
  }

  // Fallback: Structural change detected or invalid old task.
  return create(input_descs, jsettings, ctx);
}

namespace {
nlohmann::json inverse_settings(const nlohmann::json &settings) {
  auto result       = settings;
  result["inverse"] = true;
  return result;
}
} // namespace

holoflow::core::InferResult
IFFTShiftFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  return FFTShiftFactory{}.infer(input_descs, inverse_settings(jsettings));
}

std::unique_ptr<holoflow::core::ISyncTask>
IFFTShiftFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  return FFTShiftFactory{}.create(input_descs, inverse_settings(jsettings), ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
IFFTShiftFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                         std::span<const holoflow::core::TDesc>     input_descs,
                         const nlohmann::json                      &jsettings,
                         const holoflow::core::SyncCreateCtx       &ctx) const {
  return FFTShiftFactory{}.update(std::move(old_task), input_descs, inverse_settings(jsettings),
                                  ctx);
}

} // namespace holonp
