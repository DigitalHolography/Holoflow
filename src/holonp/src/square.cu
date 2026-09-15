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

#include "holonp/square.hh"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SquareSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, SquareSettings &) {}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

template <typename T>
__global__ void square_kernel(const T *__restrict__ a, T *__restrict__ out, size_t total_out,
                              size_t ndim, const size_t *__restrict__ out_shape,
                              const size_t *__restrict__ a_strides) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_out)
    return;

  size_t a_off = 0, rem = idx;

  for (int i = int(ndim) - 1; i >= 0; --i) {
    size_t coord = rem % out_shape[i];
    rem /= out_shape[i];
    a_off += coord * a_strides[i];
  }

  if constexpr (std::is_same_v<T, cuFloatComplex>) {
    out[idx] = cuCmulf(a[a_off], a[a_off]);
  } else {
    out[idx] = a[a_off] * a[a_off];
  }
}

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("Square: " + msg);
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
// Square task implementation
// -------------------------------------------------------------------------------------------------

class Square : public holoflow::core::ISyncTask {
public:
  Square(SquareSettings settings, holoflow::core::TDesc idesc, cudaStream_t stream,
         size_t total_out, size_t ndim, DevPtr<size_t> d_out_shape, DevPtr<size_t> d_a_strides)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), stream_(stream),
        total_out_(total_out), ndim_(ndim), d_out_shape_(std::move(d_out_shape)),
        d_a_strides_(std::move(d_a_strides)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  SquareSettings        settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  size_t                total_out_;
  size_t                ndim_;
  DevPtr<size_t>        d_out_shape_;
  DevPtr<size_t>        d_a_strides_;
};

} // namespace

holoflow::core::OpResult Square::execute(holoflow::core::SyncCtx &ctx) {
  if (total_out_ == 0)
    return holoflow::core::OpResult::Ok;

  int block = 256;
  int grid  = static_cast<int>((total_out_ + block - 1) / block);

#define LAUNCH_TYPE(T)                                                                             \
  square_kernel<T><<<grid, block, 0, stream_>>>((T *)ctx.inputs[0].data(),                         \
                                                (T *)ctx.outputs[0].data(), total_out_, ndim_,     \
                                                d_out_shape_.get(), d_a_strides_.get())

  switch (idesc_.dtype) {
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
    logger()->error("[Square::execute] unsupported dtype");
    std::abort();
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// SquareFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult SquareFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                 const nlohmann::json &) const {
  check(inputs.size() == 1, "expected exactly 1 input tensor");
  check(inputs[0].mem_loc == holoflow::core::MemLoc::Device,
        "input tensor 0 must be in device memory");

  const auto &a = inputs[0];
  check(a.dtype == holoflow::core::DType::U8 || a.dtype == holoflow::core::DType::U16 ||
            a.dtype == holoflow::core::DType::F32 || a.dtype == holoflow::core::DType::CF32,
        "unsupported input dtype");
  check(a.num_elements() > 0, "input tensor has zero elements");

  const size_t elem_size = holoflow::core::size_of(a.dtype);
  for (const auto stride : a.strides) {
    check(stride % elem_size == 0, "input strides must be a multiple of the element size");
  }

  holoflow::core::TDesc o(a.shape, a.dtype, holoflow::core::MemLoc::Device);

  return holoflow::core::InferResult{
      .input_descs   = {a},
      .output_descs  = {o},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SquareFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &j,
                      const holoflow::core::SyncCreateCtx &ctx) const {
  const auto &a     = inputs[0];
  auto        res   = infer(inputs, j);
  const auto &odesc = res.output_descs[0];
  size_t      ndim  = odesc.shape.size();

  const size_t        total_out = odesc.num_elements();
  std::vector<size_t> a_strides_h(ndim);
  auto                as_raw = get_elem_strides(a);

  for (size_t i = 0; i < ndim; ++i) {
    a_strides_h[i] = as_raw[i];
  }

  auto d_shape = curaii::make_unique_device_ptr<size_t>(ndim, ctx.stream);
  auto d_a_str = curaii::make_unique_device_ptr<size_t>(ndim, ctx.stream);
  auto bytes   = ndim * sizeof(size_t);
  auto h2d     = cudaMemcpyHostToDevice;

  CUDA_CHECK(cudaMemcpyAsync(d_shape.get(), odesc.shape.data(), bytes, h2d, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(d_a_str.get(), a_strides_h.data(), bytes, h2d, ctx.stream));

  const auto settings = j.get<SquareSettings>();
  return std::make_unique<Square>(settings, a, ctx.stream, total_out, ndim, std::move(d_shape),
                                  std::move(d_a_str));
}

std::unique_ptr<holoflow::core::ISyncTask>
SquareFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_square = dynamic_cast<Square *>(old_task.get());
  if (old_square == nullptr || input_descs.size() != 1) {
    return create(input_descs, jsettings, ctx);
  }

  if (same_desc(input_descs[0], old_square->idesc())) {
    old_square->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
