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

#include "holonp/unfold2d.hh"

#include <stdexcept>

#include "curaii/cuda.hh"

namespace holonp {

namespace {

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

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

// One thread per output element.
// Output layout (C-contiguous): [batch, ny_win, nx_win, win_h, win_w] * elem_size bytes
__global__ void unfold2d_kernel(const std::byte *__restrict__ src, std::byte *__restrict__ dst,
                                size_t batch, size_t H, size_t W, size_t win_h, size_t win_w,
                                size_t stride_y, size_t stride_x, size_t ny_win, size_t nx_win,
                                size_t elem_size) {
  const size_t total = batch * ny_win * nx_win * win_h * win_w;
  const size_t tid   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  // Decode flat index into (b, gy, gx, j, i)
  size_t rem = tid;
  size_t i   = rem % win_w;
  rem /= win_w;
  size_t j = rem % win_h;
  rem /= win_h;
  size_t gx = rem % nx_win;
  rem /= nx_win;
  size_t gy = rem % ny_win;
  rem /= ny_win;
  size_t b = rem;

  const size_t src_y   = gy * stride_y + j;
  const size_t src_x   = gx * stride_x + i;
  const size_t src_off = (b * H * W + src_y * W + src_x) * elem_size;
  const size_t dst_off = tid * elem_size;

  for (size_t k = 0; k < elem_size; ++k)
    dst[dst_off + k] = src[src_off + k];
}

// Extracts the batch size and validates that the input is contiguous.
// Returns {batch, H, W, ny_win, nx_win}.
struct UnfoldDims {
  size_t batch, H, W, ny_win, nx_win;
};

UnfoldDims compute_dims(const holoflow::core::TDesc &src, const Unfold2DSettings &s) {
  if (src.shape.size() < 2)
    throw std::invalid_argument("Unfold2D: input must have at least 2 dimensions");
  if (s.win_h == 0 || s.win_w == 0)
    throw std::invalid_argument("Unfold2D: window dimensions must be positive");
  if (s.stride_y == 0 || s.stride_x == 0)
    throw std::invalid_argument("Unfold2D: strides must be positive");

  const size_t H = src.shape[src.shape.size() - 2];
  const size_t W = src.shape[src.shape.size() - 1];

  if (s.win_h > H || s.win_w > W)
    throw std::invalid_argument("Unfold2D: window larger than input spatial dimensions");

  size_t batch = 1;
  for (size_t i = 0; i + 2 < src.shape.size(); ++i)
    batch *= src.shape[i];

  const size_t ny_win = (H - s.win_h) / s.stride_y + 1;
  const size_t nx_win = (W - s.win_w) / s.stride_x + 1;

  return {batch, H, W, ny_win, nx_win};
}

class Unfold2D : public holoflow::core::ISyncTask {
public:
  Unfold2D(Unfold2DSettings settings, holoflow::core::TDesc idesc, size_t batch, size_t H, size_t W,
           size_t ny_win, size_t nx_win, size_t elem_size, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), batch_(batch), H_(H), W_(W),
        ny_win_(ny_win), nx_win_(nx_win), elem_size_(elem_size), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const Unfold2DSettings      &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t s) { stream_ = s; }

private:
  Unfold2DSettings      settings_;
  holoflow::core::TDesc idesc_;
  size_t                batch_;
  size_t                H_;
  size_t                W_;
  size_t                ny_win_;
  size_t                nx_win_;
  size_t                elem_size_;
  cudaStream_t          stream_;
};

} // namespace

void to_json(nlohmann::json &j, const Unfold2DSettings &s) {
  j = nlohmann::json{
      {"win_h", s.win_h}, {"win_w", s.win_w}, {"stride_y", s.stride_y}, {"stride_x", s.stride_x}};
}

void from_json(const nlohmann::json &j, Unfold2DSettings &s) {
  j.at("win_h").get_to(s.win_h);
  j.at("win_w").get_to(s.win_w);
  j.at("stride_y").get_to(s.stride_y);
  j.at("stride_x").get_to(s.stride_x);
}

holoflow::core::OpResult Unfold2D::execute(holoflow::core::SyncCtx &ctx) {
  const auto *src = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  auto       *dst = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  const size_t total      = batch_ * ny_win_ * nx_win_ * settings_.win_h * settings_.win_w;
  const int    block_size = 256;
  const int    grid_size  = static_cast<int>((total + block_size - 1) / block_size);

  unfold2d_kernel<<<grid_size, block_size, 0, stream_>>>(
      src, dst, batch_, H_, W_, settings_.win_h, settings_.win_w, settings_.stride_y,
      settings_.stride_x, ny_win_, nx_win_, elem_size_);
  CUDA_CHECK(cudaGetLastError());

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
Unfold2DFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  if (input_descs.size() != 1)
    throw std::invalid_argument("Unfold2D: expects exactly 1 input");

  const auto &src      = input_descs[0];
  const auto  settings = jsettings.get<Unfold2DSettings>();
  if (!is_c_contiguous(src))
    throw std::invalid_argument("Unfold2D: input must be C-contiguous");
  const auto  dims     = compute_dims(src, settings);

  // Build output shape: replace last two dims (H, W) with (ny_win, nx_win, win_h, win_w)
  std::vector<size_t> out_shape(src.shape.begin(), src.shape.end() - 2);
  out_shape.push_back(dims.ny_win);
  out_shape.push_back(dims.nx_win);
  out_shape.push_back(settings.win_h);
  out_shape.push_back(settings.win_w);

  holoflow::core::TDesc out_desc(out_shape, src.dtype, src.mem_loc);

  return {.input_descs   = {src},
          .output_descs  = {out_desc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
Unfold2DFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto &src      = input_descs[0];
  const auto  settings = jsettings.get<Unfold2DSettings>();
  const auto  dims     = compute_dims(src, settings);

  return std::make_unique<Unfold2D>(settings, src, dims.batch, dims.H, dims.W, dims.ny_win,
                                    dims.nx_win, holoflow::core::size_of(src.dtype), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
Unfold2DFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  auto *old = dynamic_cast<Unfold2D *>(old_task.get());
  if (old != nullptr && input_descs.size() == 1) {
    (void)infer(input_descs, jsettings);
    const auto settings = jsettings.get<Unfold2DSettings>();
    if (settings == old->settings() && same_desc(input_descs[0], old->idesc())) {
      old->update_stream(ctx.stream);
      return old_task;
    }
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
