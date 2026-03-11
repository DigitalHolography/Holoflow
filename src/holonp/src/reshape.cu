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

#include "holonp/reshape.hh"
#include <numeric>

namespace holonp {

namespace {

// Checks if a tensor is C-contiguous (row-major without gaps)
bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  size_t stride = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != stride)
      return false;
    stride *= desc.shape[i];
  }
  return true;
}

// Resolves -1 in target shape and validates element count
std::vector<size_t> resolve_shape(const std::vector<int64_t> &target_shape, size_t total_elems) {
  std::vector<size_t> result;
  result.reserve(target_shape.size());

  int64_t unknown_idx   = -1;
  size_t  known_product = 1;

  for (size_t i = 0; i < target_shape.size(); ++i) {
    int64_t dim = target_shape[i];
    if (dim == -1) {
      if (unknown_idx != -1)
        throw std::invalid_argument("Reshape: Only one dimension can be -1");
      unknown_idx = static_cast<int64_t>(i);
    } else if (dim < 0) {
      throw std::invalid_argument("Reshape: Dimensions cannot be negative (except -1)");
    } else {
      known_product *= static_cast<size_t>(dim);
      result.push_back(static_cast<size_t>(dim));
    }
  }

  if (unknown_idx != -1) {
    if (total_elems % known_product != 0) {
      throw std::invalid_argument("Reshape: Total elements must be divisible by known dimensions");
    }
    size_t missing = total_elems / known_product;
    result.insert(result.begin() + unknown_idx, missing);
  } else {
    if (known_product != total_elems) {
      throw std::invalid_argument("Reshape: Total elements mismatch");
    }
  }
  return result;
}

// "Compaction" kernel: Reads strided data, writes contiguous data (row-major)
__global__ void reshape_copy_kernel(const std::byte *__restrict__ src, std::byte *__restrict__ dst,
                                    const int64_t *__restrict__ src_strides,
                                    const int64_t *__restrict__ src_shape, int ndim, int elem_size,
                                    int64_t total_elems) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elems)
    return;

  int64_t rem     = tid;
  int64_t src_off = 0;

  // Since destination is C-contiguous, 'tid' maps directly to the linear sequence
  // of elements. We just need to find where that element lives in the source.
  for (int i = ndim - 1; i >= 0; --i) {
    int64_t idx = rem % src_shape[i];
    rem /= src_shape[i];
    src_off += idx * src_strides[i];
  }

  const std::byte *s_ptr = src + src_off;
  std::byte       *d_ptr = dst + (tid * elem_size); // dst is contiguous

  for (int b = 0; b < elem_size; ++b) {
    d_ptr[b] = s_ptr[b];
  }
}

} // namespace

void to_json(nlohmann::json &j, const ReshapeSettings &s) {
  j = nlohmann::json{{"shape", s.shape}};
  if (s.copy.has_value()) {
    j["copy"] = s.copy.value();
  } else {
    j["copy"] = nullptr;
  }
}

void from_json(const nlohmann::json &j, ReshapeSettings &s) {
  j.at("shape").get_to(s.shape);
  if (j.contains("copy") && !j.at("copy").is_null()) {
    s.copy = j.at("copy").get<bool>();
  } else {
    s.copy = std::nullopt;
  }
}

// -----------------------------------------------------------------------------
// Factory: Infer
// -----------------------------------------------------------------------------
holoflow::core::InferResult
ReshapeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  if (input_descs.size() != 1)
    throw std::runtime_error("Reshape requires 1 input");

  const auto &src      = input_descs[0];
  auto        settings = jsettings.get<ReshapeSettings>();

  // 1. Resolve Output Shape
  auto out_shape = resolve_shape(settings.shape, src.num_elements());

  // 2. Determine if Copy is needed
  //    To return a view with order='C', the input must currently be C-contiguous.
  bool src_is_contig = is_c_contiguous(src);
  bool must_copy     = false;

  if (settings.copy.has_value()) {
    if (*settings.copy == true) {
      must_copy = true; // User forced copy
    } else {
      // User forced View (copy=false)
      if (!src_is_contig) {
        throw std::invalid_argument("Reshape: copy=false but array is not C-contiguous");
      }
      must_copy = false;
    }
  } else {
    // Auto (copy=nullopt): Copy only if we can't view
    must_copy = !src_is_contig;
  }

  holoflow::core::TDesc                out_desc;
  std::vector<holoflow::core::InPlace> in_place;

  if (must_copy) {
    // New allocation, default C-strides (calculated by TDesc ctor)
    out_desc = holoflow::core::TDesc(out_shape, src.dtype, src.mem_loc);
    in_place = {}; // No in-place possible
  } else {
    // View: Point to same data, new shape, default C-strides (valid because src is contig)
    out_desc       = src; // Copy pointer/offset
    out_desc.shape = out_shape;
    // Recalculate strides for the new shape assuming C-order
    out_desc = holoflow::core::TDesc(out_shape, src.dtype, src.mem_loc, src.offset);
    in_place = {{0, 0}}; // Input 0 reused for Output 0
  }

  return {.input_descs   = {src},
          .output_descs  = {out_desc},
          .in_place      = in_place,
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

// -----------------------------------------------------------------------------
// Factory: Create
// -----------------------------------------------------------------------------
std::unique_ptr<holoflow::core::ISyncTask>
ReshapeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto &src      = input_descs[0];
  auto        settings = jsettings.get<ReshapeSettings>();

  bool src_is_contig = is_c_contiguous(src);
  bool doing_copy    = false;

  if (settings.copy.has_value()) {
    doing_copy = *settings.copy;
    // Validation already done in infer
  } else {
    doing_copy = !src_is_contig;
  }

  if (!doing_copy) {
    // View mode: Task does nothing at runtime
    return std::make_unique<Reshape>(settings, src);
  }

  // Copy mode: Prepare kernel args
  size_t               ndim = src.shape.size();
  std::vector<int64_t> h_strides(ndim);
  std::vector<int64_t> h_shape(ndim);

  for (size_t i = 0; i < ndim; ++i) {
    h_strides[i] = static_cast<int64_t>(src.strides[i]);
    h_shape[i]   = static_cast<int64_t>(src.shape[i]);
  }

  auto d_strides = curaii::make_unique_device_ptr<int64_t>(ndim);
  auto d_shape   = curaii::make_unique_device_ptr<int64_t>(ndim);

  size_t bytes = ndim * sizeof(int64_t);
  CUDA_CHECK(cudaMemcpyAsync(d_strides.get(), h_strides.data(), bytes, cudaMemcpyHostToDevice,
                             ctx.stream));
  CUDA_CHECK(
      cudaMemcpyAsync(d_shape.get(), h_shape.data(), bytes, cudaMemcpyHostToDevice, ctx.stream));

  return std::make_unique<Reshape>(settings, src, ndim, src.num_elements(),
                                   holoflow::core::size_of(src.dtype), std::move(d_strides),
                                   std::move(d_shape), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ReshapeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {

  auto *old_reshape = dynamic_cast<Reshape *>(old_task.get());
  if (old_reshape != nullptr && input_descs.size() == 1) {

    const auto  new_settings = jsettings.get<ReshapeSettings>();
    const auto &new_idesc    = input_descs[0];
    const auto &old_idesc    = old_reshape->get_idesc();

    bool can_reuse =
        (new_settings == old_reshape->get_settings()) && (new_idesc.shape == old_idesc.shape) &&
        (new_idesc.strides == old_idesc.strides) && (new_idesc.dtype == old_idesc.dtype) &&
        (new_idesc.mem_loc == old_idesc.mem_loc);

    if (can_reuse) {
      old_reshape->update_stream(ctx.stream);
      return old_task;
    }
  }

  // Fallback: Structural change detected or invalid old task.
  return create(input_descs, jsettings, ctx);
}

// -----------------------------------------------------------------------------
// Task Execution
// -----------------------------------------------------------------------------

// View Constructor
Reshape::Reshape(const ReshapeSettings &settings, const holoflow::core::TDesc &idesc)
    : is_view_(true), settings_(settings), idesc_(idesc) {}

// Copy Constructor
Reshape::Reshape(const ReshapeSettings &settings, const holoflow::core::TDesc &idesc, size_t ndim,
                 size_t total_elems, size_t elem_size,
                 curaii::unique_device_ptr<int64_t> d_src_strides,
                 curaii::unique_device_ptr<int64_t> d_src_shape, cudaStream_t stream)
    : is_view_(false), settings_(settings), idesc_(idesc), ndim_(ndim), total_elems_(total_elems),
      elem_size_(elem_size), stream_(stream), d_src_strides_(std::move(d_src_strides)),
      d_src_shape_(std::move(d_src_shape)) {}

holoflow::core::OpResult Reshape::execute(holoflow::core::SyncCtx &ctx) {
  if (is_view_) {
    // The framework has already wired the output TView to point to the input's buffer
    // (via the TDesc returned in Infer). We don't need to move bytes.
    return holoflow::core::OpResult::Ok;
  }

  // Copy Execution
  if (total_elems_ == 0)
    return holoflow::core::OpResult::Ok;

  const std::byte *src_ptr = reinterpret_cast<const std::byte *>(ctx.inputs[0].data());
  std::byte       *dst_ptr = reinterpret_cast<std::byte *>(ctx.outputs[0].data());

  constexpr int block_size = 256;
  int           grid_size  = static_cast<int>((total_elems_ + block_size - 1) / block_size);

  reshape_copy_kernel<<<grid_size, block_size, 0, stream_>>>(
      src_ptr, dst_ptr, d_src_strides_.get(), d_src_shape_.get(), static_cast<int>(ndim_),
      static_cast<int>(elem_size_), static_cast<int64_t>(total_elems_));

  return holoflow::core::OpResult::Ok;
}

} // namespace holonp