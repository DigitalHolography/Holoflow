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

#include "curaii/cuda.hh"

#include <numeric>
#include <stdexcept>

namespace holonp {

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

// Resolves -1 in target shape and validates the total element count
std::vector<size_t> resolve_shape(const std::vector<int64_t> &target_shape, size_t total_elems) {
  std::vector<size_t> result;
  result.reserve(target_shape.size());

  int64_t unknown_idx   = -1;
  size_t  known_product = 1;

  for (size_t i = 0; i < target_shape.size(); ++i) {
    int64_t dim = target_shape[i];
    if (dim == -1) {
      if (unknown_idx != -1) {
        throw std::invalid_argument("Reshape: Only one dimension can be -1");
      }
      unknown_idx = static_cast<int64_t>(i);
    } else if (dim < 0) {
      throw std::invalid_argument("Reshape: Dimensions cannot be negative (except -1)");
    } else {
      known_product *= static_cast<size_t>(dim);
      result.push_back(static_cast<size_t>(dim));
    }
  }

  if (unknown_idx != -1) {
    if (known_product == 0 || total_elems % known_product != 0) {
      throw std::invalid_argument("Reshape: Total elements must be divisible by known dimensions");
    }
    result.insert(result.begin() + unknown_idx, total_elems / known_product);
  } else if (known_product != total_elems) {
    throw std::invalid_argument("Reshape: Total elements mismatch between input and target shape");
  }

  return result;
}

// Determines if a reshape can be done as a view by computing the new strides.
// Returns std::nullopt if the required layout changes force a copy.
std::optional<std::vector<size_t>> compute_view_strides(const std::vector<size_t> &old_shape,
                                                        const std::vector<size_t> &old_strides,
                                                        const std::vector<size_t> &new_shape) {

  // Edge case: Empty tensor. Just return standard C-contiguous strides.
  size_t numel = 1;
  for (auto s : old_shape)
    numel *= s;
  if (numel == 0) {
    std::vector<size_t> new_strides(new_shape.size(), 1);
    size_t              current_stride = 1;
    for (size_t i = new_shape.size(); i > 0; --i) {
      new_strides[i - 1] = current_stride;
      // Use max(1, ...) so preceding dims don't get a 0 stride
      current_stride *= std::max<size_t>(1, new_shape[i - 1]);
    }
    return new_strides;
  }

  std::vector<size_t> new_strides(new_shape.size(), 0);
  size_t              o_idx = 0;
  size_t              n_idx = 0;

  while (n_idx < new_shape.size() || o_idx < old_shape.size()) {
    size_t o_end  = o_idx;
    size_t n_end  = n_idx;
    size_t o_prod = 1;
    size_t n_prod = 1;

    // Advance to at least 1 element from each side
    if (o_end < old_shape.size())
      o_prod *= old_shape[o_end++];
    if (n_end < new_shape.size())
      n_prod *= new_shape[n_end++];

    // Expand the chunks until the element counts match
    while (o_prod != n_prod) {
      if (o_prod < n_prod) {
        if (o_end == old_shape.size())
          return std::nullopt;
        o_prod *= old_shape[o_end++];
      } else {
        if (n_end == new_shape.size())
          return std::nullopt;
        n_prod *= new_shape[n_end++];
      }
    }

    // Verify contiguity of the old chunk.
    if (o_idx < o_end) {
      for (size_t i = o_idx; i + 1 < o_end; ++i) {
        if (old_shape[i] == 1)
          continue; // Size-1 dimensions don't affect contiguity

        size_t next_idx = i + 1;
        while (next_idx < o_end && old_shape[next_idx] == 1) {
          next_idx++;
        }

        if (next_idx < o_end) {
          if (old_strides[i] != old_shape[next_idx] * old_strides[next_idx]) {
            return std::nullopt; // The chunk is not contiguous, requires copy
          }
        }
      }
    }

    // Determine the base stride for the new chunk.
    size_t current_stride = 1;
    if (o_idx < o_end) {
      current_stride = old_strides[o_end - 1];
      // Search backwards to use the innermost non-1 dimension for a reliable base stride
      for (size_t i = o_end; i > o_idx; --i) {
        if (old_shape[i - 1] != 1) {
          current_stride = old_strides[i - 1];
          break;
        }
      }
    }

    // Map the computed strides cleanly to the new chunk from right to left.
    // This naturally sets the correct stride for any new size-1 dimensions.
    for (size_t i = n_end; i > n_idx; --i) {
      new_strides[i - 1] = current_stride;
      current_stride *= new_shape[i - 1];
    }

    o_idx = o_end;
    n_idx = n_end;
  }

  return new_strides;
}

// Reads N-dimensional strided data and writes it linearly to contiguous memory.
__global__ void reshape_copy_kernel(const std::byte *__restrict__ src, std::byte *__restrict__ dst,
                                    const int64_t *__restrict__ src_strides,
                                    const int64_t *__restrict__ src_shape, int ndim, int elem_size,
                                    int64_t total_elems) {

  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_elems)
    return;

  int64_t rem     = tid;
  int64_t src_off = 0;

  for (int i = ndim - 1; i >= 0; --i) {
    int64_t idx = rem % src_shape[i];
    rem /= src_shape[i];
    src_off += idx * src_strides[i];
  }

  const std::byte *s_ptr = src + src_off;
  std::byte       *d_ptr = dst + (tid * elem_size);

  for (int b = 0; b < elem_size; ++b) {
    d_ptr[b] = s_ptr[b];
  }
}

bool same_desc(const holoflow::core::TDesc &a, const holoflow::core::TDesc &b) {
  return a.shape == b.shape && a.strides == b.strides && a.dtype == b.dtype &&
         a.mem_loc == b.mem_loc && a.offset == b.offset;
}

// -------------------------------------------------------------------------------------------------
// Reshape task implementation
// -------------------------------------------------------------------------------------------------

class Reshape : public holoflow::core::ISyncTask {
public:
  Reshape(ReshapeSettings settings, holoflow::core::TDesc idesc, size_t ndim, size_t total_elems,
          size_t elem_size, curaii::unique_device_ptr<int64_t> d_src_strides,
          curaii::unique_device_ptr<int64_t> d_src_shape, cudaStream_t stream)
      : is_view_(false), settings_(std::move(settings)), idesc_(std::move(idesc)), ndim_(ndim),
        total_elems_(total_elems), elem_size_(elem_size), stream_(stream),
        d_src_strides_(std::move(d_src_strides)), d_src_shape_(std::move(d_src_shape)) {}

  Reshape(ReshapeSettings settings, holoflow::core::TDesc idesc)
      : is_view_(true), settings_(std::move(settings)), idesc_(std::move(idesc)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  const ReshapeSettings       &settings() const { return settings_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  bool                  is_view_;
  ReshapeSettings       settings_;
  holoflow::core::TDesc idesc_;
  size_t                ndim_        = 0;
  size_t                total_elems_ = 0;
  size_t                elem_size_   = 0;
  cudaStream_t          stream_      = nullptr;
  curaii::unique_device_ptr<int64_t> d_src_strides_;
  curaii::unique_device_ptr<int64_t> d_src_shape_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ReshapeSettings &s) {
  j = nlohmann::json{{"shape", s.shape}};
  if (s.copy.has_value())
    j["copy"] = s.copy.value();
  else
    j["copy"] = nullptr;
}

void from_json(const nlohmann::json &j, ReshapeSettings &s) {
  j.at("shape").get_to(s.shape);
  if (j.contains("copy") && !j.at("copy").is_null()) {
    s.copy = j.at("copy").get<bool>();
  } else {
    s.copy = std::nullopt;
  }
}

// -------------------------------------------------------------------------------------------------
// ReshapeFactory
// -------------------------------------------------------------------------------------------------
holoflow::core::InferResult
ReshapeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {

  if (input_descs.size() != 1) {
    throw std::runtime_error("Reshape requires exactly 1 input");
  }

  const auto &src      = input_descs[0];
  auto        settings = jsettings.get<ReshapeSettings>();

  auto out_shape    = resolve_shape(settings.shape, src.num_elements());
  auto view_strides = compute_view_strides(src.shape, src.strides, out_shape);

  // Decide if we must copy
  bool must_copy = false;
  if (settings.copy.has_value()) {
    must_copy = *settings.copy;
    if (!must_copy && !view_strides.has_value()) {
      throw std::invalid_argument(
          "Reshape: copy=false requested, but tensor layout prevents a view.");
    }
  } else {
    must_copy = !view_strides.has_value();
  }

  holoflow::core::TDesc                out_desc;
  std::vector<holoflow::core::InPlace> in_place;

  if (must_copy) {
    out_desc = holoflow::core::TDesc(out_shape, src.dtype, src.mem_loc);
    in_place = {};
  } else {
    out_desc         = src;
    out_desc.shape   = out_shape;
    out_desc.strides = view_strides.value(); // Apply the freshly computed strides
    in_place         = {{0, 0}};
  }

  return {.input_descs   = {src},
          .output_descs  = {out_desc},
          .in_place      = in_place,
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
ReshapeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {

  const auto &src      = input_descs[0];
  auto        settings = jsettings.get<ReshapeSettings>();

  auto resolved_shape = resolve_shape(settings.shape, src.num_elements());
  auto view_strides   = compute_view_strides(src.shape, src.strides, resolved_shape);

  bool doing_copy = settings.copy.value_or(!view_strides.has_value());

  if (!doing_copy) {
    return std::make_unique<Reshape>(settings, src);
  }

  // Copy mode: Prepare kernel arguments
  size_t               ndim = src.shape.size();
  std::vector<int64_t> h_strides(ndim);
  std::vector<int64_t> h_shape(ndim);

  for (size_t i = 0; i < ndim; ++i) {
    h_strides[i] = static_cast<int64_t>(src.strides[i]);
    h_shape[i]   = static_cast<int64_t>(src.shape[i]);
  }

  auto   d_strides = curaii::make_unique_device_ptr<int64_t>(ndim);
  auto   d_shape   = curaii::make_unique_device_ptr<int64_t>(ndim);
  size_t bytes     = ndim * sizeof(int64_t);

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
    const auto &old_idesc    = old_reshape->idesc();

    bool can_reuse =
        (new_settings == old_reshape->settings()) && same_desc(new_idesc, old_idesc);

    if (can_reuse) {
      old_reshape->update_stream(ctx.stream);
      return old_task;
    }
  }

  return create(input_descs, jsettings, ctx);
}

holoflow::core::OpResult Reshape::execute(holoflow::core::SyncCtx &ctx) {
  if (is_view_ || total_elems_ == 0) {
    return holoflow::core::OpResult::Ok;
  }

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
