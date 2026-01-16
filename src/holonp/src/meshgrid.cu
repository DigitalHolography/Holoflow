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

#include "holonp/meshgrid.hh"

#include <cuComplex.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace holonp {

void to_json(nlohmann::json &j, const MeshgridSettings &s) {
  j = nlohmann::json{
      {"indexing", (s.indexing == MeshgridIndexing::XY ? "xy" : "ij")},
  };

  if (s.copy) {
    j["copy"] = *s.copy;
  }
  if (s.sparse) {
    j["sparse"] = *s.sparse;
  }
}

void from_json(const nlohmann::json &j, MeshgridSettings &s) {
  if (j.contains("indexing")) {
    const auto idx = j.at("indexing").get<std::string>();
    if (idx == "xy") {
      s.indexing = MeshgridIndexing::XY;
    } else if (idx == "ij") {
      s.indexing = MeshgridIndexing::IJ;
    } else {
      throw std::invalid_argument("MeshgridSettings: indexing must be 'xy' or 'ij'");
    }
  } else {
    s.indexing = MeshgridIndexing::XY;
  }

  if (j.contains("copy")) {
    s.copy = j.at("copy").get<bool>();
  } else {
    s.copy = std::nullopt;
  }

  if (j.contains("sparse")) {
    s.sparse = j.at("sparse").get<bool>();
  } else {
    s.sparse = std::nullopt;
  }
}

namespace {

static constexpr int kMaxRank = 8;

inline std::vector<std::int64_t> make_c_strides(const std::vector<std::int64_t> &dims) {
  std::vector<std::int64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

inline std::int64_t checked_numel(const std::vector<std::int64_t> &dims) {
  std::int64_t n = 1;
  for (auto d : dims) {
    if (d < 0)
      throw std::invalid_argument("Meshgrid: negative dimension");
    if (d == 0)
      return 0;
    if (n > (std::numeric_limits<std::int64_t>::max() / d)) {
      throw std::invalid_argument("Meshgrid: output too large");
    }
    n *= d;
  }
  return n;
}

inline int axis_for_input(MeshgridIndexing indexing, int rank, int input_i) {
  if (indexing == MeshgridIndexing::IJ || rank < 2) {
    return input_i;
  }
  if (input_i == 0)
    return 1;
  if (input_i == 1)
    return 0;
  return input_i;
}

inline std::vector<std::int64_t> output_dims(MeshgridIndexing                 indexing,
                                             const std::vector<std::int64_t> &lens) {
  auto dims = lens;
  if (indexing == MeshgridIndexing::XY && dims.size() >= 2) {
    std::swap(dims[0], dims[1]);
  }
  return dims;
}

template <class T>
__global__ void meshgrid_fill_kernel(T *out, std::int64_t total, const T *in, std::int64_t in_len,
                                     int axis, const std::int64_t *__restrict__ strides,
                                     const std::int64_t *__restrict__ dims) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  const std::int64_t coord = (idx / strides[axis]) % dims[axis];
  if (coord >= 0 && coord < in_len) {
    out[idx] = in[coord];
  }
}

__global__ void meshgrid_fill_kernel_cf32(cuFloatComplex *out, std::int64_t total,
                                          const cuFloatComplex *in, std::int64_t in_len, int axis,
                                          const std::int64_t *__restrict__ strides,
                                          const std::int64_t *__restrict__ dims) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  const std::int64_t coord = (idx / strides[axis]) % dims[axis];
  if (coord >= 0 && coord < in_len) {
    out[idx] = in[coord];
  }
}

struct DeviceDims {
  std::int64_t *d_dims    = nullptr;
  std::int64_t *d_strides = nullptr;
};

inline DeviceDims upload_dims_and_strides(const std::vector<std::int64_t> &dims,
                                          const std::vector<std::int64_t> &strides,
                                          cudaStream_t                     stream) {
  DeviceDims dd{};

  const auto bytes = static_cast<size_t>(dims.size() * sizeof(std::int64_t));
  CUDA_CHECK(cudaMallocAsync(&dd.d_dims, bytes, stream));
  CUDA_CHECK(cudaMallocAsync(&dd.d_strides, bytes, stream));

  CUDA_CHECK(cudaMemcpyAsync(dd.d_dims, dims.data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dd.d_strides, strides.data(), bytes, cudaMemcpyHostToDevice, stream));

  return dd;
}

inline void free_dims_and_strides(DeviceDims &dd, cudaStream_t stream) {
  if (dd.d_dims)
    CUDA_CHECK(cudaFreeAsync(dd.d_dims, stream));
  if (dd.d_strides)
    CUDA_CHECK(cudaFreeAsync(dd.d_strides, stream));
  dd = {};
}

} // namespace

Meshgrid::Meshgrid(const MeshgridSettings &settings, cudaStream_t stream)
    : settings_(settings), stream_(stream) {}

holoflow::core::OpResult Meshgrid::execute(holoflow::core::SyncCtx &ctx) {
  const int n = static_cast<int>(ctx.inputs.size());
  if (n <= 0 || static_cast<int>(ctx.outputs.size()) != n) {
    logger()->error("[Meshgrid::execute] expected N inputs and N outputs");
    std::abort();
  }

  const auto &odesc0 = ctx.outputs[0].desc;
  const auto  total  = static_cast<std::int64_t>(odesc0.num_elements());
  if (total <= 0) {
    return holoflow::core::OpResult::Ok;
  }

  std::vector<std::int64_t> dims;
  dims.reserve(odesc0.shape.size());
  for (auto d : odesc0.shape) {
    dims.push_back(static_cast<std::int64_t>(d));
  }

  const int rank = static_cast<int>(dims.size());
  if (rank > kMaxRank) {
    logger()->error("[Meshgrid::execute] rank too large (>{})", kMaxRank);
    std::abort();
  }

  const auto strides = make_c_strides(dims);
  auto       dd      = upload_dims_and_strides(dims, strides, stream_);

  constexpr int block_size = 256;
  const int     grid_size  = static_cast<int>((total + block_size - 1) / block_size);

  const auto dtype = odesc0.dtype;

  for (int k = 0; k < n; ++k) {
    auto [idata, idesc] = ctx.inputs[k];
    auto [odata, odesc] = ctx.outputs[k];

    const auto in_len = static_cast<std::int64_t>(idesc.num_elements());
    const int  axis   = axis_for_input(settings_.indexing, rank, k);

    switch (dtype) {
    case holoflow::core::DType::U8: {
      auto *out = reinterpret_cast<std::uint8_t *>(odata);
      auto *in  = reinterpret_cast<const std::uint8_t *>(idata);
      meshgrid_fill_kernel<<<grid_size, block_size, 0, stream_>>>(out, total, in, in_len, axis,
                                                                  dd.d_strides, dd.d_dims);
      break;
    }
    case holoflow::core::DType::U16: {
      auto *out = reinterpret_cast<std::uint16_t *>(odata);
      auto *in  = reinterpret_cast<const std::uint16_t *>(idata);
      meshgrid_fill_kernel<<<grid_size, block_size, 0, stream_>>>(out, total, in, in_len, axis,
                                                                  dd.d_strides, dd.d_dims);
      break;
    }
    case holoflow::core::DType::F32: {
      auto *out = reinterpret_cast<float *>(odata);
      auto *in  = reinterpret_cast<const float *>(idata);
      meshgrid_fill_kernel<<<grid_size, block_size, 0, stream_>>>(out, total, in, in_len, axis,
                                                                  dd.d_strides, dd.d_dims);
      break;
    }
    case holoflow::core::DType::CF32: {
      auto *out = reinterpret_cast<cuFloatComplex *>(odata);
      auto *in  = reinterpret_cast<const cuFloatComplex *>(idata);
      meshgrid_fill_kernel_cf32<<<grid_size, block_size, 0, stream_>>>(out, total, in, in_len, axis,
                                                                       dd.d_strides, dd.d_dims);
      break;
    }
    default:
      free_dims_and_strides(dd, stream_);
      logger()->error("[Meshgrid::execute] unsupported dtype");
      std::abort();
    }

    CUDA_CHECK(cudaGetLastError());
  }

  free_dims_and_strides(dd, stream_);
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
MeshgridFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition)
      throw std::invalid_argument("MeshgridFactory inference error: " + msg);
  };

  const auto settings = jsettings.get<MeshgridSettings>();

  // Apply NumPy defaults if missing
  const bool copy   = settings.copy.value_or(true);
  const bool sparse = settings.sparse.value_or(false);

  // Supported subset (enforced at infer time)
  check(copy == true, "copy=false is not supported (only copy=true)");
  check(sparse == false, "sparse=true is not supported (only sparse=false)");

  check(!input_descs.empty(), "expected at least 1 input");
  check(input_descs.size() <= static_cast<size_t>(kMaxRank), "too many inputs (rank too large)");

  const auto dtype  = input_descs[0].dtype;
  const auto memloc = input_descs[0].mem_loc;

  check(memloc == holoflow::core::MemLoc::Device,
        "only Device inputs/outputs are supported (for now)");

  std::vector<std::int64_t> lens;
  lens.reserve(input_descs.size());

  for (size_t i = 0; i < input_descs.size(); ++i) {
    const auto &td = input_descs[i];
    check(td.mem_loc == memloc, "all inputs must share the same mem_loc");
    check(td.dtype == dtype, "all inputs must share the same dtype");
    check(td.shape.size() == 1, "all inputs must be 1-D vectors");
    lens.push_back(static_cast<std::int64_t>(td.shape[0]));
  }

  const auto odims = output_dims(settings.indexing, lens);
  (void)checked_numel(odims);

  std::vector<size_t> shape;
  shape.reserve(odims.size());
  for (auto d : odims)
    shape.push_back(static_cast<size_t>(d));

  std::vector<holoflow::core::TDesc> out_descs;
  out_descs.reserve(input_descs.size());
  for (size_t i = 0; i < input_descs.size(); ++i) {
    out_descs.push_back(holoflow::core::TDesc(shape, dtype, memloc));
  }

  return holoflow::core::InferResult{
      .input_descs   = {input_descs.begin(), input_descs.end()},
      .output_descs  = std::move(out_descs),
      .in_place      = {},
      .owned_inputs  = std::vector<bool>(input_descs.size(), false),
      .owned_outputs = std::vector<bool>(input_descs.size(), false),
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
MeshgridFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)this->infer(input_descs, jsettings);
  const auto settings = jsettings.get<MeshgridSettings>();
  return std::unique_ptr<holoflow::core::ISyncTask>(new Meshgrid(settings, ctx.stream));
}

} // namespace holonp
