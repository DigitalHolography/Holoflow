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

#include "holotask/syncs/pct_clip.hh"

#include <cub/cub.cuh>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "logger.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const PctClipSettings::Ellipse &e) {
  j = nlohmann::json{
      {"cx", e.cx}, {"cy", e.cy}, {"rx", e.rx}, {"ry", e.ry}, {"angle", e.angle},
  };
}

void from_json(const nlohmann::json &j, PctClipSettings::Ellipse &e) {
  j.at("cx").get_to(e.cx);
  j.at("cy").get_to(e.cy);
  j.at("rx").get_to(e.rx);
  j.at("ry").get_to(e.ry);
  j.at("angle").get_to(e.angle);
}

void to_json(nlohmann::json &j, const PctClipSettings &s) {
  j = nlohmann::json{
      {"min_pct", s.min_pct},
      {"max_pct", s.max_pct},
      {"roi", s.roi},
  };
}

void from_json(const nlohmann::json &j, PctClipSettings &s) {
  j.at("min_pct").get_to(s.min_pct);
  j.at("max_pct").get_to(s.max_pct);
  j.at("roi").get_to(s.roi);
}

namespace {

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[PctClipFactory::infer] error: {}", msg);
    throw std::invalid_argument("PctClipFactory inference error: " + msg);
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

__global__ void gather_roi_kernel(float *roi_values, const float *idata, const int *spatial_indices,
                                  int spatial_roi_count, int plane_size, int roi_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= roi_count) {
    return;
  }

  const int plane         = idx / spatial_roi_count;
  const int spatial_index = spatial_indices[idx % spatial_roi_count];
  roi_values[idx]         = idata[plane * plane_size + spatial_index];
}

__global__ void load_percentile_bounds_kernel(float *bounds, const float *sorted_values,
                                              int min_idx, int max_idx) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    bounds[0] = sorted_values[min_idx];
    bounds[1] = sorted_values[max_idx];
  }
}

__global__ void clip_kernel(float *odata, const float *idata, int n, const float *bounds) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float val  = idata[idx];
  val        = fminf(fmaxf(val, bounds[0]), bounds[1]);
  odata[idx] = val;
}

std::vector<int> make_spatial_roi_indices(int width, int height,
                                          const PctClipSettings::Ellipse &roi) {
  constexpr float  pi = 3.14159265358979323846f;
  const float      th = roi.angle * (pi / 180.0f);
  const float      c  = std::cos(th);
  const float      s  = std::sin(th);
  std::vector<int> indices;
  indices.reserve(static_cast<size_t>(width) * height);

  for (int y = 0; y < height; ++y) {
    const float yn = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    for (int x = 0; x < width; ++x) {
      const float xn = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
      const float dx = xn - roi.cx;
      const float dy = yn - roi.cy;
      const float xr = c * dx + s * dy;
      const float yr = -s * dx + c * dy;
      if ((xr * xr) / (roi.rx * roi.rx) + (yr * yr) / (roi.ry * roi.ry) <= 1.0f) {
        indices.push_back(y * width + x);
      }
    }
  }
  return indices;
}

class PctClipCudaGraph {
public:
  using Addresses = std::array<const void *, 2>;

  PctClipCudaGraph() = default;
  ~PctClipCudaGraph() noexcept { reset(); }

  PctClipCudaGraph(const PctClipCudaGraph &)            = delete;
  PctClipCudaGraph &operator=(const PctClipCudaGraph &) = delete;

  [[nodiscard]] bool ready() const noexcept { return executable_ != nullptr; }
  [[nodiscard]] bool matches(const Addresses &addresses) const noexcept {
    return ready() && addresses_ == addresses;
  }
  [[nodiscard]] bool matches_output(const Addresses &addresses) const noexcept {
    return ready() && addresses_[1] == addresses[1];
  }

  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

  void reset() noexcept {
    if (executable_ != nullptr) {
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
      executable_ = nullptr;
    }
    if (graph_ != nullptr) {
      CUDA_CHECK_NT(cudaGraphDestroy(graph_));
      graph_ = nullptr;
    }
    addresses_   = {};
    gather_node_ = nullptr;
    clip_node_   = nullptr;
  }

  template <typename Enqueue>
  bool capture(cudaStream_t stream, const Addresses &addresses, Enqueue &&enqueue) {
    cudaGraph_t graph = capture_graph(stream, std::forward<Enqueue>(enqueue));
    if (graph == nullptr) {
      return false;
    }
    cudaGraphExec_t executable = nullptr;
    try {
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
      const auto nodes = find_update_nodes(graph);
      reset();
      executable_  = executable;
      graph_       = graph;
      addresses_   = addresses;
      gather_node_ = nodes[0];
      clip_node_   = nodes[1];
      return true;
    } catch (...) {
      if (executable != nullptr) {
        CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
      }
      if (graph != nullptr) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
      }
      throw;
    }
  }

  void update_input(const cudaKernelNodeParams &gather_params,
                    const cudaKernelNodeParams &clip_params, const Addresses &addresses) {
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable_, gather_node_, &gather_params));
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable_, clip_node_, &clip_params));
    addresses_ = addresses;
  }

  template <typename Enqueue>
  bool update(cudaStream_t stream, const Addresses &addresses, Enqueue &&enqueue) {
    cudaGraph_t candidate = capture_graph(stream, std::forward<Enqueue>(enqueue));
    if (candidate == nullptr) {
      return false;
    }

    cudaGraphExecUpdateResultInfo update_info{};
    const auto update_error = cudaGraphExecUpdate(executable_, candidate, &update_info);
    if (update_error == cudaSuccess) {
      CUDA_CHECK_NT(cudaGraphDestroy(candidate));
      addresses_ = addresses;
      return true;
    }
    if (update_error != cudaErrorGraphExecUpdateFailure) {
      CUDA_CHECK_NT(cudaGraphDestroy(candidate));
      CUDA_CHECK(update_error);
    }

    cudaGraphExec_t executable = nullptr;
    try {
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, candidate, 0));
      const auto nodes = find_update_nodes(candidate);
      reset();
      executable_  = executable;
      graph_       = candidate;
      addresses_   = addresses;
      gather_node_ = nodes[0];
      clip_node_   = nodes[1];
      return true;
    } catch (...) {
      if (executable != nullptr) {
        CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
      }
      CUDA_CHECK_NT(cudaGraphDestroy(candidate));
      throw;
    }
  }

private:
  template <typename Enqueue>
  static cudaGraph_t capture_graph(cudaStream_t stream, Enqueue &&enqueue) {
    cudaGraph_t graph     = nullptr;
    bool        capturing = false;
    try {
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      capturing           = true;
      const bool complete = std::forward<Enqueue>(enqueue)();
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      capturing = false;
      if (!complete) {
        if (graph != nullptr) {
          CUDA_CHECK_NT(cudaGraphDestroy(graph));
        }
        return nullptr;
      }
      return graph;
    } catch (...) {
      if (capturing) {
        cudaGraph_t discarded = nullptr;
        (void)cudaStreamEndCapture(stream, &discarded);
        if (discarded != nullptr) {
          CUDA_CHECK_NT(cudaGraphDestroy(discarded));
        }
      } else if (graph != nullptr) {
        CUDA_CHECK_NT(cudaGraphDestroy(graph));
      }
      throw;
    }
  }

  static std::array<cudaGraphNode_t, 2> find_update_nodes(cudaGraph_t graph) {
    size_t node_count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));

    cudaGraphNode_t gather_node = nullptr;
    cudaGraphNode_t clip_node   = nullptr;
    for (const auto node : nodes) {
      cudaGraphNodeType type{};
      CUDA_CHECK(cudaGraphNodeGetType(node, &type));
      if (type != cudaGraphNodeTypeKernel) {
        continue;
      }

      cudaKernelNodeParams params{};
      CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
      if (params.func == reinterpret_cast<void *>(gather_roi_kernel)) {
        HOLOVIBES_CHECK(gather_node == nullptr, "Multiple gather kernels in percentile clip graph");
        gather_node = node;
      } else if (params.func == reinterpret_cast<void *>(clip_kernel)) {
        HOLOVIBES_CHECK(clip_node == nullptr, "Multiple clip kernels in percentile clip graph");
        clip_node = node;
      }
    }

    HOLOVIBES_CHECK(gather_node != nullptr, "Missing gather kernel in percentile clip graph");
    HOLOVIBES_CHECK(clip_node != nullptr, "Missing clip kernel in percentile clip graph");
    return {gather_node, clip_node};
  }

  // Kernel-node executable updates require their originating graph nodes to stay alive.
  cudaGraph_t     graph_       = nullptr;
  cudaGraphExec_t executable_  = nullptr;
  Addresses       addresses_   = {};
  cudaGraphNode_t gather_node_ = nullptr;
  cudaGraphNode_t clip_node_   = nullptr;
};

// -------------------------------------------------------------------------------------------------
// PctClip task implementation
// -------------------------------------------------------------------------------------------------

class PctClip : public holoflow::core::ISyncTask {
public:
  PctClip(PctClipSettings settings, holoflow::core::TDesc idesc, int spatial_roi_count,
          int roi_count, int min_idx, int max_idx, DevPtr<int> &&d_spatial_indices,
          DevPtr<float> &&d_bounds, size_t sort_tmp_bytes, DevPtr<uint8_t> &&d_sort_tmp,
          DevPtr<float> &&d_roi, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), roi_count_(roi_count),
        spatial_roi_count_(spatial_roi_count), min_idx_(min_idx), max_idx_(max_idx),
        d_spatial_indices_(std::move(d_spatial_indices)), d_bounds_(std::move(d_bounds)),
        sort_tmp_bytes_(sort_tmp_bytes), d_sort_tmp_(std::move(d_sort_tmp)),
        d_roi_(std::move(d_roi)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const PctClipCudaGraph::Addresses addresses{ctx.inputs[0].data(), ctx.outputs[0].data()};
    if (stream_ == nullptr) {
      return enqueue(ctx);
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream_, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      return enqueue(ctx);
    }

    if (graph_.matches(addresses)) {
      graph_.launch(stream_);
      return holoflow::core::OpResult::Ok;
    }

    if (graph_.matches_output(addresses)) {
      try {
        // Only these two custom kernels consume the rotating sliding-average input. CUB's
        // captured nodes use stable scratch and output addresses and do not need updating.
        update_graph_input(ctx, addresses);
        graph_.launch(stream_);
        return holoflow::core::OpResult::Ok;
      } catch (const std::exception &error) {
        logger()->warn("[PctClip] CUDA Graph input update failed; recapturing: {}", error.what());
        graph_.reset();
      }
    } else if (graph_.ready() && graph_capture_enabled_) {
      try {
        const bool updated = graph_.update(
            stream_, addresses, [&]() { return enqueue(ctx) == holoflow::core::OpResult::Ok; });
        if (updated) {
          graph_.launch(stream_);
          return holoflow::core::OpResult::Ok;
        }
        graph_capture_enabled_ = false;
      } catch (const std::exception &error) {
        logger()->warn("[PctClip] CUDA Graph executable update failed; recapturing: {}",
                       error.what());
        graph_.reset();
      }
    }

    const auto result = enqueue(ctx);
    if (result == holoflow::core::OpResult::Ok && graph_capture_enabled_) {
      try_capture(ctx, addresses);
    }
    return result;
  }

  void update_stream(cudaStream_t stream) {
    if (stream_ == stream) {
      return;
    }
    graph_capture_enabled_ = true;
    stream_                = stream;
  }

  const PctClipSettings       &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  void update_graph_input(holoflow::core::SyncCtx           &ctx,
                          const PctClipCudaGraph::Addresses &addresses) {
    auto *idata      = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto *odata      = reinterpret_cast<float *>(ctx.outputs[0].data());
    int   count      = static_cast<int>(ctx.inputs[0].desc.num_elements());
    int   plane_size = static_cast<int>(idesc_.shape[1] * idesc_.shape[2]);
    auto *roi_values = d_roi_.get();
    auto *indices    = d_spatial_indices_.get();
    auto *bounds     = d_bounds_.get();

    constexpr int        block_size    = 256;
    const int            roi_grid_size = (roi_count_ + block_size - 1) / block_size;
    void                *gather_args[] = {&roi_values,         &idata,      &indices,
                                          &spatial_roi_count_, &plane_size, &roi_count_};
    cudaKernelNodeParams gather_params{
        .func           = reinterpret_cast<void *>(gather_roi_kernel),
        .gridDim        = {static_cast<unsigned int>(roi_grid_size), 1, 1},
        .blockDim       = {block_size, 1, 1},
        .sharedMemBytes = 0,
        .kernelParams   = gather_args,
        .extra          = nullptr,
    };

    const int            grid_size   = (count + block_size - 1) / block_size;
    void                *clip_args[] = {&odata, &idata, &count, &bounds};
    cudaKernelNodeParams clip_params{
        .func           = reinterpret_cast<void *>(clip_kernel),
        .gridDim        = {static_cast<unsigned int>(grid_size), 1, 1},
        .blockDim       = {block_size, 1, 1},
        .sharedMemBytes = 0,
        .kernelParams   = clip_args,
        .extra          = nullptr,
    };

    graph_.update_input(gather_params, clip_params, addresses);
  }

  void try_capture(holoflow::core::SyncCtx &ctx, const PctClipCudaGraph::Addresses &addresses) {
    try {
      const bool captured = graph_.capture(
          stream_, addresses, [&]() { return enqueue(ctx) == holoflow::core::OpResult::Ok; });
      if (!captured) {
        graph_capture_enabled_ = false;
      }
    } catch (const std::exception &error) {
      graph_capture_enabled_ = false;
      logger()->warn("[PctClip] CUDA Graph capture disabled: {}", error.what());
    }
  }

  holoflow::core::OpResult enqueue(holoflow::core::SyncCtx &ctx) {
    auto     *idata      = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto     *odata      = reinterpret_cast<float *>(ctx.outputs[0].data());
    const int count      = static_cast<int>(ctx.inputs[0].desc.num_elements());
    const int plane_size = static_cast<int>(idesc_.shape[1] * idesc_.shape[2]);

    constexpr int block_size    = 256;
    const int     roi_grid_size = (roi_count_ + block_size - 1) / block_size;
    gather_roi_kernel<<<roi_grid_size, block_size, 0, stream_>>>(
        d_roi_.get(), idata, d_spatial_indices_.get(), spatial_roi_count_, plane_size, roi_count_);

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_sort_tmp_.get(), sort_tmp_bytes_, d_roi_.get(),
                                              odata, roi_count_, 0, 32, stream_));

    load_percentile_bounds_kernel<<<1, 1, 0, stream_>>>(d_bounds_.get(), odata, min_idx_, max_idx_);

    const int grid_size = (count + block_size - 1) / block_size;
    clip_kernel<<<grid_size, block_size, 0, stream_>>>(odata, idata, count, d_bounds_.get());

    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }
  PctClipSettings       settings_;
  holoflow::core::TDesc idesc_;
  int                   roi_count_;
  int                   spatial_roi_count_;
  int                   min_idx_;
  int                   max_idx_;
  DevPtr<int>           d_spatial_indices_;
  DevPtr<float>         d_bounds_;
  size_t                sort_tmp_bytes_;
  DevPtr<uint8_t>       d_sort_tmp_;
  DevPtr<float>         d_roi_;
  cudaStream_t          stream_;
  bool                  graph_capture_enabled_ = true;
  PctClipCudaGraph      graph_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// PctClipFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
PctClipFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<PctClipSettings>();

  check(settings.min_pct >= 0.0f, "min_pct must be >= 0");
  check(settings.min_pct <= 100.0f, "min_pct must be <= 100");
  check(settings.max_pct >= 0.0f, "max_pct must be >= 0");
  check(settings.max_pct <= 100.0f, "max_pct must be <= 100");
  check(settings.min_pct < settings.max_pct, "min_pct must be < max_pct");
  check(settings.roi.rx > 0.0f, "roi.rx must be > 0");
  check(settings.roi.rx <= 1.0f, "roi.rx must be <= 1");
  check(settings.roi.ry > 0.0f, "roi.ry must be > 0");
  check(settings.roi.ry <= 1.0f, "roi.ry must be <= 1");
  check(settings.roi.cx >= 0.0f, "roi.cx must be >= 0");
  check(settings.roi.cx <= 1.0f, "roi.cx must be <= 1");
  check(settings.roi.cy >= 0.0f, "roi.cy must be >= 0");
  check(settings.roi.cy <= 1.0f, "roi.cy must be <= 1");
  check(input_descs.size() == 1, "expected exactly one input");

  const auto &idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(idesc.rank() == 3, "input must be 3D");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(is_c_contiguous(idesc), "input must be C-contiguous");

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
PctClipFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  using curaii::make_unique_device_ptr;

  (void)this->infer(input_descs, jsettings);
  const auto  settings = jsettings.get<PctClipSettings>();
  const auto &idesc    = input_descs[0];

  const int  depth               = static_cast<int>(idesc.shape[0]);
  const int  height              = static_cast<int>(idesc.shape[1]);
  const int  width               = static_cast<int>(idesc.shape[2]);
  const auto spatial_roi_indices = make_spatial_roi_indices(width, height, settings.roi);
  HOLOVIBES_CHECK(!spatial_roi_indices.empty(), "No pixels in ROI");
  const int spatial_roi_count = static_cast<int>(spatial_roi_indices.size());
  const int h_roi_count       = depth * spatial_roi_count;

  const int min_idx = static_cast<int>(settings.min_pct / 100.0f * (h_roi_count - 1));
  const int max_idx = static_cast<int>(settings.max_pct / 100.0f * (h_roi_count - 1));

  auto d_spatial_indices = make_unique_device_ptr<int>(spatial_roi_count);
  CUDA_CHECK(cudaMemcpy(d_spatial_indices.get(), spatial_roi_indices.data(),
                        spatial_roi_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  auto d_bounds = make_unique_device_ptr<float>(2);

  size_t sort_tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_tmp_bytes, static_cast<float *>(nullptr),
                                            static_cast<float *>(nullptr), h_roi_count, 0, 32,
                                            ctx.stream));
  auto d_sort_tmp = make_unique_device_ptr<uint8_t>(sort_tmp_bytes);

  auto d_roi = make_unique_device_ptr<float>(h_roi_count);

  return std::make_unique<PctClip>(settings, idesc, spatial_roi_count, h_roi_count, min_idx,
                                   max_idx, std::move(d_spatial_indices), std::move(d_bounds),
                                   sort_tmp_bytes, std::move(d_sort_tmp), std::move(d_roi),
                                   ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
PctClipFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)this->infer(input_descs, jsettings);

  auto *old_pct_clip = dynamic_cast<PctClip *>(old_task.get());
  if (old_pct_clip == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_pct_clip->idesc();
  const auto  settings  = jsettings.get<PctClipSettings>();
  const bool  can_reuse =
      settings == old_pct_clip->settings() && new_idesc.shape == old_idesc.shape &&
      new_idesc.strides == old_idesc.strides && new_idesc.dtype == old_idesc.dtype &&
      new_idesc.mem_loc == old_idesc.mem_loc;

  if (can_reuse) {
    old_pct_clip->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
