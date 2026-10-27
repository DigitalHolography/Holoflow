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

#include "holotask/syncs/pca.hh"
#include "holotask/asyncs/pca_decomposition_queue.hh"

#include <cuda.h>
#include <nvJitLink.h>
#include <nvtx3/nvtx3.hpp>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "curaii/nvrtc.hh"

#include "../asyncs/reservation_queue.hh"
#include "logger.hh"

namespace holotask::syncs {

namespace {

// -------------------------------------------------------------------------------------------------
// PCA data model
// -------------------------------------------------------------------------------------------------

// PCA operates on tensors shaped [..., features, height, width]. All leading dimensions are
// flattened into independent batches.
struct PcaLayout {
  size_t feature_axis;
  int    features;
  int    height;
  int    width;
  int    samples;
  size_t batches;

  // Strides in elements between consecutive flattened batches.
  long long input_batch_stride;
  long long matrix_batch_stride;

  explicit PcaLayout(const holoflow::core::TDesc &desc) {
    const auto rank = desc.rank();

    feature_axis = rank - 3;
    features     = static_cast<int>(desc.shape.at(feature_axis));
    height       = static_cast<int>(desc.shape.at(rank - 2));
    width        = static_cast<int>(desc.shape.at(rank - 1));
    samples      = height * width;

    batches = 1;
    for (size_t axis = 0; axis < feature_axis; ++axis) {
      batches *= desc.shape.at(axis);
    }

    input_batch_stride  = static_cast<long long>(features) * samples;
    matrix_batch_stride = static_cast<long long>(features) * features;
  }

  [[nodiscard]] long long output_batch_stride(int components) const {
    return static_cast<long long>(samples) * components;
  }
};

struct PcaWorkspace {
  // The matrix buffer contains covariance matrices before HEEV and eigenvectors afterwards.
  curaii::unique_device_ptr<float> matrices;
  curaii::unique_device_ptr<float> eigenvalues;
  curaii::unique_device_ptr<int>   solver_info;

  explicit PcaWorkspace(const PcaLayout &layout)
      : matrices(curaii::make_unique_device_ptr<float>(
            layout.batches * static_cast<size_t>(layout.features) * layout.features)),
        eigenvalues(curaii::make_unique_device_ptr<float>(layout.batches * layout.features)),
        solver_info(curaii::make_unique_device_ptr<int>(layout.batches)) {}
};

// -------------------------------------------------------------------------------------------------
// CUDA Graph replay
// -------------------------------------------------------------------------------------------------

using PcaGraphKey = std::array<const void *, 2>;

template <typename Enqueue> cudaGraph_t capture_cuda_graph(cudaStream_t stream, Enqueue &&enqueue) {
  cudaGraph_t graph     = nullptr;
  bool        capturing = false;

  try {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    capturing = true;

    std::forward<Enqueue>(enqueue)();

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    capturing = false;
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

[[nodiscard]] bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

class PcaCudaGraph {
public:
  explicit PcaCudaGraph(PcaGraphKey key) : key_(key) {}

  ~PcaCudaGraph() noexcept {
    if (executable_ != nullptr) {
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable_));
    }
  }

  PcaCudaGraph(const PcaCudaGraph &)            = delete;
  PcaCudaGraph &operator=(const PcaCudaGraph &) = delete;

  [[nodiscard]] bool matches(const PcaGraphKey &key) const noexcept {
    return executable_ != nullptr && key_ == key;
  }

  void launch(cudaStream_t stream) const { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

  template <typename Enqueue> void capture(cudaStream_t stream, Enqueue &&enqueue) {
    cudaGraph_t graph = capture_cuda_graph(stream, std::forward<Enqueue>(enqueue));

    try {
      CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable_, graph, 0));
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
    } catch (...) {
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
      throw;
    }
  }

private:
  PcaGraphKey     key_{};
  cudaGraphExec_t executable_{nullptr};
};

class PcaGraphCache {
public:
  void invalidate() {
    graphs_.clear();
    capture_enabled_ = true;
  }

  void enable_capture() noexcept { capture_enabled_ = true; }

  [[nodiscard]] bool try_launch(const PcaGraphKey &key, cudaStream_t stream) {
    const auto graph = std::find_if(graphs_.begin(), graphs_.end(),
                                    [&](const PcaCudaGraph &entry) { return entry.matches(key); });

    if (graph == graphs_.end()) {
      return false;
    }

    graph->launch(stream);
    graphs_.splice(graphs_.begin(), graphs_, graph);
    return true;
  }

  template <typename Enqueue>
  void try_capture(const PcaGraphKey &key, cudaStream_t stream, Enqueue &&enqueue) {
    if (!capture_enabled_) {
      return;
    }

    bool entry_inserted = false;
    try {
      graphs_.emplace_front(key);
      entry_inserted = true;

      graphs_.front().capture(stream, std::forward<Enqueue>(enqueue));

      if (graphs_.size() > capacity) {
        graphs_.pop_back();
      }
    } catch (const std::exception &error) {
      if (entry_inserted) {
        graphs_.pop_front();
      }
      capture_enabled_ = false;
      logger()->warn("[Pca] CUDA Graph capture disabled: {}", error.what());
    }
  }

private:
  // cuBLAS and cuSOLVER graph nodes embed their buffer addresses. Cache the finite set of rotating
  // pipeline buffers instead of attempting to update opaque library nodes.
  static constexpr size_t capacity = 128;

  bool                    capture_enabled_{true};
  std::list<PcaCudaGraph> graphs_;
};

// -------------------------------------------------------------------------------------------------
// Eigensolver abstraction
// -------------------------------------------------------------------------------------------------

class Eigensolver {
public:
  virtual ~Eigensolver() = default;

  Eigensolver(const Eigensolver &)            = delete;
  Eigensolver &operator=(const Eigensolver &) = delete;

  [[nodiscard]] virtual bool is_compatible_stream(cudaStream_t stream) const = 0;

  virtual void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) = 0;

protected:
  Eigensolver() = default;
};

std::unique_ptr<Eigensolver> make_eigensolver(const PcaLayout    &layout,
                                              const PcaWorkspace &workspace, cudaStream_t stream);

void enqueue_pca_covariance(cublasHandle_t cublas, const PcaLayout &layout, const float *input,
                            float *matrices) {
  nvtx3::scoped_range range("PCA covariance");
  constexpr float     alpha = 1.0f;
  constexpr float     beta  = 0.0f;
  // Independent GEMMs preserve Split-K selection for the large sample dimension.
  for (size_t batch = 0; batch < layout.batches; ++batch) {
    const auto x = input + batch * layout.input_batch_stride;
    CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, layout.features, layout.features,
                              layout.samples, &alpha, x, CUDA_R_32F, layout.samples, x, CUDA_R_32F,
                              layout.samples, &beta, matrices + batch * layout.matrix_batch_stride,
                              CUDA_R_32F, layout.features, CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT));
  }
}

void enqueue_pca_projection(cublasHandle_t cublas, const PcaLayout &layout,
                            const PcaSettings &settings, const float *input,
                            const float *eigenvectors, float *output) {
  nvtx3::scoped_range range("PCA projection");
  constexpr float     alpha      = 1.0f;
  constexpr float     beta       = 0.0f;
  const int           components = settings.components();
  for (size_t batch = 0; batch < layout.batches; ++batch) {
    CUBLAS_CHECK(cublasGemmEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, layout.samples, components, layout.features, &alpha,
        input + batch * layout.input_batch_stride, CUDA_R_32F, layout.samples,
        eigenvectors + batch * layout.matrix_batch_stride +
            static_cast<long long>(settings.begin) * layout.features,
        CUDA_R_32F, layout.features, &beta, output + batch * layout.output_batch_stride(components),
        CUDA_R_32F, layout.samples, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
  }
}

// -------------------------------------------------------------------------------------------------
// PCA task
// -------------------------------------------------------------------------------------------------

class PcaTask final : public holoflow::core::ISyncTask {
public:
  PcaTask(const PcaSettings &settings, const holoflow::core::TDesc &input_desc,
          const holoflow::core::SyncCreateCtx &ctx)
      : settings_(settings), input_desc_(input_desc), layout_(input_desc), stream_(ctx.stream),
        workspace_(layout_) {
    CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));
    eigensolver_ = make_eigensolver(layout_, workspace_, stream_);
  }

  [[nodiscard]] bool can_reuse(const holoflow::core::TDesc &input_desc, cudaStream_t stream) const {
    return input_desc.shape == input_desc_.shape && input_desc.strides == input_desc_.strides &&
           input_desc.dtype == input_desc_.dtype && input_desc.mem_loc == input_desc_.mem_loc &&
           eigensolver_->is_compatible_stream(stream);
  }

  void reconfigure(const PcaSettings &settings, cudaStream_t stream) {
    if (settings_ != settings) {
      settings_ = settings;
      graph_cache_.invalidate();
    }

    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));

      // Graph executables are not tied to the stream on which they were captured. Reusing them is
      // valid here because can_reuse() already guarantees that the CUDA context is unchanged.
      graph_cache_.enable_capture();
    }
  }

  bool same_settings(const PcaSettings &settings) const { return settings_ == settings; }

  bool supports_cuda_graph() const noexcept override { return true; }

  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs, &cancelled, nullptr, nullptr};
    (void)enqueue(execution);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range range("PCA Sync Task");

    if (stream_ == nullptr || stream_is_capturing(stream_)) {
      return enqueue(ctx);
    }

    const PcaGraphKey key{ctx.inputs[0].data(), ctx.outputs[0].data()};
    if (graph_cache_.try_launch(key, stream_)) {
      return holoflow::core::OpResult::Ok;
    }

    const auto result = enqueue(ctx);
    if (result == holoflow::core::OpResult::Ok) {
      graph_cache_.try_capture(key, stream_, [&]() { (void)enqueue(ctx); });
    }
    return result;
  }

private:
  holoflow::core::OpResult enqueue(holoflow::core::SyncCtx &ctx) {
    auto *input  = reinterpret_cast<float *>(ctx.inputs[0].data());
    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());

    enqueue_covariance(input);
    enqueue_eigendecomposition();
    enqueue_projection(input, output);

    return holoflow::core::OpResult::Ok;
  }

  void enqueue_covariance(const float *input) {
    enqueue_pca_covariance(cublas_.get(), layout_, input, workspace_.matrices.get());
  }

  void enqueue_eigendecomposition() {
    nvtx3::scoped_range range("PCA eigendecomposition");

    eigensolver_->solve(workspace_.matrices.get(), workspace_.eigenvalues.get(),
                        workspace_.solver_info.get(), stream_);
  }

  void enqueue_projection(const float *input, float *output) {
    enqueue_pca_projection(cublas_.get(), layout_, settings_, input, workspace_.matrices.get(),
                           output);
  }

  PcaSettings           settings_;
  holoflow::core::TDesc input_desc_;
  PcaLayout             layout_;
  cudaStream_t          stream_;

  curaii::CublasHandle         cublas_;
  PcaWorkspace                 workspace_;
  std::unique_ptr<Eigensolver> eigensolver_;
  PcaGraphCache                graph_cache_;
};

// -------------------------------------------------------------------------------------------------
// Pipelined PCA
// -------------------------------------------------------------------------------------------------

class PcaProjectionTask final : public holoflow::core::ISyncTask {
public:
  PcaProjectionTask(const PcaSettings &settings, const holoflow::core::TDesc &input,
                    cudaStream_t stream)
      : settings_(settings), layout_(input) {
    CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream));
  }

  bool supports_cuda_graph() const noexcept override { return true; }

  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    enqueue(ctx.inputs, ctx.outputs);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    enqueue(ctx.inputs, ctx.outputs);
    return holoflow::core::OpResult::Ok;
  }

private:
  void enqueue(std::span<holoflow::core::TView> inputs, std::span<holoflow::core::TView> outputs) {
    enqueue_pca_projection(cublas_.get(), layout_, settings_,
                           reinterpret_cast<const float *>(inputs[0].data()),
                           reinterpret_cast<const float *>(inputs[1].data()),
                           reinterpret_cast<float *>(outputs[0].data()));
  }

  PcaSettings          settings_;
  PcaLayout            layout_;
  curaii::CublasHandle cublas_;
};

class PcaEvent {
public:
  PcaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
  ~PcaEvent() { CUDA_CHECK_NT(cudaEventDestroy(event_)); }
  PcaEvent(const PcaEvent &)              = delete;
  PcaEvent   &operator=(const PcaEvent &) = delete;
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

struct PcaQueueSlot {
  PcaQueueSlot(const PcaLayout &layout, size_t elements)
      : input(curaii::make_unique_device_ptr<float>(elements)), workspace(layout) {}

  curaii::unique_device_ptr<float> input;
  PcaWorkspace                     workspace;
  PcaEvent                         covariance_done, solver_done;
  PcaGraphCache                    covariance_graph;
};

// The queue owns X through projection. Producer and consumer use independent FIFO cursors;
// reservations consume capacity immediately, and only completed consumption recycles a slot.
// The solver/JIT implementation stays private to this translation unit for both PCA factories.
class PcaDecompositionQueue final : public holoflow::core::IAsyncTask {
public:
  PcaDecompositionQueue(const holotask::asyncs::PcaDecompositionQueueSettings &settings,
                        const holoflow::core::InferResult &infer, cudaStream_t producer_stream)
      : settings_(settings), input_desc_(infer.input_descs[0]), output_descs_(infer.output_descs),
        layout_(output_descs_[0]), producer_stream_(producer_stream),
        solver_stream_(cudaStreamNonBlocking), window_elements_(input_desc_.num_elements()) {
    try {
      CUBLAS_CHECK(cublasSetStream(cublas_.get(), producer_stream_));
      for (size_t i = 0; i < settings_.slot_count; ++i)
        slots_.push_back(std::make_unique<PcaQueueSlot>(layout_, output_descs_[0].num_elements()));
      eigensolver_ = make_eigensolver(layout_, slots_[0]->workspace, solver_stream_.get());
      for (auto &slot : slots_) {
        // Warm cuBLAS before capture, outside the steady-state producer path.
        CUDA_CHECK(
            cudaMemsetAsync(slot->input.get(), 0, output_descs_[0].num_bytes(), producer_stream_));
        const auto enqueue = [&] {
          enqueue_pca_covariance(cublas_.get(), layout_, slot->input.get(),
                                 slot->workspace.matrices.get());
        };
        enqueue();
        slot->covariance_graph.try_capture({slot->input.get(), slot->workspace.matrices.get()},
                                           producer_stream_, enqueue);
      }
      CUDA_CHECK(cudaStreamSynchronize(producer_stream_));
    } catch (...) {
      // A throwing constructor still has to drain work before its members are destroyed.
      CUDA_CHECK_NT(cudaStreamSynchronize(producer_stream_));
      CUDA_CHECK_NT(cudaStreamSynchronize(solver_stream_.get()));
      throw;
    }
  }

  ~PcaDecompositionQueue() override {
    // The runtime drains section streams. This private stream can still own unpopped groups.
    CUDA_CHECK_NT(cudaStreamSynchronize(solver_stream_.get()));
  }

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    check_input(index);
    if (!writes_.empty())
      throw std::logic_error("PCA queue has outstanding write reservations");
    const auto write = written_.load(std::memory_order_relaxed);
    if (!can_write(write))
      return std::nullopt;
    bind_input(write);
    return holoflow::core::TView{input_desc_, &storage_access().owned_input_storage(0)};
  }

  holoflow::core::OpResult reserve_push(holoflow::core::AsyncPushCtx &ctx,
                                        holoflow::core::AsyncTicket  &ticket) override {
    using holoflow::core::OpResult;
    if (ctx.cancelled && ctx.cancelled->load())
      return OpResult::Cancelled;
    if (writes_.full())
      return OpResult::NotReady;
    const auto write =
        writes_.empty() ? written_.load(std::memory_order_relaxed) : writes_.back().next;
    if (!can_write(write))
      return OpResult::NotReady;
    bind_input(write);
    ticket = writes_.push({write + 1});
    return OpResult::Ok;
  }

  void enqueue_push(holoflow::core::AsyncTicket ticket) override {
    auto &write = writes_.get(ticket);
    if (write.submitted)
      throw std::logic_error("PCA queue write already submitted");
    // Once submitted, cancellation must not make this work replayable.
    write.submitted = true;
    if (write.next % settings_.batch_size != 0)
      return;
    auto             &slot = slot_for(write.next - 1);
    const PcaGraphKey key{slot.input.get(), slot.workspace.matrices.get()};
    if (!slot.covariance_graph.try_launch(key, producer_stream_))
      enqueue_pca_covariance(cublas_.get(), layout_, slot.input.get(),
                             slot.workspace.matrices.get());
    CUDA_CHECK(cudaEventRecord(slot.covariance_done.get(), producer_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(solver_stream_.get(), slot.covariance_done.get(), 0));
    {
      nvtx3::scoped_range range("PCA eigendecomposition");
      eigensolver_->solve(slot.workspace.matrices.get(), slot.workspace.eigenvalues.get(),
                          slot.workspace.solver_info.get(), solver_stream_.get());
    }
    CUDA_CHECK(cudaEventRecord(slot.solver_done.get(), solver_stream_.get()));
  }

  void commit_push(holoflow::core::AsyncTicket ticket) override {
    if (!writes_.get(ticket).submitted)
      throw std::logic_error("PCA queue write not submitted");
    written_.store(writes_.pop(ticket).next, std::memory_order_release);
  }

  void cancel_push(holoflow::core::AsyncTicket ticket) override {
    if (writes_.get(ticket).submitted)
      throw std::logic_error("Cannot cancel submitted PCA queue write");
    writes_.cancel(ticket);
  }

  holoflow::core::OpResult try_push(holoflow::core::AsyncPushCtx &ctx) override {
    if (!writes_.empty())
      throw std::logic_error("PCA queue has outstanding write reservations");
    holoflow::core::AsyncTicket ticket;
    const auto                  result = reserve_push(ctx, ticket);
    if (result != holoflow::core::OpResult::Ok)
      return result;
    enqueue_push(ticket);
    CUDA_CHECK(cudaStreamSynchronize(producer_stream_));
    commit_push(ticket);
    storage_access().owned_input_storage(0).ptr = nullptr;
    return holoflow::core::OpResult::Ok;
  }

  holoflow::core::OpResult reserve_pop(holoflow::core::AsyncPopCtx &ctx,
                                       holoflow::core::AsyncTicket &ticket) override {
    using holoflow::core::OpResult;
    if (legacy_read_)
      throw std::logic_error("PCA queue legacy output still held");
    if (ctx.cancelled && ctx.cancelled->load())
      return OpResult::Cancelled;
    if (reads_.full())
      return OpResult::NotReady;
    const auto read = reads_.empty() ? read_.load(std::memory_order_relaxed) : reads_.back();
    if (written_.load(std::memory_order_acquire) - read < settings_.batch_size)
      return OpResult::NotReady;
    auto &slot = slot_for(read);
    // The acquire above also guarantees this generation's event has been recorded. Querying
    // a never-recorded event would otherwise report success and expose an incomplete group.
    const auto status = cudaEventQuery(slot.solver_done.get());
    if (status == cudaErrorNotReady)
      return OpResult::NotReady;
    CUDA_CHECK(status);
    storage_access().owned_output_storage(0).ptr = reinterpret_cast<std::byte *>(slot.input.get());
    storage_access().owned_output_storage(1).ptr =
        reinterpret_cast<std::byte *>(slot.workspace.matrices.get());
    for (size_t port = 0; port < 2; ++port)
      ctx.outputs[port] = {output_descs_[port], &storage_access().owned_output_storage(port)};
    ticket = reads_.push(read + settings_.batch_size);
    return OpResult::Ok;
  }

  void release_pop(holoflow::core::AsyncTicket ticket) override {
    read_.store(reads_.pop(ticket), std::memory_order_release);
  }

  void cancel_pop(holoflow::core::AsyncTicket ticket) override { reads_.cancel(ticket); }

  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &ctx) override {
    if (!reads_.empty())
      throw std::logic_error("PCA queue has outstanding read reservations");
    holoflow::core::AsyncTicket ticket;
    const auto                  result = reserve_pop(ctx, ticket);
    if (result == holoflow::core::OpResult::Ok) {
      legacy_read_ = ticket;
      released_    = {false, false};
    }
    return result;
  }

  void release_output(int index) override {
    check_output(index);
    if (!legacy_read_ || released_[index])
      throw std::logic_error("PCA queue output is not held");
    storage_access().owned_output_storage(index).ptr = nullptr;
    released_[index]                                 = true;
    if (released_[0] && released_[1]) {
      release_pop(*legacy_read_);
      legacy_read_.reset();
    }
  }

  std::optional<std::vector<std::byte *>> owned_input_pointers(size_t index) const override {
    check_input(index);
    std::vector<std::byte *> result;
    for (const auto &slot : slots_)
      for (size_t window = 0; window < settings_.batch_size; ++window)
        result.push_back(
            reinterpret_cast<std::byte *>(slot->input.get() + window * window_elements_));
    return result;
  }

  std::optional<std::vector<std::byte *>> owned_output_pointers(size_t index) const override {
    check_output(index);
    std::vector<std::byte *> result;
    for (const auto &slot : slots_)
      result.push_back(reinterpret_cast<std::byte *>(index == 0 ? slot->input.get()
                                                                : slot->workspace.matrices.get()));
    return result;
  }

  std::optional<holoflow::core::PointerSequence>
  owned_input_pointer_sequence(size_t index) const override {
    check_input(index);
    return sequence(written_.load(std::memory_order_relaxed), capacity());
  }

  std::optional<holoflow::core::PointerSequence>
  owned_output_pointer_sequence(size_t index) const override {
    check_output(index);
    return sequence(read_.load(std::memory_order_relaxed) / settings_.batch_size,
                    settings_.slot_count);
  }

private:
  static void check_input(size_t index) {
    if (index != 0)
      throw std::out_of_range("PCA queue input port");
  }
  static void check_output(size_t index) {
    if (index > 1)
      throw std::out_of_range("PCA queue output port");
  }
  static holoflow::core::PointerSequence sequence(size_t phase, size_t count) {
    holoflow::core::PointerSequence result;
    for (size_t i = 0; i < count; ++i)
      result.cycle.push_back((phase + i) % count);
    return result;
  }
  size_t capacity() const { return settings_.batch_size * settings_.slot_count; }
  bool   can_write(size_t write) const {
    return write - read_.load(std::memory_order_acquire) < capacity();
  }
  PcaQueueSlot &slot_for(size_t window) const {
    return *slots_[(window / settings_.batch_size) % settings_.slot_count];
  }
  void bind_input(size_t write) {
    storage_access().owned_input_storage(0).ptr = reinterpret_cast<std::byte *>(
        slot_for(write).input.get() + (write % settings_.batch_size) * window_elements_);
  }

  struct PendingWrite {
    size_t next;
    bool   submitted = false;
  };
  holotask::asyncs::PcaDecompositionQueueSettings settings_;
  holoflow::core::TDesc                           input_desc_;
  std::vector<holoflow::core::TDesc>              output_descs_;
  PcaLayout                                       layout_;
  cudaStream_t                                    producer_stream_;
  curaii::CudaStream                              solver_stream_;
  size_t                                          window_elements_;
  curaii::CublasHandle                            cublas_;
  std::vector<std::unique_ptr<PcaQueueSlot>>      slots_;
  std::unique_ptr<Eigensolver>                    eigensolver_;
  alignas(64) std::atomic<size_t> written_{0};
  alignas(64) std::atomic<size_t> read_{0};
  holotask::asyncs::detail::ReservationQueue<PendingWrite> writes_;
  holotask::asyncs::detail::ReservationQueue<size_t>       reads_;
  std::optional<holoflow::core::AsyncTicket>               legacy_read_;
  std::array<bool, 2>                                      released_{false, false};
};

// -------------------------------------------------------------------------------------------------
// CUDA Driver API support
// -------------------------------------------------------------------------------------------------

std::string driver_error_message(CUresult result, const char *expression, const char *file,
                                 int line) {
  const char *name    = nullptr;
  const char *message = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &message);

  return std::format("CUDA Driver error: {} ({})\n  expression : {}\n  location   : {}:{}",
                     message != nullptr ? message : "unknown",
                     name != nullptr ? name : std::to_string(static_cast<int>(result)), expression,
                     file, line);
}

void driver_check(CUresult result, const char *expression, const char *file, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }

  const auto message = driver_error_message(result, expression, file, line);
  logger()->error("{}", message);
  throw std::runtime_error(message);
}

#define PCA_DRIVER_CHECK(expr) driver_check((expr), #expr, __FILE__, __LINE__)

class ScopedCudaContext {
public:
  explicit ScopedCudaContext(CUcontext context) {
    CUcontext current = nullptr;
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&current));

    if (current != context) {
      PCA_DRIVER_CHECK(cuCtxPushCurrent(context));
      pushed_ = true;
    }
  }

  ~ScopedCudaContext() noexcept {
    if (!pushed_) {
      return;
    }

    CUcontext  popped = nullptr;
    const auto result = cuCtxPopCurrent(&popped);
    if (result != CUDA_SUCCESS) {
      logger()->critical("{}", driver_error_message(result, "cuCtxPopCurrent", __FILE__, __LINE__));
      std::abort();
    }
  }

  ScopedCudaContext(const ScopedCudaContext &)            = delete;
  ScopedCudaContext &operator=(const ScopedCudaContext &) = delete;

private:
  bool pushed_{false};
};

CUcontext cuda_context_for_stream(cudaStream_t stream) {
  CUcontext context = nullptr;

  if (stream != nullptr) {
    PCA_DRIVER_CHECK(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context));
  } else {
    PCA_DRIVER_CHECK(cuCtxGetCurrent(&context));
  }

  if (context == nullptr) {
    throw std::runtime_error("PCA eigensolver requires an active CUDA context");
  }

  return context;
}

template <typename T> T read_module_constant(CUmodule module, const char *name) {
  CUdeviceptr address = 0;
  size_t      size    = 0;

  PCA_DRIVER_CHECK(cuModuleGetGlobal(&address, &size, module, name));
  if (size != sizeof(T)) {
    throw std::runtime_error(
        std::format("Unexpected size for cuSolverDx module constant {}", name));
  }

  T value{};
  PCA_DRIVER_CHECK(cuMemcpyDtoH(&value, address, sizeof(T)));
  return value;
}

CUfunction module_function(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  PCA_DRIVER_CHECK(cuModuleGetFunction(&function, module, name));
  return function;
}

// -------------------------------------------------------------------------------------------------
// Conventional cuSOLVER eigensolver
// -------------------------------------------------------------------------------------------------

class CusolverEigensolver final : public Eigensolver {
public:
  CusolverEigensolver(int features, size_t batches, float *matrices, float *eigenvalues,
                      cudaStream_t stream)
      : features_(features), batches_(batches), context_(cuda_context_for_stream(stream)),
        handle_(std::make_unique<curaii::CusolverDnHandle>()),
        params_(std::make_unique<curaii::CusolverDnParams>()) {
    if (batches_ > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
      throw std::invalid_argument("[Pca] Batch count exceeds the cuSOLVER API limit");
    }

    CUSOLVER_CHECK(cusolverDnSetStream(handle_->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
        handle_->get(), params_->get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, features_,
        CUDA_R_32F, matrices, features_, CUDA_R_32F, eigenvalues, CUDA_R_32F,
        &device_workspace_size_, &host_workspace_size_, static_cast<int64_t>(batches_)));

    if (device_workspace_size_ != 0) {
      device_workspace_ = curaii::make_unique_device_ptr<std::byte>(device_workspace_size_);
    }
    host_workspace_.resize(host_workspace_size_);
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const override {
    return cuda_context_for_stream(stream) == context_;
  }

  void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) override {
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument(
          "[Pca] cuSOLVER eigensolver and PCA stream use different contexts");
    }

    CUSOLVER_CHECK(cusolverDnSetStream(handle_->get(), stream));
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
        handle_->get(), params_->get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, features_,
        CUDA_R_32F, matrices, features_, CUDA_R_32F, eigenvalues, CUDA_R_32F,
        device_workspace_.get(), device_workspace_size_, host_workspace_.data(),
        host_workspace_size_, info, static_cast<int64_t>(batches_)));
  }

private:
  int       features_;
  size_t    batches_;
  CUcontext context_{nullptr};

  std::unique_ptr<curaii::CusolverDnHandle> handle_;
  std::unique_ptr<curaii::CusolverDnParams> params_;

  curaii::unique_device_ptr<std::byte> device_workspace_;
  size_t                               device_workspace_size_{0};
  std::vector<std::byte>               host_workspace_;
  size_t                               host_workspace_size_{0};
};

// -------------------------------------------------------------------------------------------------
// cuSolverDx runtime compilation
// -------------------------------------------------------------------------------------------------

const char *cusolverdx_source();

namespace cusolverdx_runtime {

struct Assets {
  std::filesystem::path cuda_include;
  std::filesystem::path cusolverdx_include;
  std::filesystem::path cutlass_include;
  std::filesystem::path cusolverdx_fatbin;
};

std::string jitlink_error_log(nvJitLinkHandle linker) {
  size_t size = 0;
  if (linker == nullptr || nvJitLinkGetErrorLogSize(linker, &size) != NVJITLINK_SUCCESS ||
      size == 0) {
    return {};
  }

  std::vector<char> log(size);
  if (nvJitLinkGetErrorLog(linker, log.data()) != NVJITLINK_SUCCESS) {
    return {};
  }

  return std::string(log.data());
}

void jitlink_check(nvJitLinkHandle linker, nvJitLinkResult result, const char *expression,
                   const char *file, int line) {
  if (result == NVJITLINK_SUCCESS) {
    return;
  }

  const auto log     = jitlink_error_log(linker);
  const auto message = std::format("nvJitLink error: {} at {}:{}\n  expression : {}\n{}",
                                   static_cast<int>(result), file, line, expression, log);
  logger()->error("{}", message);
  throw std::runtime_error(message);
}

#define PCA_NVJITLINK_CHECK(linker, expr) jitlink_check((linker), (expr), #expr, __FILE__, __LINE__)

std::filesystem::path executable_directory() {
  std::vector<wchar_t> buffer(MAX_PATH);

  while (true) {
    const auto length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));

    if (length == 0) {
      throw std::runtime_error(
          std::format("[Pca] Failed to locate the executable (Windows error {})", GetLastError()));
    }

    if (length < buffer.size() - 1) {
      return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    }

    buffer.resize(buffer.size() * 2);
  }
}

bool assets_exist(const Assets &assets) {
  return std::filesystem::is_directory(assets.cuda_include) &&
         std::filesystem::is_regular_file(assets.cuda_include / "cuda_runtime.h") &&
         std::filesystem::is_directory(assets.cuda_include / "cccl") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx.hpp") &&
         std::filesystem::is_regular_file(assets.cusolverdx_include / "cusolverdx_io.hpp") &&
         std::filesystem::is_directory(assets.cutlass_include) &&
         std::filesystem::is_regular_file(assets.cusolverdx_fatbin);
}

Assets locate_assets() {
  const auto installed_root =
      (executable_directory() / HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR).lexically_normal();

  const Assets installed{
      installed_root / "cuda/include",
      installed_root / "mathdx/include",
      installed_root / "mathdx/cutlass/include",
      installed_root / "mathdx/lib/libcusolverdx.fatbin",
  };

  if (assets_exist(installed)) {
    return installed;
  }

  const Assets build{
      HOLOFLOW_CUDA_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR,
      HOLOFLOW_CUSOLVERDX_FATBIN,
  };

  if (assets_exist(build)) {
    return build;
  }

  throw std::runtime_error(std::format(
      "[Pca] cuSolverDx runtime-compilation assets are missing. Expected installed assets under "
      "'{}' (CUDA headers, cuSolverDx/CommonDx headers, CUTLASS headers, and "
      "libcusolverdx.fatbin); build-tree fallback is also unavailable",
      installed_root.string()));
}

std::vector<char> compile_lto_ir(int features, int solver_sm, int architecture,
                                 const Assets &assets) {
  curaii::NvrtcProgram program(cusolverdx_source(), "pca_heev_kernel.cu");

  std::vector<std::string> options = {
      "--std=c++17",
      "--device-as-default-execution-space",
      "-dlto",
      "--relocatable-device-code=true",
      std::format("--gpu-architecture=sm_{}", architecture),
      std::format("-DPCA_FEATURES={}", features),
      std::format("-DPCA_SOLVER_SM={}", solver_sm),
      std::format("--include-path={}", assets.cuda_include.string()),
      std::format("--include-path={}", (assets.cuda_include / "cccl").string()),
      std::format("--include-path={}", assets.cusolverdx_include.string()),
      std::format("--include-path={}", assets.cutlass_include.string()),
  };

  std::vector<const char *> option_ptrs;
  option_ptrs.reserve(options.size());
  for (const auto &option : options) {
    option_ptrs.push_back(option.c_str());
  }

  const auto compile_result =
      nvrtcCompileProgram(program.get(), static_cast<int>(option_ptrs.size()), option_ptrs.data());

  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program.get(), &log_size));

    std::vector<char> log(log_size);
    NVRTC_CHECK(nvrtcGetProgramLog(program.get(), log.data()));
    logger()->error("[Pca] NVRTC compilation log:\n{}", log.data());
    NVRTC_CHECK(compile_result);
  }

  size_t lto_size = 0;
  NVRTC_CHECK(nvrtcGetLTOIRSize(program.get(), &lto_size));

  std::vector<char> lto_ir(lto_size);
  NVRTC_CHECK(nvrtcGetLTOIR(program.get(), lto_ir.data()));
  return lto_ir;
}

std::vector<char> link_cubin(const std::vector<char> &lto_ir, int architecture,
                             const Assets &assets) {
  nvJitLinkHandle linker      = nullptr;
  const auto      arch_option = std::format("-arch=sm_{}", architecture);
  const char     *options[]   = {"-lto", arch_option.c_str()};

  PCA_NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker, 2, options));

  try {
    PCA_NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN,
                                                 assets.cusolverdx_fatbin.string().c_str()));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR,
                                                 const_cast<char *>(lto_ir.data()), lto_ir.size(),
                                                 "pca_heev_lto_ir"));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));

    size_t cubin_size = 0;
    PCA_NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));

    std::vector<char> cubin(cubin_size);
    PCA_NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    PCA_NVJITLINK_CHECK(linker, nvJitLinkDestroy(&linker));
    return cubin;
  } catch (...) {
    if (linker != nullptr) {
      (void)nvJitLinkDestroy(&linker);
    }
    throw;
  }
}

std::vector<char> compile_cubin(int features, int solver_sm, int architecture) {
  const auto assets = locate_assets();
  const auto lto_ir = compile_lto_ir(features, solver_sm, architecture, assets);
  return link_cubin(lto_ir, architecture, assets);
}

} // namespace cusolverdx_runtime

// -------------------------------------------------------------------------------------------------
// cuSolverDx eigensolver
// -------------------------------------------------------------------------------------------------

struct DxKernel {
  CUfunction   function{nullptr};
  unsigned int block_x{0};
  unsigned int shared_memory{0};
};

class CusolverDxEigensolver final : public Eigensolver {
public:
  CusolverDxEigensolver(int features, size_t batches, cudaStream_t stream)
      : features_(features), batches_(batches) {
    CUDA_CHECK(cudaGetDevice(&device_));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_));

    architecture_       = major * 10 + minor;
    const int solver_sm = architecture_ * 10;

    PCA_DRIVER_CHECK(cuInit(0));
    context_ = cuda_context_for_stream(stream);

    ScopedCudaContext context_guard(context_);

    const auto cubin = cusolverdx_runtime::compile_cubin(features_, solver_sm, architecture_);
    PCA_DRIVER_CHECK(cuModuleLoadDataEx(&module_, cubin.data(), 0, nullptr, nullptr));

    try {
      load_kernel_configuration();
      configure_shared_memory();
    } catch (...) {
      (void)cuModuleUnload(module_);
      module_ = nullptr;
      throw;
    }
  }

  ~CusolverDxEigensolver() noexcept override {
    if (module_ == nullptr) {
      return;
    }

    try {
      ScopedCudaContext context_guard(context_);
      const auto        result = cuModuleUnload(module_);

      if (result != CUDA_SUCCESS) {
        logger()->critical("{}",
                           driver_error_message(result, "cuModuleUnload", __FILE__, __LINE__));
        std::abort();
      }
    } catch (...) {
      std::abort();
    }
  }

  [[nodiscard]] bool is_compatible_stream(cudaStream_t stream) const override {
    return cuda_context_for_stream(stream) == context_;
  }

  void solve(float *matrices, float *eigenvalues, int *info, cudaStream_t stream) override {
    if (!is_compatible_stream(stream)) {
      throw std::invalid_argument(
          "[Pca] cuSolverDx eigensolver and PCA stream use different contexts");
    }

    const size_t full_batches = (batches_ / batches_per_block_) * batches_per_block_;
    if (full_batches != 0) {
      launch_batched(matrices, eigenvalues, info, full_batches, stream);
    }

    const size_t tail_batches = batches_ - full_batches;
    if (tail_batches != 0) {
      launch_tail(matrices, eigenvalues, info, full_batches, tail_batches, stream);
    }
  }

private:
  void load_kernel_configuration() {
    batched_kernel_ = {
        .function      = module_function(module_, "pca_heev_batched"),
        .block_x       = read_module_constant<unsigned int>(module_, "pca_batched_block_x"),
        .shared_memory = read_module_constant<unsigned int>(module_, "pca_batched_shared_memory"),
    };

    tail_kernel_ = {
        .function      = module_function(module_, "pca_heev_tail"),
        .block_x       = read_module_constant<unsigned int>(module_, "pca_tail_block_x"),
        .shared_memory = read_module_constant<unsigned int>(module_, "pca_tail_shared_memory"),
    };

    batches_per_block_ = read_module_constant<unsigned int>(module_, "pca_batches_per_block");
  }

  void configure_shared_memory() {
    int max_shared_memory = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device_));

    const bool uses_batched = batches_ >= batches_per_block_;
    const bool uses_tail    = batches_ < batches_per_block_ || (batches_ % batches_per_block_) != 0;

    if (uses_batched &&
        batched_kernel_.shared_memory > static_cast<unsigned int>(max_shared_memory)) {
      throw_shared_memory_error(batched_kernel_.shared_memory, max_shared_memory);
    }

    if (uses_tail && tail_kernel_.shared_memory > static_cast<unsigned int>(max_shared_memory)) {
      throw_shared_memory_error(tail_kernel_.shared_memory, max_shared_memory);
    }

    if (uses_batched) {
      PCA_DRIVER_CHECK(cuFuncSetAttribute(batched_kernel_.function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          static_cast<int>(batched_kernel_.shared_memory)));
    }

    if (uses_tail) {
      PCA_DRIVER_CHECK(cuFuncSetAttribute(tail_kernel_.function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          static_cast<int>(tail_kernel_.shared_memory)));
    }
  }

  [[noreturn]] void throw_shared_memory_error(unsigned int required, int max_shared_memory) const {
    throw std::runtime_error(
        std::format("[Pca] cuSolverDx HEEV for {} features requires {} bytes of dynamic "
                    "shared memory, but device sm_{} provides {} bytes",
                    features_, required, architecture_, max_shared_memory));
  }

  void launch_batched(float *matrices, float *eigenvalues, int *info, size_t full_batches,
                      cudaStream_t stream) const {
    CUdeviceptr matrix_arg      = reinterpret_cast<CUdeviceptr>(matrices);
    CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues);
    CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info);
    auto        batch_arg       = static_cast<unsigned int>(full_batches);
    void       *arguments[]     = {&matrix_arg, &eigenvalues_arg, &info_arg, &batch_arg};

    PCA_DRIVER_CHECK(cuLaunchKernel(batched_kernel_.function,
                                    static_cast<unsigned int>(full_batches / batches_per_block_), 1,
                                    1, batched_kernel_.block_x, 1, 1, batched_kernel_.shared_memory,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  void launch_tail(float *matrices, float *eigenvalues, int *info, size_t full_batches,
                   size_t tail_batches, cudaStream_t stream) const {
    const size_t matrix_offset = full_batches * static_cast<size_t>(features_) * features_;
    const size_t value_offset  = full_batches * static_cast<size_t>(features_);

    CUdeviceptr matrix_arg      = reinterpret_cast<CUdeviceptr>(matrices + matrix_offset);
    CUdeviceptr eigenvalues_arg = reinterpret_cast<CUdeviceptr>(eigenvalues + value_offset);
    CUdeviceptr info_arg        = reinterpret_cast<CUdeviceptr>(info + full_batches);
    auto        batch_arg       = static_cast<unsigned int>(tail_batches);
    void       *arguments[]     = {&matrix_arg, &eigenvalues_arg, &info_arg, &batch_arg};

    PCA_DRIVER_CHECK(cuLaunchKernel(tail_kernel_.function, static_cast<unsigned int>(tail_batches),
                                    1, 1, tail_kernel_.block_x, 1, 1, tail_kernel_.shared_memory,
                                    reinterpret_cast<CUstream>(stream), arguments, nullptr));
  }

  int       features_;
  size_t    batches_;
  int       device_{0};
  int       architecture_{0};
  CUcontext context_{nullptr};
  CUmodule  module_{nullptr};

  DxKernel     batched_kernel_;
  DxKernel     tail_kernel_;
  unsigned int batches_per_block_{1};
};

// -------------------------------------------------------------------------------------------------
// Eigensolver selection
// -------------------------------------------------------------------------------------------------

constexpr int cusolverdx_max_features_exclusive = 256;

std::unique_ptr<Eigensolver> make_eigensolver(const PcaLayout    &layout,
                                              const PcaWorkspace &workspace, cudaStream_t stream) {
  if (layout.features < cusolverdx_max_features_exclusive) {
    try {
      return std::make_unique<CusolverDxEigensolver>(layout.features, layout.batches, stream);
    } catch (const std::exception &error) {
      logger()->warn("[Pca] cuSolverDx initialization failed for depth {}: {}\n"
                     "[Pca] Falling back to the conventional cuSOLVER eigensolver",
                     layout.features, error.what());
    }
  }

  try {
    return std::make_unique<CusolverEigensolver>(layout.features, layout.batches,
                                                 workspace.matrices.get(),
                                                 workspace.eigenvalues.get(), stream);
  } catch (const std::exception &error) {
    logger()->error("[Pca] Failed to initialize any GPU eigensolver for depth {}: {}",
                    layout.features, error.what());

    throw std::runtime_error(std::format("PCA could not initialize a GPU eigensolver for depth {}. "
                                         "See the terminal log for details.",
                                         layout.features));
  }
}

// -------------------------------------------------------------------------------------------------
// Embedded cuSolverDx device program
// -------------------------------------------------------------------------------------------------

const char *cusolverdx_source() {
  static constexpr char source[] = R"cusolverdx(
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

using namespace cusolverdx;

using Base = decltype(Size<PCA_FEATURES>() + Precision<float>() + Type<type::real>() +
                      Function<heev>() + FillMode<fill_mode::lower>() +
                      Arrangement<arrangement::col_major>() +
                      Job<job::overwrite_vectors>() + SM<PCA_SOLVER_SM>() + Block());
using BatchedSolver =
    decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());
using TailSolver = decltype(Base() + BatchesPerBlock<1>());

extern "C" __constant__ unsigned int pca_batched_block_x = BatchedSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_batched_shared_memory =
    BatchedSolver::shared_memory_size;
extern "C" __constant__ unsigned int pca_batches_per_block =
    BatchedSolver::batches_per_block;
extern "C" __constant__ unsigned int pca_tail_block_x = TailSolver::block_dim.x;
extern "C" __constant__ unsigned int pca_tail_shared_memory = TailSolver::shared_memory_size;

template <class Solver>
__device__ void solve(float *covariance, float *eigenvalues, int *info,
                      const unsigned int batches) {
  constexpr unsigned int m = Solver::m_size;
  constexpr unsigned int batches_per_block = Solver::batches_per_block;
  constexpr unsigned int lda_shared = Solver::lda;
  constexpr unsigned int matrix_elements = m * m;

  const unsigned int batch = blockIdx.x * batches_per_block;
  if (batch >= batches) {
    return;
  }

  extern __shared__ __align__(16) cusolverdx::byte shared_memory[];
  auto [matrix_shared, eigenvalues_shared, workspace_shared] =
      cusolverdx::shared_memory::slice<float, float, float>(
          shared_memory, alignof(float), lda_shared * m * batches_per_block, alignof(float),
          m * batches_per_block, alignof(float));

  float *matrix_global = covariance + matrix_elements * batch;
  float *eigenvalues_global = eigenvalues + m * batch;

  cusolverdx::copy_2d<Solver, m, m, arrangement::col_major, batches_per_block>(
      matrix_global, m, matrix_shared, lda_shared);
  __syncthreads();

  Solver().execute(matrix_shared, lda_shared, eigenvalues_shared, workspace_shared, info + batch);

  cusolverdx::copy_2d<Solver, m, 1, arrangement::col_major, batches_per_block>(
      eigenvalues_shared, m, eigenvalues_global, m);
  __syncthreads();
  cusolverdx::copy_2d<Solver, m, m, arrangement::col_major, batches_per_block>(
      matrix_shared, lda_shared, matrix_global, m);
}

extern "C" __global__ void pca_heev_batched(float *covariance, float *eigenvalues, int *info,
                                             const unsigned int batches) {
  solve<BatchedSolver>(covariance, eigenvalues, info, batches);
}

extern "C" __global__ void pca_heev_tail(float *covariance, float *eigenvalues, int *info,
                                          const unsigned int batches) {
  solve<TailSolver>(covariance, eigenvalues, info, batches);
}
)cusolverdx";

  return source;
}

#undef PCA_NVJITLINK_CHECK
#undef PCA_DRIVER_CHECK

} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const PcaSettings &settings) {
  j = nlohmann::json{
      {"begin", settings.begin},
      {"end", settings.end},
  };
}

void from_json(const nlohmann::json &j, PcaSettings &settings) {
  j.at("begin").get_to(settings.begin);
  j.at("end").get_to(settings.end);
}

// -------------------------------------------------------------------------------------------------
// Factory methods
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
PcaProjectionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings) const {
  if (input_descs.size() != 2)
    throw std::invalid_argument("PCA projection expects X and eigenvectors");
  auto                result = PcaFactory{}.infer(input_descs.first(1), jsettings);
  const auto         &x      = input_descs[0];
  const auto         &v      = input_descs[1];
  std::vector<size_t> shape(x.shape.begin(), x.shape.end() - 2);
  shape.push_back(x.shape[x.rank() - 3]);
  const holoflow::core::TDesc contiguous_x(x.shape, x.dtype, x.mem_loc, x.offset);
  const holoflow::core::TDesc contiguous_v(shape, holoflow::core::DType::F32,
                                           holoflow::core::MemLoc::Device, v.offset);
  if (x.strides != contiguous_x.strides || v.shape != contiguous_v.shape ||
      v.dtype != contiguous_v.dtype || v.mem_loc != contiguous_v.mem_loc ||
      v.strides != contiguous_v.strides)
    throw std::invalid_argument("PCA projection expects contiguous X and matching eigenvectors");
  result.input_descs.push_back(v);
  result.owned_inputs.push_back(false);
  return result;
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaProjectionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                             const nlohmann::json                  &jsettings,
                             const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  return std::make_unique<PcaProjectionTask>(jsettings.get<PcaSettings>(), input_descs[0],
                                             ctx.stream);
}

holoflow::core::InferResult PcaFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &message) {
    if (!condition) {
      logger()->error("[PcaFactory::infer] error: {}", message);
      throw std::invalid_argument("PcaFactory inference error: " + message);
    }
  };

  const auto settings = jsettings.get<PcaSettings>();

  check(input_descs.size() == 1, "expected exactly one input");

  const auto &input_desc = input_descs.front();
  check(input_desc.rank() >= 3, "expected input rank >= 3");
  check(input_desc.dtype == holoflow::core::DType::F32, "PCA currently supports F32 input only");
  check(input_desc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");

  const PcaLayout layout(input_desc);
  check(settings.end <= layout.features, "expected end <= n_features");

  auto output_shape                    = input_desc.shape;
  output_shape.at(layout.feature_axis) = static_cast<size_t>(settings.components());

  holoflow::core::TDesc output_desc(output_shape, input_desc.dtype, input_desc.mem_loc);

  return holoflow::core::InferResult{
      .input_descs   = {input_desc},
      .output_descs  = {output_desc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  this->infer(input_descs, jsettings);

  return std::make_unique<PcaTask>(jsettings.get<PcaSettings>(), input_descs.front(), ctx);
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  this->infer(input_descs, jsettings);

  const auto  settings   = jsettings.get<PcaSettings>();
  const auto &input_desc = input_descs.front();

  auto *pca = dynamic_cast<PcaTask *>(old_task.get());
  if (pca != nullptr && pca->can_reuse(input_desc, ctx.stream)) {
    if (!pca->same_settings(settings))
      ctx.invalidate_execution();
    pca->reconfigure(settings, ctx.stream);
    return old_task;
  }

  ctx.invalidate_execution();

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs

namespace holotask::asyncs {

void to_json(nlohmann::json &j, const PcaDecompositionQueueSettings &settings) {
  j = {{"batch_size", settings.batch_size}, {"slot_count", settings.slot_count}};
}

void from_json(const nlohmann::json &j, PcaDecompositionQueueSettings &settings) {
  j.at("batch_size").get_to(settings.batch_size);
  j.at("slot_count").get_to(settings.slot_count);
}

holoflow::core::InferResult
PcaDecompositionQueueFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings) const {
  using namespace holoflow::core;
  const auto settings = jsettings.get<PcaDecompositionQueueSettings>();
  if (input_descs.size() != 1 || input_descs[0].rank() != 4 || input_descs[0].shape[0] != 1)
    throw std::invalid_argument("PCA queue expects one [1,T,H,W] input");
  const auto &input = input_descs[0];
  const TDesc contiguous(input.shape, DType::F32, MemLoc::Device);
  if (input.dtype != contiguous.dtype || input.mem_loc != contiguous.mem_loc ||
      input.strides != contiguous.strides || input.offset != 0 || settings.batch_size == 0 ||
      settings.slot_count == 0)
    throw std::invalid_argument(
        "PCA queue expects contiguous device F32 input and positive capacity");
  const auto int_max = static_cast<size_t>((std::numeric_limits<int>::max)());
  for (size_t dimension : input.shape)
    if (dimension == 0 || dimension > int_max)
      throw std::invalid_argument("PCA queue dimension exceeds the GEMM limit");
  if (input.shape[2] > int_max / input.shape[3])
    throw std::invalid_argument("PCA queue sample count exceeds the GEMM limit");
  // Check all allocation products before constructing descriptors or PcaLayout.
  const auto size_max      = (std::numeric_limits<size_t>::max)();
  auto       check_product = [&](std::initializer_list<size_t> factors) {
    size_t product = sizeof(float);
    for (auto factor : factors) {
      if (factor > size_max / product)
        throw std::overflow_error("PCA queue allocation size overflow");
      product *= factor;
    }
  };
  check_product(
      {settings.slot_count, settings.batch_size, input.shape[1], input.shape[2], input.shape[3]});
  check_product({settings.slot_count, settings.batch_size, input.shape[1], input.shape[1]});
  auto shape = input.shape;
  shape[0]   = settings.batch_size;
  const TDesc x(shape, DType::F32, MemLoc::Device);
  const TDesc vectors({settings.batch_size, shape[1], shape[1]}, DType::F32, MemLoc::Device);
  return {
      .input_descs                 = {input},
      .output_descs                = {x, vectors},
      .in_place                    = {},
      .owned_inputs                = {true},
      .owned_outputs               = {true, true},
      .kind                        = TaskKind::Async,
      .owned_input_pointer_counts  = {settings.batch_size * settings.slot_count},
      .owned_output_pointer_counts = {settings.slot_count, settings.slot_count},
      .supports_prequeued_push     = true,
      .supports_prequeued_pop      = true,
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
PcaDecompositionQueueFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                     const nlohmann::json                  &jsettings,
                                     const holoflow::core::AsyncCreateCtx  &ctx) const {
  const auto inferred = infer(input_descs, jsettings);
  return std::make_unique<syncs::PcaDecompositionQueue>(
      jsettings.get<PcaDecompositionQueueSettings>(), inferred, ctx.producer_stream);
}

std::unique_ptr<holoflow::core::IAsyncTask>
PcaDecompositionQueueFactory::update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
                                     std::span<const holoflow::core::TDesc>      input_descs,
                                     const nlohmann::json                       &jsettings,
                                     const holoflow::core::AsyncCreateCtx       &ctx) const {
  ctx.invalidate_execution();
  // Destroying the old task drains its private solver before captured allocations disappear.
  old_task.reset();
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::asyncs
