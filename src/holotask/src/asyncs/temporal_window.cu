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

#include "holotask/asyncs/temporal_window.hh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace holotask::asyncs {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void to_json(nlohmann::json &j, const TemporalWindowSettings &s) {
  j = nlohmann::json{{"window_size", s.window_size},
                     {"stride_size", s.stride_size},
                     {"source_cycle_frames", s.source_cycle_frames}};
}

void from_json(const nlohmann::json &j, TemporalWindowSettings &s) {
  s.window_size = j.value("window_size", size_t{3});
  s.stride_size = j.value("stride_size", size_t{1});
  s.source_cycle_frames = j.value("source_cycle_frames", size_t{0});
}

namespace {

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

class TemporalWindow final : public holoflow::core::IAsyncTask {
public:
  TemporalWindow(TemporalWindowSettings settings, holoflow::core::TDesc input_desc,
                 holoflow::core::TDesc output_desc, cudaStream_t producer_stream,
                 cudaStream_t consumer_stream, size_t capacity, size_t frame_bytes,
                 DevPtr<std::byte> &&frame_buffer, DevPtr<std::byte> &&window_buffer)
      : settings_(settings), input_desc_(std::move(input_desc)),
        output_desc_(std::move(output_desc)), producer_stream_(producer_stream),
        consumer_stream_(consumer_stream), capacity_(capacity), frame_bytes_(frame_bytes),
        frame_buffer_(std::move(frame_buffer)), window_buffer_(std::move(window_buffer)) {}

  void release_output(int index) override {
    if (index != 0)
      throw std::out_of_range("TemporalWindow::release_output: invalid index");

    std::scoped_lock state_lock(state_mutex_);
    if (!pop_active_)
      throw std::logic_error("TemporalWindow::release_output: no active output");

    storage_access().owned_output_storage(0).ptr = nullptr;
    size_t stride_frames = settings_.stride_size;
    if (settings_.source_cycle_frames != 0) {
      const size_t next_offset =
          (window_start_offset_ + stride_frames % settings_.source_cycle_frames) %
          settings_.source_cycle_frames;
      // A replayed holofile must not contribute frames from two different passes to one window.
      if (next_offset > settings_.source_cycle_frames - settings_.window_size) {
        stride_frames += settings_.source_cycle_frames - next_offset;
      }
      window_start_offset_ =
          (window_start_offset_ + stride_frames % settings_.source_cycle_frames) %
          settings_.source_cycle_frames;
    }

    const size_t buffered_frames = writer_size();
    const size_t advance_frames = std::min(stride_frames, buffered_frames);
    read_count_.fetch_add(advance_frames, std::memory_order_release);
    pending_skip_ += stride_frames - advance_frames;
    pop_active_ = false;
  }

  std::optional<std::vector<std::byte *>> owned_output_pointers(size_t index) const override {
    if (index != 0)
      throw std::out_of_range("TemporalWindow output port");
    return std::vector<std::byte *>{window_buffer_.get()};
  }

  std::optional<holoflow::core::PointerSequence>
  owned_output_pointer_sequence(size_t index) const override {
    if (index != 0)
      throw std::out_of_range("TemporalWindow output port");
    return holoflow::core::PointerSequence{.cycle = {0}};
  }

  holoflow::core::OpResult try_push(holoflow::core::AsyncPushCtx &ctx) override {
    std::scoped_lock state_lock(state_mutex_);
    if (pending_skip_ > 0) {
      --pending_skip_;
      return holoflow::core::OpResult::Ok;
    }

    if (writer_size() >= capacity_)
      return holoflow::core::OpResult::NotReady;

    const size_t write_count = write_count_.load(std::memory_order_relaxed);
    auto       *destination = frame_buffer_.get() + (write_count % capacity_) * frame_bytes_;
    CUDA_CHECK(cudaMemcpyAsync(destination, ctx.inputs[0].data(), frame_bytes_,
                               cudaMemcpyDeviceToDevice, producer_stream_));
    // The upstream owner may release and reuse its output as soon as this push returns.
    CUDA_CHECK(cudaStreamSynchronize(producer_stream_));
    write_count_.fetch_add(1, std::memory_order_release);
    return holoflow::core::OpResult::Ok;
  }

  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &ctx) override {
    if (writer_size() < settings_.window_size)
      return holoflow::core::OpResult::NotReady;
    if (pop_active_)
      throw std::logic_error("TemporalWindow::try_pop: prior output was not released");

    const size_t first_frame = read_count_.load(std::memory_order_relaxed);
    const size_t first_index = first_frame % capacity_;
    const size_t first_frames = std::min(settings_.window_size, capacity_ - first_index);
    const size_t first_bytes = first_frames * frame_bytes_;
    CUDA_CHECK(cudaMemcpyAsync(window_buffer_.get(),
                               frame_buffer_.get() + first_index * frame_bytes_, first_bytes,
                               cudaMemcpyDeviceToDevice, consumer_stream_));
    const size_t remaining_frames = settings_.window_size - first_frames;
    if (remaining_frames > 0) {
      CUDA_CHECK(cudaMemcpyAsync(window_buffer_.get() + first_bytes, frame_buffer_.get(),
                                 remaining_frames * frame_bytes_, cudaMemcpyDeviceToDevice,
                                 consumer_stream_));
    }

    auto &storage = storage_access().owned_output_storage(0);
    storage.ptr   = window_buffer_.get();
    ctx.outputs[0] = holoflow::core::TView{.desc = output_desc_, .storage = &storage};
    pop_active_    = true;
    return holoflow::core::OpResult::Ok;
  }

  void update_streams(cudaStream_t producer_stream, cudaStream_t consumer_stream) {
    producer_stream_ = producer_stream;
    consumer_stream_ = consumer_stream;
  }

  void reset() {
    std::scoped_lock state_lock(state_mutex_);
    write_count_.store(0, std::memory_order_relaxed);
    read_count_.store(0, std::memory_order_relaxed);
    pending_skip_ = 0;
    window_start_offset_ = 0;
    pop_active_ = false;
    storage_access().owned_output_storage(0).ptr = nullptr;
  }

  const TemporalWindowSettings &settings() const { return settings_; }
  const holoflow::core::TDesc  &input_desc() const { return input_desc_; }

private:
  size_t writer_size() const {
    const size_t written = write_count_.load(std::memory_order_acquire);
    const size_t read    = read_count_.load(std::memory_order_acquire);
    return written - read;
  }

  TemporalWindowSettings  settings_;
  holoflow::core::TDesc   input_desc_;
  holoflow::core::TDesc   output_desc_;
  cudaStream_t            producer_stream_;
  cudaStream_t            consumer_stream_;
  size_t                  capacity_;
  size_t                  frame_bytes_;
  DevPtr<std::byte>       frame_buffer_;
  DevPtr<std::byte>       window_buffer_;
  std::mutex              state_mutex_;
  size_t                  pending_skip_ = 0;
  size_t                  window_start_offset_ = 0;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_count_{0};
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_count_{0};
  bool pop_active_ = false;
};

} // namespace

holoflow::core::InferResult
TemporalWindowFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                             const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<TemporalWindowSettings>();
  if (input_descs.size() != 1)
    throw std::invalid_argument("TemporalWindow expects exactly one input");
  if (settings.window_size < 3)
    throw std::invalid_argument("TemporalWindow window_size must be at least 3");
  if (settings.stride_size == 0)
    throw std::invalid_argument("TemporalWindow stride_size must be positive");
  if (settings.source_cycle_frames != 0 && settings.source_cycle_frames < settings.window_size)
    throw std::invalid_argument("TemporalWindow source cycle must contain a full window");

  const auto &input = input_descs[0];
  if (input.rank() != 3 || input.shape[0] != 1)
    throw std::invalid_argument("TemporalWindow input must have shape [1, height, width]");
  if (input.mem_loc != holoflow::core::MemLoc::Device || !is_contiguous(input))
    throw std::invalid_argument("TemporalWindow input must be contiguous device memory");
  if (input.dtype != holoflow::core::DType::U8 && input.dtype != holoflow::core::DType::U16 &&
      input.dtype != holoflow::core::DType::F32)
    throw std::invalid_argument("TemporalWindow input must be U8, U16, or F32");
  if (input.shape[1] == 0 || input.shape[2] == 0)
    throw std::invalid_argument("TemporalWindow image dimensions must be positive");

  auto output = input;
  output.shape[0] = settings.window_size;
  if (settings.window_size > std::numeric_limits<size_t>::max() - 2)
    throw std::overflow_error("TemporalWindow capacity overflows");
  const size_t frame_bytes = input.num_bytes();
  const size_t capacity    = settings.window_size + 2;
  if (frame_bytes > std::numeric_limits<size_t>::max() / capacity ||
      frame_bytes > std::numeric_limits<size_t>::max() / settings.window_size)
    throw std::overflow_error("TemporalWindow buffer size overflows");

  return holoflow::core::InferResult{
      .input_descs   = {input},
      .output_descs  = {output},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {true},
      .kind          = holoflow::core::TaskKind::Async,
      .synchronizes_producer_stream = true,
      .owned_output_pointer_counts  = {1},
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
TemporalWindowFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings,
                              const holoflow::core::AsyncCreateCtx   &ctx) const {
  const auto result   = infer(input_descs, jsettings);
  const auto settings = jsettings.get<TemporalWindowSettings>();
  const auto frame_bytes = input_descs[0].num_bytes();
  const auto capacity    = settings.window_size + 2;
  auto       frame_buffer = curaii::make_unique_device_ptr<std::byte>(capacity * frame_bytes);
  auto       window_buffer = curaii::make_unique_device_ptr<std::byte>(
      settings.window_size * frame_bytes);
  return std::make_unique<TemporalWindow>(settings, input_descs[0], result.output_descs[0],
                                          ctx.producer_stream, ctx.consumer_stream, capacity,
                                          frame_bytes, std::move(frame_buffer),
                                          std::move(window_buffer));
}

std::unique_ptr<holoflow::core::IAsyncTask>
TemporalWindowFactory::update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
                              std::span<const holoflow::core::TDesc>      input_descs,
                              const nlohmann::json                       &jsettings,
                              const holoflow::core::AsyncCreateCtx       &ctx) const {
  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  (void)infer(input_descs, jsettings);
  auto *old_window = dynamic_cast<TemporalWindow *>(old_task.get());
  const auto settings = jsettings.get<TemporalWindowSettings>();
  if (old_window != nullptr && settings == old_window->settings() &&
      input_descs[0].shape == old_window->input_desc().shape &&
      input_descs[0].strides == old_window->input_desc().strides &&
      input_descs[0].dtype == old_window->input_desc().dtype &&
      input_descs[0].mem_loc == old_window->input_desc().mem_loc &&
      input_descs[0].offset == old_window->input_desc().offset) {
    old_window->update_streams(ctx.producer_stream, ctx.consumer_stream);
    old_window->reset();
    return old_task;
  }

  ctx.invalidate_execution();
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::asyncs
