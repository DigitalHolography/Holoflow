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

#include "holotask/asyncs/dual_reader_batch_queue.hh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "logger.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace holotask::asyncs {

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

void to_json(nlohmann::json &j, const DualReaderBatchQueueSettings &s) {
  j = nlohmann::json{{"target_capacity", s.target_capacity}, {"window_size", s.window_size}};
}

void from_json(const nlohmann::json &j, DualReaderBatchQueueSettings &s) {
  j.at("target_capacity").get_to(s.target_capacity);
  j.at("window_size").get_to(s.window_size);
}

namespace {

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

size_t lcm_above(size_t x, size_t y, size_t minimum) {
  const size_t base = std::lcm(x, y);
  if (base == 0) {
    throw std::overflow_error("DualReaderBatchQueue: capacity alignment overflow");
  }
  return ((minimum + base - 1) / base) * base;
}

class DualReaderBatchQueue final : public holoflow::core::IAsyncTask {
public:
  DualReaderBatchQueue(DualReaderBatchQueueSettings settings, holoflow::core::TDesc input_desc,
                       holoflow::core::TDesc output_desc, HostPtr<std::byte> host_buffer,
                       DevPtr<std::byte> device_buffer, HostPtr<std::byte> host_scratch,
                       DevPtr<std::byte> device_scratch, std::byte *buffer, std::byte *scratch,
                       size_t slot_count, size_t input_size, size_t element_size)
      : settings_(settings), input_desc_(std::move(input_desc)),
        output_desc_(std::move(output_desc)), host_buffer_(std::move(host_buffer)),
        device_buffer_(std::move(device_buffer)), host_scratch_(std::move(host_scratch)),
        device_scratch_(std::move(device_scratch)), buffer_(buffer), scratch_(scratch),
        slot_count_(slot_count), input_size_(input_size), element_size_(element_size),
        delay_((settings_.window_size - 1) / 2) {}

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    if (index != 0) {
      throw std::out_of_range("DualReaderBatchQueue::acquire_input: invalid index");
    }
    if (slot_count_ - writer_size() <= input_size_) {
      return std::nullopt;
    }

    const size_t write_idx = write_idx_.load(std::memory_order_relaxed);
    auto        &storage   = storage_access().owned_input_storage(0);
    storage.ptr            = buffer_ + write_idx * element_size_;
    return holoflow::core::TView{.desc = input_desc_, .storage = &storage};
  }

  void release_output(int index) override {
    if (index != 0 && index != 1) {
      throw std::out_of_range("DualReaderBatchQueue::release_output: invalid index");
    }
    if (!pop_active_) {
      throw std::logic_error("DualReaderBatchQueue::release_output: no active output");
    }

    auto &storage = storage_access().owned_output_storage(static_cast<size_t>(index));
    storage.ptr   = nullptr;
    if (index == 0) {
      current_released_ = true;
    } else {
      delayed_released_ = true;
    }

    if (!current_released_ || !delayed_released_) {
      return;
    }

    current_read_idx_ = increment(current_read_idx_);
    if (delayed_active_) {
      const size_t delayed_read_idx = delayed_read_idx_.load(std::memory_order_relaxed);
      delayed_read_idx_.store(increment(delayed_read_idx), std::memory_order_release);
    }
    ++sequence_;
    pop_active_ = false;
  }

  holoflow::core::OpResult try_push(holoflow::core::AsyncPushCtx &) override {
    size_t next_write_idx = write_idx_.load(std::memory_order_relaxed) + input_size_;
    if (next_write_idx >= slot_count_) {
      next_write_idx = 0;
    }
    storage_access().owned_input_storage(0).ptr = nullptr;
    write_idx_.store(next_write_idx, std::memory_order_release);
    return holoflow::core::OpResult::Ok;
  }

  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &ctx) override {
    if (reader_size(current_read_idx_) < 1) {
      return holoflow::core::OpResult::NotReady;
    }
    if (pop_active_) {
      throw std::logic_error("DualReaderBatchQueue::try_pop: prior output was not released");
    }

    auto &current_storage         = storage_access().owned_output_storage(0);
    auto &delayed_storage         = storage_access().owned_output_storage(1);
    current_storage.ptr           = buffer_ + current_read_idx_ * element_size_;
    delayed_active_               = sequence_ >= delay_;
    const size_t delayed_read_idx = delayed_read_idx_.load(std::memory_order_relaxed);
    delayed_storage.ptr = delayed_active_ ? buffer_ + delayed_read_idx * element_size_ : scratch_;

    ctx.outputs[0] = {.desc = output_desc_, .storage = &current_storage};
    ctx.outputs[1] = {.desc = output_desc_, .storage = &delayed_storage};
    *reinterpret_cast<std::uint8_t *>(ctx.outputs[2].data()) =
        sequence_ >= settings_.window_size - 1 ? std::uint8_t{1} : std::uint8_t{0};

    current_released_ = false;
    delayed_released_ = false;
    pop_active_       = true;
    return holoflow::core::OpResult::Ok;
  }

private:
  size_t increment(size_t index) const { return index + 1 == slot_count_ ? 0 : index + 1; }

  size_t distance(size_t begin, size_t end) const {
    return end >= begin ? end - begin : end + slot_count_ - begin;
  }

  size_t writer_size() const {
    const size_t write_idx        = write_idx_.load(std::memory_order_relaxed);
    const size_t delayed_read_idx = delayed_read_idx_.load(std::memory_order_acquire);
    return distance(delayed_read_idx, write_idx);
  }

  size_t reader_size(size_t read_idx) const {
    const size_t write_idx = write_idx_.load(std::memory_order_acquire);
    return distance(read_idx, write_idx);
  }

  DualReaderBatchQueueSettings settings_;
  holoflow::core::TDesc        input_desc_;
  holoflow::core::TDesc        output_desc_;
  HostPtr<std::byte>           host_buffer_;
  DevPtr<std::byte>            device_buffer_;
  HostPtr<std::byte>           host_scratch_;
  DevPtr<std::byte>            device_scratch_;
  std::byte                   *buffer_;
  std::byte                   *scratch_;
  size_t                       slot_count_;
  size_t                       input_size_;
  size_t                       element_size_;
  size_t                       delay_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_{0};
  size_t current_read_idx_ = 0;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> delayed_read_idx_{0};
  size_t sequence_         = 0;
  bool   pop_active_       = false;
  bool   delayed_active_   = false;
  bool   current_released_ = false;
  bool   delayed_released_ = false;
};

} // namespace

holoflow::core::InferResult
DualReaderBatchQueueFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &message) {
    if (!condition) {
      logger()->error("[DualReaderBatchQueueFactory::infer] error: {}", message);
      throw std::invalid_argument("DualReaderBatchQueueFactory inference error: " + message);
    }
  };

  const auto settings = jsettings.get<DualReaderBatchQueueSettings>();
  check(input_descs.size() == 1, "task must have exactly one input");
  const auto &input = input_descs[0];
  check(input.rank() > 0, "input rank must be positive");
  check(input.shape[0] > 0, "input leading dimension must be positive");
  check(is_contiguous(input), "input must be contiguous");
  check(settings.target_capacity > 0, "target_capacity must be positive");
  check(settings.window_size > 0, "window_size must be positive");

  auto output     = input;
  output.shape[0] = 1;
  const holoflow::core::TDesc valid({1}, holoflow::core::DType::U8, holoflow::core::MemLoc::Host);
  return {
      .input_descs   = {input},
      .output_descs  = {output, output, valid},
      .in_place      = {},
      .owned_inputs  = {true},
      .owned_outputs = {true, true, false},
      .kind          = holoflow::core::TaskKind::Async,
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
DualReaderBatchQueueFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings,
                                    const holoflow::core::AsyncCreateCtx  &ctx) const {
  const auto   infer        = this->infer(input_descs, jsettings);
  const auto   settings     = jsettings.get<DualReaderBatchQueueSettings>();
  const auto  &input        = input_descs[0];
  const size_t input_size   = input.shape[0];
  const size_t element_size = input.num_bytes() / input_size;
  const size_t delay        = (settings.window_size - 1) / 2;
  const size_t slot_count =
      lcm_above(input_size, size_t{1}, settings.target_capacity + input_size + delay + size_t{1});
  const size_t bytes = slot_count * element_size;

  HostPtr<std::byte> host_buffer;
  DevPtr<std::byte>  device_buffer;
  HostPtr<std::byte> host_scratch;
  DevPtr<std::byte>  device_scratch;
  std::byte         *buffer  = nullptr;
  std::byte         *scratch = nullptr;

  if (input.mem_loc == holoflow::core::MemLoc::Host) {
    host_buffer  = curaii::make_unique_host_ptr<std::byte>(bytes);
    host_scratch = curaii::make_unique_host_ptr<std::byte>(element_size);
    buffer       = host_buffer.get();
    scratch      = host_scratch.get();
    std::fill_n(scratch, element_size, std::byte{0});
  } else {
    device_buffer  = curaii::make_unique_device_ptr<std::byte>(bytes);
    device_scratch = curaii::make_unique_device_ptr<std::byte>(element_size);
    buffer         = device_buffer.get();
    scratch        = device_scratch.get();
    CUDA_CHECK(cudaMemsetAsync(scratch, 0, element_size, ctx.consumer_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.consumer_stream));
  }

  logger()->debug("[DualReaderBatchQueueFactory::create] slots={}, input_size={}, delay={}, "
                  "window_size={}, total_bytes={}",
                  slot_count, input_size, delay, settings.window_size, bytes);

  return std::make_unique<DualReaderBatchQueue>(
      settings, input, infer.output_descs[0], std::move(host_buffer), std::move(device_buffer),
      std::move(host_scratch), std::move(device_scratch), buffer, scratch, slot_count, input_size,
      element_size);
}

std::unique_ptr<holoflow::core::IAsyncTask> DualReaderBatchQueueFactory::update(
    std::unique_ptr<holoflow::core::IAsyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &jsettings, const holoflow::core::AsyncCreateCtx &ctx) const {
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::asyncs
