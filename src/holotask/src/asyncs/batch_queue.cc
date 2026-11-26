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

#include "holotask/asyncs/batch_queue.hh"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "logger.hh"
#include "reservation_queue.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace holotask::asyncs {

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

void to_json(nlohmann::json &j, const BatchQueueSettings &bqs) {
  j = nlohmann::json{
      {"tail", bqs.tail},
      {"target_capacity", bqs.target_capacity},
      {"output_size", bqs.output_size},
      {"output_stride", bqs.output_stride},
  };
  if (bqs.reader_count != 1)
    j["reader_count"] = bqs.reader_count;
}

void from_json(const nlohmann::json &j, BatchQueueSettings &bqs) {
  bqs.reader_count = j.value("reader_count", size_t{1});
  bqs.tail         = j.value("tail", std::string{"error"});
  j.at("target_capacity").get_to(bqs.target_capacity);
  j.at("output_size").get_to(bqs.output_size);
  j.at("output_stride").get_to(bqs.output_stride);
}

namespace {

class BatchQueue : public holoflow::core::IAsyncTask {
public:
  BatchQueue(BatchQueueSettings settings, holoflow::core::TDesc idesc, holoflow::core::TDesc odesc,
             HostPtr<std::byte> &&h_buf, DevPtr<std::byte> &&d_buf, std::byte *buf, size_t nb_slots,
             size_t input_size, size_t element_size)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), odesc_(std::move(odesc)),
        h_buf_(std::move(h_buf)), d_buf_(std::move(d_buf)), buf_(buf), nb_slots_(nb_slots),
        input_size_(input_size), element_size_(element_size), readers_(settings_.reader_count) {}

  std::optional<holoflow::core::TView> acquire_input(int index) override;
  void                                 release_output(int index) override;
  holoflow::core::OpResult             try_push(holoflow::core::AsyncPushCtx &ctx) override;
  holoflow::core::OpResult             try_pop(holoflow::core::AsyncPopCtx &ctx) override;
  holoflow::core::OpResult             reserve_push(holoflow::core::AsyncPushCtx &,
                                                    holoflow::core::AsyncTicket &) override;
  void                                 enqueue_push(holoflow::core::AsyncTicket ticket) override;
  void                                 commit_push(holoflow::core::AsyncTicket ticket) override;
  void                                 cancel_push(holoflow::core::AsyncTicket ticket) override;
  holoflow::core::OpResult             reserve_pop(holoflow::core::AsyncPopCtx &,
                                                   holoflow::core::AsyncTicket &) override;
  void                                 release_pop(holoflow::core::AsyncTicket ticket) override;
  void cancel_pop(holoflow::core::AsyncTicket ticket) override { cancel_output_group(0, ticket); }
  holoflow::core::OpResult reserve_output_group(size_t, holoflow::core::AsyncPopCtx &,
                                                holoflow::core::AsyncTicket &) override;
  void                     release_output_group(size_t, holoflow::core::AsyncTicket) override;
  void cancel_output_group(size_t group, holoflow::core::AsyncTicket ticket) override {
    readers_.at(group).pending.cancel(ticket);
  }

  void close_input() override { closed_.store(true, std::memory_order_release); }

  std::optional<std::vector<std::byte *>> owned_input_pointers(size_t index) const override {
    if (index != 0)
      throw std::out_of_range("BatchQueue input port");
    std::vector<std::byte *> pointers;
    for (size_t i = 0; i < nb_slots_; i += input_size_)
      pointers.push_back(buf_ + i * element_size_);
    return pointers;
  }

  std::optional<std::vector<std::byte *>> owned_output_pointers(size_t index) const override {
    if (index >= readers_.size())
      throw std::out_of_range("BatchQueue output port");
    std::vector<std::byte *> pointers;
    for (size_t i = 0; i < nb_slots_; i += settings_.output_stride)
      pointers.push_back(buf_ + i * element_size_);
    return pointers;
  }

  const holoflow::core::TDesc &idesc() const { return idesc_; }
  std::optional<holoflow::core::PointerSequence>
  owned_input_pointer_sequence(size_t index) const override {
    if (index != 0)
      throw std::out_of_range("BatchQueue input port");
    return sequence(write_idx_.load(std::memory_order_relaxed) / input_size_,
                    nb_slots_ / input_size_);
  }
  std::optional<holoflow::core::PointerSequence>
  owned_output_pointer_sequence(size_t index) const override {
    if (index >= readers_.size())
      throw std::out_of_range("BatchQueue output port");
    return sequence(readers_[index].committed.load(std::memory_order_relaxed) /
                        settings_.output_stride,
                    nb_slots_ / settings_.output_stride);
  }
  size_t             nb_slots() const { return nb_slots_; }
  size_t             element_size() const { return element_size_; }
  HostPtr<std::byte> take_host_buffer() { return std::move(h_buf_); }
  DevPtr<std::byte>  take_device_buffer() { return std::move(d_buf_); }
  void               take_buffers_from(BatchQueue &other) noexcept {
    h_buf_ = std::move(other.h_buf_);
    d_buf_ = std::move(other.d_buf_);
  }
  std::byte *buffer() const { return buf_; }

private:
  static holoflow::core::PointerSequence sequence(size_t phase, size_t count) {
    holoflow::core::PointerSequence result;
    for (size_t i = 0; i < count; ++i)
      result.cycle.push_back((phase + i) % count);
    return result;
  }
  std::atomic<bool> closed_{false};
  size_t            writer_size() const;
  size_t            used_capacity(size_t write) const;

  BatchQueueSettings    settings_;
  holoflow::core::TDesc idesc_;
  holoflow::core::TDesc odesc_;
  HostPtr<std::byte>    h_buf_;
  DevPtr<std::byte>     d_buf_;
  std::byte            *buf_;
  size_t                nb_slots_;
  size_t                input_size_;
  size_t                element_size_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;
  struct alignas(CACHE_LINE_SIZE) Reader {
    std::atomic<size_t>              committed{0};
    detail::ReservationQueue<size_t> pending;
  };
  struct PendingWrite {
    size_t next;
    bool   submitted = false;
  };
  alignas(CACHE_LINE_SIZE) detail::ReservationQueue<PendingWrite> pending_writes_;
  std::vector<Reader> readers_;
};

holoflow::core::OpResult BatchQueue::reserve_push(holoflow::core::AsyncPushCtx &ctx,
                                                  holoflow::core::AsyncTicket  &ticket) {
  using holoflow::core::OpResult;
  if (ctx.cancelled && ctx.cancelled->load())
    return OpResult::Cancelled;
  if (pending_writes_.full())
    return OpResult::NotReady;
  const size_t write = pending_writes_.empty() ? write_idx_.load(std::memory_order_relaxed)
                                               : pending_writes_.back().next;
  const size_t used  = used_capacity(write);
  if (nb_slots_ - used <= input_size_)
    return OpResult::NotReady;
  storage_access().owned_input_storage(0).ptr = buf_ + write * element_size_;
  ticket = pending_writes_.push({(write + input_size_) % nb_slots_});
  return OpResult::Ok;
}

void BatchQueue::enqueue_push(holoflow::core::AsyncTicket ticket) {
  auto &pending = pending_writes_.get(ticket);
  if (pending.submitted)
    throw std::logic_error("BatchQueue reservation already submitted");
  pending.submitted = true;
}

void BatchQueue::commit_push(holoflow::core::AsyncTicket ticket) {
  if (!pending_writes_.get(ticket).submitted)
    throw std::logic_error("BatchQueue reservation not submitted");
  write_idx_.store(pending_writes_.pop(ticket).next, std::memory_order_release);
}

void BatchQueue::cancel_push(holoflow::core::AsyncTicket ticket) {
  if (pending_writes_.get(ticket).submitted)
    throw std::logic_error("Cannot cancel submitted BatchQueue reservation");
  pending_writes_.cancel(ticket);
}

holoflow::core::OpResult BatchQueue::reserve_pop(holoflow::core::AsyncPopCtx &ctx,
                                                 holoflow::core::AsyncTicket &ticket) {
  if (readers_.size() != 1)
    throw std::logic_error("MultiReaderBatchQueue requires a group-specific pop");
  return reserve_output_group(0, ctx, ticket);
}

holoflow::core::OpResult BatchQueue::reserve_output_group(size_t                       group,
                                                          holoflow::core::AsyncPopCtx &ctx,
                                                          holoflow::core::AsyncTicket &ticket) {
  using holoflow::core::OpResult;
  if (ctx.cancelled && ctx.cancelled->load())
    return OpResult::Cancelled;
  auto &reader = readers_.at(group);
  if (reader.pending.full())
    return OpResult::NotReady;
  const bool   closed    = closed_.load(std::memory_order_acquire);
  const size_t read      = reader.pending.empty() ? reader.committed.load(std::memory_order_relaxed)
                                                  : reader.pending.back();
  const size_t write     = write_idx_.load(std::memory_order_acquire);
  const size_t remaining = (write + nb_slots_ - read) % nb_slots_;
  if (remaining < static_cast<size_t>(settings_.output_stride)) {
    if (!closed)
      return OpResult::NotReady;
    if (remaining && settings_.tail == "error")
      throw std::runtime_error("Incomplete final BatchQueue stride group");
    return OpResult::Eof;
  }
  auto &storage  = storage_access().owned_output_storage(group);
  storage.ptr    = buf_ + read * element_size_;
  ctx.outputs[0] = {odesc_, &storage};
  ticket         = reader.pending.push((read + settings_.output_stride) % nb_slots_);
  return OpResult::Ok;
}

void BatchQueue::release_pop(holoflow::core::AsyncTicket ticket) {
  release_output_group(0, ticket);
}

void BatchQueue::release_output_group(size_t group, holoflow::core::AsyncTicket ticket) {
  auto &reader = readers_.at(group);
  reader.committed.store(reader.pending.pop(ticket), std::memory_order_release);
}

std::optional<holoflow::core::TView> BatchQueue::acquire_input(int index) {
  if (index != 0) {
    throw std::out_of_range("BatchQueue::acquire_input: invalid index");
  }

  if (nb_slots_ - writer_size() <= input_size_) {
    return std::nullopt;
  }

  size_t     write_idx = write_idx_.load(std::memory_order_relaxed);
  std::byte *data      = buf_ + write_idx * element_size_;
  auto      &storage   = storage_access().owned_input_storage(0);
  storage.ptr          = data;

  return holoflow::core::TView{
      .desc    = idesc_,
      .storage = &storage,
  };
}

void BatchQueue::release_output(int index) {
  if (index != 0) {
    throw std::out_of_range("BatchQueue::release_output: invalid index");
  }

  size_t read_idx      = readers_[0].committed.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + settings_.output_stride;
  if (next_read_idx == nb_slots_) {
    next_read_idx = 0;
  }
  auto &storage = storage_access().owned_output_storage(0);
  storage.ptr   = nullptr;
  readers_[0].committed.store(next_read_idx, std::memory_order_release);
}

holoflow::core::OpResult BatchQueue::try_push(holoflow::core::AsyncPushCtx &) {
  size_t write_idx      = write_idx_.load(std::memory_order_relaxed);
  size_t next_write_idx = write_idx + input_size_;
  if (next_write_idx >= nb_slots_) {
    next_write_idx = 0;
  }
  auto &storage = storage_access().owned_input_storage(0);
  storage.ptr   = nullptr;
  write_idx_.store(next_write_idx, std::memory_order_release);
  return holoflow::core::OpResult::Ok;
}

holoflow::core::OpResult BatchQueue::try_pop(holoflow::core::AsyncPopCtx &ctx) {
  holoflow::core::AsyncTicket ticket;
  const auto                  result = reserve_pop(ctx, ticket);
  if (result == holoflow::core::OpResult::Ok)
    cancel_pop(ticket); // The legacy release advances the committed cursor directly.
  return result;
}

size_t BatchQueue::writer_size() const {
  return used_capacity(write_idx_.load(std::memory_order_relaxed));
}

size_t BatchQueue::used_capacity(size_t write) const {
  size_t used = 0;
  for (const auto &reader : readers_) {
    const size_t read     = reader.committed.load(std::memory_order_acquire);
    const size_t distance = write >= read ? write - read : nb_slots_ - (read - write);
    used                  = (std::max)(used, distance);
  }
  return used;
}

size_t slot_count(size_t input, size_t stride, size_t target) {
  const size_t gcd   = std::gcd(input, stride);
  const size_t limit = (std::numeric_limits<size_t>::max)();
  if (input > limit / (stride / gcd) || target > limit - input || stride - gcd >= limit - input)
    throw std::overflow_error("BatchQueue capacity overflow");
  const size_t alignment = input * (stride / gcd);
  const size_t minimum   = (std::max)(target + input, input + stride - gcd + 1);
  const size_t blocks    = (minimum - 1) / alignment + 1;
  if (blocks > limit / alignment || blocks * alignment > limit / 2)
    throw std::overflow_error("BatchQueue capacity overflow");
  return blocks * alignment;
}

size_t allocation_bytes(size_t slots, const holoflow::core::TDesc &desc) {
  if (desc.strides[0] > ((std::numeric_limits<size_t>::max)() - desc.offset) / slots)
    throw std::overflow_error("BatchQueue allocation size overflow");
  return slots * desc.strides[0] + desc.offset;
}

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

} // namespace

holoflow::core::InferResult
BatchQueueFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[BatchQueueFactory::infer] error: {}", msg);
      throw std::invalid_argument("BatchQueueFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<BatchQueueSettings>();

  // Validate
  check(settings.tail == "error" || settings.tail == "drop", "tail must be error or drop");
  check(input_descs.size() == 1, "BatchQueue task must have exactly one input");
  check(input_descs[0].rank() > 0, "BatchQueue task input must have rank > 0");
  check(input_descs[0].shape[0] > 0, "BatchQueue task input batch must be positive");
  check(is_contiguous(input_descs[0]), "BatchQueue task input must be contiguous");
  check(settings.reader_count > 0 && settings.reader_count <= INT_MAX, "invalid reader count");
  check(settings.target_capacity > 0, "BatchQueue task target capacity must be > 0");
  check(settings.output_size > 0, "BatchQueue task output size must be > 0");
  check(settings.output_stride > 0, "BatchQueue task output stride must be > 0");
  auto is_factor = settings.output_stride % settings.output_size == 0;
  check(is_factor, "BatchQueue task output stride must be a multiple of output size");

  // Success
  auto odesc     = input_descs[0];
  odesc.shape[0] = settings.output_size;
  const auto slots =
      slot_count(input_descs[0].shape[0], settings.output_stride, settings.target_capacity);
  std::vector<std::vector<size_t>> groups;
  if (settings.reader_count > 1)
    for (size_t reader = 0; reader < settings.reader_count; ++reader)
      groups.push_back({reader});
  return holoflow::core::InferResult{
      .input_descs                = {input_descs[0]},
      .output_descs               = std::vector(settings.reader_count, odesc),
      .in_place                   = {},
      .owned_inputs               = {true},
      .owned_outputs              = std::vector<bool>(settings.reader_count, true),
      .kind                       = holoflow::core::TaskKind::Async,
      .owned_input_pointer_counts = {slots / input_descs[0].shape[0]},
      .owned_output_pointer_counts =
          std::vector<std::optional<size_t>>(settings.reader_count, slots / settings.output_stride),
      .supports_prequeued_push = true,
      .supports_prequeued_pop  = true,
      .output_groups           = std::move(groups),
      .read_only_outputs       = settings.reader_count > 1,
      .reserved_pop_eof        = true,
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
BatchQueueFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings,
                          const holoflow::core::AsyncCreateCtx &) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<BatchQueueSettings>();

  const size_t input_size   = input_descs[0].shape[0];
  const size_t element_size = input_descs[0].strides[0];
  const size_t nb_slots = slot_count(input_size, settings.output_stride, settings.target_capacity);
  const size_t bytes    = allocation_bytes(nb_slots, input_descs[0]);

  logger()->debug(
      "[BatchQueueFactory::create] Creating BatchQueue with {} slots, capacity={}, input_size={}, "
      "output_size={}, output_stride={}, "
      "element_size={}, "
      "total_bytes={}",
      nb_slots, settings.target_capacity, input_size, settings.output_size, settings.output_stride,
      element_size, bytes);

  HostPtr<std::byte> h_buf = nullptr;
  DevPtr<std::byte>  d_buf = nullptr;
  std::byte         *buf   = nullptr;
  switch (input_descs[0].mem_loc) {
  case holoflow::core::MemLoc::Host:
    h_buf = curaii::make_unique_host_ptr<std::byte>(bytes);
    buf   = h_buf.get();
    break;
  case holoflow::core::MemLoc::Device:
    d_buf = curaii::make_unique_device_ptr<std::byte>(bytes);
    buf   = d_buf.get();
    break;
  }

  return std::make_unique<BatchQueue>(settings, input_descs[0], infer.output_descs[0],
                                      std::move(h_buf), std::move(d_buf), buf, nb_slots, input_size,
                                      element_size);
}

std::unique_ptr<holoflow::core::IAsyncTask>
BatchQueueFactory::update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
                          std::span<const holoflow::core::TDesc>      input_descs,
                          const nlohmann::json                       &jsettings,
                          const holoflow::core::AsyncCreateCtx       &ctx) const {
  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  auto                                 infer    = this->infer(input_descs, jsettings);
  auto                                 settings = jsettings.get<BatchQueueSettings>();
  auto                                 old_bq   = dynamic_cast<BatchQueue *>(old_task.get());
  if (old_bq == nullptr) {
    ctx.invalidate_execution();
    return create(input_descs, jsettings, ctx);
  }

  const size_t input_size   = input_descs[0].shape[0];
  const size_t element_size = input_descs[0].strides[0];
  const size_t nb_slots = slot_count(input_size, settings.output_stride, settings.target_capacity);
  const size_t bytes    = allocation_bytes(nb_slots, input_descs[0]);
  const bool   same_buffer = bytes == allocation_bytes(old_bq->nb_slots(), old_bq->idesc()) &&
                             input_descs[0].mem_loc == old_bq->idesc().mem_loc;

  if (same_buffer) {
    logger()->debug("[BatchQueueFactory::update] Reusing existing BatchQueue task");
    // Allocate/copy descriptors before transferring captured buffers: a construction failure must
    // leave them owned by old_task until the exception guard invalidates dependent executables.
    auto replacement = std::make_unique<BatchQueue>(
        settings, input_descs[0], infer.output_descs[0], HostPtr<std::byte>{}, DevPtr<std::byte>{},
        old_bq->buffer(), nb_slots, input_size, element_size);
    replacement->take_buffers_from(*old_bq);
    return replacement;
  }

  // Fallback to recreate
  logger()->debug("[BatchQueueFactory::update] Recreating BatchQueue task");
  ctx.invalidate_execution();
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::asyncs
