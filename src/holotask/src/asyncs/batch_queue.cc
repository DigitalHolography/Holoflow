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

#include <atomic>
#include <cstddef>
#include <memory>
#include <numeric>

#include "bug.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::asyncs {

void to_json(nlohmann::json &j, const BatchQueueSettings &bqs) {
  j = nlohmann::json{
      {"target_capacity", bqs.target_capacity},
      {"output_size", bqs.output_size},
      {"output_stride", bqs.output_stride},
  };
}

void from_json(const nlohmann::json &j, BatchQueueSettings &bqs) {
  j.at("target_capacity").get_to(bqs.target_capacity);
  j.at("output_size").get_to(bqs.output_size);
  j.at("output_stride").get_to(bqs.output_stride);
}

BatchQueue::BatchQueue(const BatchQueueSettings &settings, const holoflow::core::TDesc &idesc,
                       const holoflow::core::TDesc &odesc, HostPtr<std::byte> &&h_buf,
                       DevPtr<std::byte> &&d_buf, std::byte *buf, size_t nb_slots,
                       size_t input_size, size_t element_size)
    : settings_(settings), idesc_(idesc), odesc_(odesc), h_buf_(std::move(h_buf)),
      d_buf_(std::move(d_buf)), buf_(buf), nb_slots_(nb_slots), input_size_(input_size),
      element_size_(element_size) {}

std::optional<holoflow::core::TView> BatchQueue::acquire_input(int index) {
  if (index != 0) {
    throw std::out_of_range("BatchQueue::acquire_input: invalid index");
  }

  if (nb_slots_ - writer_size() <= input_size_) {
    return std::nullopt;
  }

  // if (settings_.target_capacity == 32) {
  //   logger()->debug(
  //       "[BatchQueue::acquire_input] reader_size={}, writer_size={}, target_capacity={}",
  //       reader_size(), writer_size(), settings_.target_capacity);
  // }

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

  // if (settings_.target_capacity == 32) {
  //   logger()->debug(
  //       "[BatchQueue::release_output] reader_size={}, writer_size={}, target_capacity={}",
  //       reader_size(), writer_size(), settings_.target_capacity);
  // }

  size_t read_idx      = read_idx_.load(std::memory_order_relaxed);
  size_t next_read_idx = read_idx + settings_.output_stride;
  if (next_read_idx == nb_slots_) {
    next_read_idx = 0;
  }
  auto &storage = storage_access().owned_output_storage(0);
  storage.ptr   = nullptr;
  read_idx_.store(next_read_idx, std::memory_order_release);
}

holoflow::core::OpResult BatchQueue::try_push(holoflow::core::AsyncPushCtx &) {
  // if (settings_.target_capacity == 32) {
  //   logger()->debug("[BatchQueue::try_push] reader_size={}, writer_size={}, target_capacity={}",
  //                   reader_size(), writer_size(), settings_.target_capacity);
  // }

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
  if (reader_size() < settings_.output_stride) {
    return holoflow::core::OpResult::NotReady;
  }

  // if (settings_.target_capacity == 32) {
  //   logger()->debug("[BatchQueue::try_pop] reader_size={}, writer_size={}, target_capacity={}",
  //                   reader_size(), writer_size(), settings_.target_capacity);
  // }

  size_t     read_idx = read_idx_.load(std::memory_order_relaxed);
  std::byte *data     = buf_ + read_idx * element_size_;
  auto      &storage  = storage_access().owned_output_storage(0);
  storage.ptr         = data;

  ctx.outputs[0] = holoflow::core::TView{
      .desc    = odesc_,
      .storage = &storage,
  };
  return holoflow::core::OpResult::Ok;
}

size_t BatchQueue::writer_size() const {
  size_t write_idx = write_idx_.load(std::memory_order_relaxed);
  size_t read_idx  = read_idx_.load(std::memory_order_acquire);
  size_t diff      = write_idx - read_idx;
  if (write_idx < read_idx) {
    diff += nb_slots_;
  }
  return diff;
}

size_t BatchQueue::reader_size() const {
  size_t write_idx = write_idx_.load(std::memory_order_acquire);
  size_t read_idx  = read_idx_.load(std::memory_order_relaxed);
  size_t diff      = write_idx - read_idx;
  if (write_idx < read_idx) {
    diff += nb_slots_;
  }
  return diff;
}

namespace {

int lcm_above(int x, int y, int k) {
  auto base = std::lcm(x, y);
  HOLOVIBES_CHECK(base > 0, "lcm_above: lcm overflow");
  auto mult = (k + base - 1) / base;
  return base * mult;
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
  check(input_descs.size() == 1, "BatchQueue task must have exactly one input");
  check(input_descs[0].rank() > 0, "BatchQueue task input must have rank > 0");
  check(settings.target_capacity > 0, "BatchQueue task target capacity must be > 0");
  check(settings.output_size > 0, "BatchQueue task output size must be > 0");
  check(settings.output_stride > 0, "BatchQueue task output stride must be > 0");
  auto is_factor = settings.output_stride % settings.output_size == 0;
  check(is_factor, "BatchQueue task output stride must be a multiple of output size");

  // Success
  auto odesc     = input_descs[0];
  odesc.shape[0] = settings.output_size;
  return holoflow::core::InferResult{
      .input_descs   = {input_descs[0]},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {true},
      .owned_outputs = {true},
      .kind          = holoflow::core::TaskKind::Async,
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
BatchQueueFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings,
                          const holoflow::core::AsyncCreateCtx &) const {
  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<BatchQueueSettings>();

  // Compute n_slots such that:
  // - nb_slots >= target_capacity
  // - nb_slots % stride == 0
  // - nb_slots % input_descs[0].shape[0] == 0
  int x        = static_cast<int>(input_descs[0].shape[0]);
  int y        = settings.output_stride;
  int k        = settings.target_capacity + x;
  int nb_slots = lcm_above(x, y, k);

  // Setup buffers
  size_t input_size   = static_cast<int>(input_descs[0].shape[0]);
  size_t element_size = static_cast<int>(input_descs[0].num_bytes() / input_size);
  size_t bytes        = nb_slots * element_size;

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

  // Success
  auto *task = new BatchQueue(settings, input_descs[0], infer.output_descs[0], std::move(h_buf),
                              std::move(d_buf), buf, nb_slots, input_size, element_size);
  return std::unique_ptr<holoflow::core::IAsyncTask>(task);
}

std::unique_ptr<holoflow::core::IAsyncTask>
BatchQueueFactory::update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
                          std::span<const holoflow::core::TDesc>      input_descs,
                          const nlohmann::json                       &jsettings,
                          const holoflow::core::AsyncCreateCtx       &ctx) const {
  // Validate
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<BatchQueueSettings>();
  auto old_bq   = dynamic_cast<BatchQueue *>(old_task.get());
  HOLOVIBES_CHECK(old_bq != nullptr, "old_task is not a BatchQueue instance");

  // Update
  int    x            = static_cast<int>(input_descs[0].shape[0]);
  int    y            = settings.output_stride;
  int    k            = settings.target_capacity + x;
  int    nb_slots     = lcm_above(x, y, k);
  size_t input_size   = static_cast<int>(input_descs[0].shape[0]);
  size_t element_size = static_cast<int>(input_descs[0].num_bytes() / input_size);
  size_t bytes        = nb_slots * element_size;
  bool   same_buffer  = (bytes == old_bq->nb_slots_ * old_bq->element_size_) &&
                     (input_descs[0].mem_loc == old_bq->idesc_.mem_loc);

  if (same_buffer) {
    logger()->debug("[BatchQueueFactory::update] Reusing existing BatchQueue task");
    auto  h_buf = std::move(old_bq->h_buf_);
    auto  d_buf = std::move(old_bq->d_buf_);
    auto  buf   = old_bq->buf_;
    auto *task  = new BatchQueue(settings, input_descs[0], infer.output_descs[0], std::move(h_buf),
                                 std::move(d_buf), buf, nb_slots, input_size, element_size);
    return std::unique_ptr<holoflow::core::IAsyncTask>(task);
  }

  // Fallback to recreate
  logger()->debug("[BatchQueueFactory::update] Recreating BatchQueue task");
  return this->create(input_descs, jsettings, ctx);
}

} // namespace holotask::asyncs