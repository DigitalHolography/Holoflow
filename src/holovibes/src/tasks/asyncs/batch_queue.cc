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

#include "batch_queue.hh"

#include <atomic>
#include <cstddef>
#include <memory>
#include <numeric>

#include "bug.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holovibes::tasks::asyncs {

// -------------------------------------------------------------------------------------------------
// JSON Serialization
// -------------------------------------------------------------------------------------------------

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

// -------------------------------------------------------------------------------------------------
// BatchQueue Implementation
// -------------------------------------------------------------------------------------------------

BatchQueue::BatchQueue(const BatchQueueSettings    &settings,
                       const holoflow::core::TDesc &idesc,
                       const holoflow::core::TDesc &odesc,
                       HostPtr<std::byte>         &&h_buf,
                       DevPtr<std::byte>          &&d_buf,
                       std::byte                   *buf,
                       size_t                       nb_slots,
                       size_t                       input_size,
                       size_t                       element_size) :
    settings_(settings),
    idesc_(idesc),
    odesc_(odesc),
    h_buffer_(std::move(h_buf)),
    d_buffer_(std::move(d_buf)),
    buffer_(buf),
    nb_slots_(nb_slots),
    input_size_(input_size),
    element_size_(element_size) {}

size_t BatchQueue::writer_size() const {
  const size_t w = write_idx_.load(std::memory_order_relaxed);
  const size_t r = read_idx_.load(std::memory_order_acquire);
  return (w >= r) ? (w - r) : (w - r + nb_slots_);
}

size_t BatchQueue::reader_size() const {
  const size_t w = write_idx_.load(std::memory_order_acquire);
  const size_t r = read_idx_.load(std::memory_order_relaxed);
  return (w >= r) ? (w - r) : (w - r + nb_slots_);
}

std::byte *BatchQueue::get_slot_ptr(size_t index) const {
  return buffer_ + (index * element_size_);
}

size_t BatchQueue::next_write_index(size_t current_index) const {
  size_t next = current_index + input_size_;
  return (next == nb_slots_) ? 0 : next;
}

size_t BatchQueue::next_read_index(size_t current_index) const {
  size_t next = current_index + settings_.output_stride;
  return (next == nb_slots_) ? 0 : next;
}

std::optional<holoflow::core::TView> BatchQueue::acquire_input(int index) {
  if (index != 0) {
    throw std::out_of_range("BatchQueue::acquire_input: invalid index");
  }

  // Ensure we have space (at least batch size slots free)
  if (nb_slots_ - writer_size() <= input_size_) {
    return std::nullopt;
  }

  const size_t idx = write_idx_.load(std::memory_order_relaxed);

  return holoflow::core::TView{
      .data = get_slot_ptr(idx),
      .desc = idesc_,
  };
}

void BatchQueue::release_output(int index) {
  if (index != 0) {
    throw std::out_of_range("BatchQueue::release_output: invalid index");
  }

  const size_t idx = read_idx_.load(std::memory_order_relaxed);
  read_idx_.store(next_read_index(idx), std::memory_order_release);
}

holoflow::core::OpResult BatchQueue::try_push(holoflow::core::AsyncPushCtx &) {
  const size_t idx = write_idx_.load(std::memory_order_relaxed);
  write_idx_.store(next_write_index(idx), std::memory_order_release);
  return holoflow::core::OpResult::Ok;
}

holoflow::core::OpResult BatchQueue::try_pop(holoflow::core::AsyncPopCtx &ctx) {
  // Ensure we have data to read (at least output stride slots)
  if (reader_size() < settings_.output_stride) {
    return holoflow::core::OpResult::NotReady;
  }

  const size_t idx = read_idx_.load(std::memory_order_relaxed);

  ctx.outputs[0] = holoflow::core::TView{
      .data = get_slot_ptr(idx),
      .desc = odesc_,
  };

  return holoflow::core::OpResult::Ok;
}

namespace {

/**
 * @brief Calculates the Least Common Multiple of x and y, then finds the smallest
 * multiple of that base LCM that is greater than or equal to k.
 */
size_t lcm_above(size_t x, size_t y, size_t k) {
  HOLOVIBES_CHECK(x > 0 && y > 0, "lcm_above: x and y must be positive integers");

  const size_t base = std::lcm(x, y);
  HOLOVIBES_CHECK(base > 0, "lcm_above: lcm overflow");

  // Integer ceiling division: (k + base - 1) / base
  const size_t mult = (k + base - 1) / base;
  return base * mult;
}
/**
 * @brief Struct to hold pre-calculated buffer parameters to avoid logic duplication.
 */
struct BufferParams {
  size_t nb_slots;
  size_t input_size;
  size_t element_size;
  size_t total_bytes;
};

BufferParams calculate_buffer_params(const holoflow::core::TDesc &input_desc,
                                     const BatchQueueSettings    &settings) {
  const size_t x = static_cast<size_t>(input_desc.shape[0]);
  const size_t y = static_cast<size_t>(settings.output_stride);
  const size_t k = static_cast<size_t>(settings.target_capacity) + x;

  const size_t nb_slots     = lcm_above(x, y, k);
  const size_t input_size   = x;
  const size_t element_size = input_desc.num_bytes() / input_size;
  const size_t total_bytes  = nb_slots * element_size;

  return {nb_slots, input_size, element_size, total_bytes};
}

} // namespace

// -------------------------------------------------------------------------------------------------
// BatchQueueFactory Implementation
// -------------------------------------------------------------------------------------------------

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
  check(input_descs.size() == 1, "Task must have exactly one input");
  check(input_descs[0].rank() > 0, "Input must have rank > 0");
  check(settings.target_capacity > 0, "Target capacity must be > 0");
  check(settings.output_size > 0, "Output size must be > 0");
  check(settings.output_stride > 0, "Output stride must be > 0");
  check(settings.output_stride % settings.output_size == 0,
        "Output stride must be a multiple of output size");

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
  auto        infer    = this->infer(input_descs, jsettings);
  auto        settings = jsettings.get<BatchQueueSettings>();
  const auto &idesc    = input_descs[0];
  const auto &odesc    = infer.output_descs[0];
  const auto  params   = calculate_buffer_params(idesc, settings);

  HostPtr<std::byte> h_buf = nullptr;
  DevPtr<std::byte>  d_buf = nullptr;
  std::byte         *buf   = nullptr;

  // Allocate buffer based on memory location
  switch (idesc.mem_loc) {
  case holoflow::core::MemLoc::Host:
    h_buf = curaii::make_unique_host_ptr<std::byte>(params.total_bytes);
    buf   = h_buf.get();
    break;
  case holoflow::core::MemLoc::Device:
    d_buf = curaii::make_unique_device_ptr<std::byte>(params.total_bytes);
    buf   = d_buf.get();
    break;
  }

  // Success
  auto *task = new BatchQueue(settings,
                              idesc,
                              odesc,
                              std::move(h_buf),
                              std::move(d_buf),
                              buf,
                              params.nb_slots,
                              params.input_size,
                              params.element_size);
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

  const auto &idesc  = input_descs[0];
  const auto &odesc  = infer.output_descs[0];
  const auto  params = calculate_buffer_params(idesc, settings);

  // Check if we can reuse existing buffers
  // Reuse requires exact size match and same memory location
  const bool size_match = params.total_bytes == (old_bq->nb_slots_ * old_bq->element_size_);
  const bool same_loc   = idesc.mem_loc == old_bq->idesc_.mem_loc;

  if (size_match && same_loc) {
    logger()->debug("[BatchQueueFactory::update] Reusing existing BatchQueue task");

    auto *task = new BatchQueue(settings,
                                idesc,
                                odesc,
                                std::move(old_bq->h_buffer_),
                                std::move(old_bq->d_buffer_),
                                old_bq->buffer_,
                                params.nb_slots,
                                params.input_size,
                                params.element_size);
    return std::unique_ptr<holoflow::core::IAsyncTask>(task);
  }

  // Fallback to recreate
  logger()->debug("[BatchQueueFactory::update] Recreating BatchQueue task");
  return this->create(input_descs, jsettings, ctx);
}

} // namespace holovibes::tasks::asyncs