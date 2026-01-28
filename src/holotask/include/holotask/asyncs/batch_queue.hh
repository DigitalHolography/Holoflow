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

#pragma once

#include <nlohmann/json.hpp>
#include <optional>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#if defined(_MSC_VER)
#if !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
#endif

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holotask::asyncs {

struct BatchQueueSettings {
  int target_capacity; // Target capacity of the queue
  int output_size;     // Size of each output batch
  int output_stride;   // Stride between batches
  int nb_readers;      // Number of concurrent readers
};

struct alignas(CACHE_LINE_SIZE) AlignedAtomicSizeT {
  std::atomic<size_t> value;
};

/// @name JSON serialization
/// @brief JSON serialization for @ref BatchQueueSettings.
/// @{
void to_json(nlohmann::json &j, const BatchQueueSettings &bqs);
void from_json(const nlohmann::json &j, BatchQueueSettings &bqs);
/// @}

class BatchQueue : public holoflow::core::IAsyncTask {
public:
  std::optional<holoflow::core::TView> acquire_input(int index) override;
  void                                 release_output(int index) override;
  holoflow::core::OpResult             try_push(holoflow::core::AsyncPushCtx &ctx) override;
  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &ctx, size_t idx) override;

private:
  BatchQueue(const BatchQueueSettings &settings, const holoflow::core::TDesc &idesc,
             const std::vector<holoflow::core::TDesc> &odesc, HostPtr<std::byte> &&h_buf,
             DevPtr<std::byte> &&d_buf, std::byte *buf, size_t nb_slots, size_t input_size,
             size_t element_size);

  size_t writer_size() const;
  size_t reader_size(size_t idx) const;

  friend class BatchQueueFactory;

  BatchQueueSettings                 settings_;
  holoflow::core::TDesc              idesc_;
  std::vector<holoflow::core::TDesc> odesc_;
  HostPtr<std::byte>                 h_buf_;
  DevPtr<std::byte>                  d_buf_;
  std::byte                         *buf_;
  size_t                             nb_slots_;
  size_t                             input_size_;
  size_t                             element_size_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;
  std::vector<AlignedAtomicSizeT> read_idx_vector_;
};

class BatchQueueFactory : public holoflow::core::IAsyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::IAsyncTask>
  update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;
};

} // namespace holotask::asyncs

#if defined(_MSC_VER)
#if !defined(__clang__)
#pragma warning(pop)
#endif
#endif