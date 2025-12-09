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

#include <atomic>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::asyncs {

struct SlidingAverageSettings {
  size_t target_capacity;
  size_t window_size;
};

/// @name JSON serialization
/// @{
void to_json(nlohmann::json &j, const SlidingAverageSettings &s);
void from_json(const nlohmann::json &j, SlidingAverageSettings &s);
/// @}

class SlidingAverage : public holoflow::core::IAsyncTask {
public:
  ~SlidingAverage() override = default;

  holoflow::core::OpResult try_push(holoflow::core::AsyncPushCtx &ctx) override;
  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &ctx) override;

  std::optional<holoflow::core::TView> acquire_input(int index) override;

  void release_output(int index) override;

private:
  SlidingAverage(SlidingAverageSettings settings, const holoflow::core::TDesc &idesc,
                 holoflow::core::TDesc &odesc, cudaStream_t producer_stream,
                 cudaStream_t consumer_stream, size_t nb_slots, size_t element_size,
                 DevPtr<std::byte> &&d_buffer, DevPtr<float> &&d_running_avg);

  int writer_size() const;
  int reader_size() const;

  friend class SlidingAverageFactory;

  // Settings
  SlidingAverageSettings settings_;

  holoflow::core::TDesc idesc_;
  holoflow::core::TDesc odesc_;

  // CUDA streams
  cudaStream_t producer_stream_;
  cudaStream_t consumer_stream_;

  // Buffer dimensions
  size_t nb_slots_;
  size_t element_size_;

  // Device buffers
  DevPtr<std::byte> d_buffer_;      ///< Circular buffer storage
  DevPtr<float>     d_running_avg_; ///< Running sum buffer

  alignas(CACHE_LINE_SIZE) std::atomic<size_t> avg_idx_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

/// @brief Factory for creating SlidingAverage tasks
class SlidingAverageFactory : public holoflow::core::IAsyncTaskFactory {
public:
  ~SlidingAverageFactory() override = default;

  /// @brief Infer task metadata and requirements
  /// @param input_descs Input tensor descriptors
  /// @param jsettings JSON configuration
  /// @return Inference result with I/O descriptions
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  /// @brief Create a new SlidingAverage task instance
  /// @param input_descs Input tensor descriptors
  /// @param jsettings JSON configuration
  /// @param ctx Async creation context with streams
  /// @return New task instance
  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;

  /// @brief Update an existing task with new configuration
  /// @param old_task Existing task to update
  /// @param input_descs New input descriptors
  /// @param jsettings New configuration
  /// @param ctx Async creation context
  /// @return Updated task instance
  std::unique_ptr<holoflow::core::IAsyncTask>
  update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::AsyncCreateCtx &ctx) const override;
};

} // namespace holotask::asyncs