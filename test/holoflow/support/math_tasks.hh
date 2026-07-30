// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "holoflow/core/registry.hh"

namespace holoflow::test {

using core::DType;
using core::InferResult;
using core::MemLoc;
using core::OpResult;
using core::TaskKind;
using core::TDesc;

struct MathState {
  std::mutex                      mutex;
  std::vector<float>              collected;
  std::vector<std::vector<float>> collected_frames;
  std::atomic<int>                source_calls{0};
  std::atomic<int>                add_calls{0};
  std::atomic<int>                scale_calls{0};
  std::atomic<int>                sink_calls{0};
  std::atomic<int>                async_push_calls{0};
  std::atomic<int>                async_pop_calls{0};
  std::atomic<int>                async_push_not_ready{0};
  std::atomic<int>                async_pop_not_ready{0};
  cudaStream_t                    last_sync_stream = nullptr;
  cudaStream_t                    producer_stream  = nullptr;
  cudaStream_t                    consumer_stream  = nullptr;
};

inline std::span<float> floats(core::TView &view) {
  return {reinterpret_cast<float *>(view.data()), view.desc.num_elements()};
}

inline std::span<const float> floats(const core::TView &view) {
  auto &mutable_view = const_cast<core::TView &>(view);
  return {reinterpret_cast<const float *>(mutable_view.data()), view.desc.num_elements()};
}

class VectorSourceTask final : public core::ISyncTask {
public:
  VectorSourceTask(std::vector<float> values, std::shared_ptr<MathState> state)
      : values_(std::move(values)), state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    if (ctx.outputs.size() != 1) {
      throw std::invalid_argument("source requires one output");
    }
    auto output = floats(ctx.outputs[0]);
    std::copy(values_.begin(), values_.end(), output.begin());
    ++state_->source_calls;
    return OpResult::Ok;
  }

private:
  std::vector<float>         values_;
  std::shared_ptr<MathState> state_;
};

class SequenceSourceTask final : public core::ISyncTask {
public:
  SequenceSourceTask(std::vector<float> base, std::shared_ptr<MathState> state)
      : base_(std::move(base)), state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    auto       output = floats(ctx.outputs[0]);
    const auto frame  = state_->source_calls.fetch_add(1);
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = base_[i] + static_cast<float>(frame);
    }
    return OpResult::Ok;
  }

private:
  std::vector<float>         base_;
  std::shared_ptr<MathState> state_;
};

class AddTask final : public core::ISyncTask {
public:
  explicit AddTask(std::shared_ptr<MathState> state) : state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    auto lhs = floats(ctx.inputs[0]);
    auto rhs = floats(ctx.inputs[1]);
    auto out = floats(ctx.outputs[0]);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = lhs[i] + rhs[i];
    }
    ++state_->add_calls;
    return OpResult::Ok;
  }

private:
  std::shared_ptr<MathState> state_;
};

class ScaleTask final : public core::ISyncTask {
public:
  ScaleTask(float factor, std::shared_ptr<MathState> state)
      : factor_(factor), state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    auto input = floats(ctx.inputs[0]);
    auto out   = floats(ctx.outputs[0]);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = input[i] * factor_;
    }
    ++state_->scale_calls;
    return OpResult::Ok;
  }

private:
  float                      factor_;
  std::shared_ptr<MathState> state_;
};

class CollectTask final : public core::ISyncTask {
public:
  explicit CollectTask(std::shared_ptr<MathState> state) : state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    const auto input = floats(ctx.inputs[0]);
    {
      std::lock_guard lock(state_->mutex);
      state_->collected.assign(input.begin(), input.end());
    }
    ++state_->sink_calls;
    return OpResult::Eof;
  }

private:
  std::shared_ptr<MathState> state_;
};

class HistoryCollectTask final : public core::ISyncTask {
public:
  HistoryCollectTask(size_t frame_count, std::shared_ptr<MathState> state)
      : frame_count_(frame_count), state_(std::move(state)) {}

  OpResult execute(core::SyncCtx &ctx) override {
    const auto input = floats(ctx.inputs[0]);
    size_t     collected_count;
    {
      std::lock_guard lock(state_->mutex);
      state_->collected_frames.emplace_back(input.begin(), input.end());
      collected_count = state_->collected_frames.size();
    }
    ++state_->sink_calls;
    if (frame_count_ != 0 && collected_count >= frame_count_) {
      return OpResult::Eof;
    }
    return OpResult::Ok;
  }

private:
  size_t                     frame_count_;
  std::shared_ptr<MathState> state_;
};

class AsyncBridgeTask final : public core::IAsyncTask {
public:
  AsyncBridgeTask(std::shared_ptr<MathState> state, uint32_t jitter_seed = 0,
                  uint32_t max_yields = 0, uint32_t forced_push_not_ready = 0,
                  uint32_t forced_pop_not_ready = 0)
      : state_(std::move(state)), jitter_seed_(jitter_seed), max_yields_(max_yields),
        forced_push_not_ready_(forced_push_not_ready), forced_pop_not_ready_(forced_pop_not_ready) {
  }

  OpResult try_push(core::AsyncPushCtx &ctx) override {
    ++state_->async_push_calls;
    jitter(state_->async_push_calls.load());
    if (forced_push_not_ready_ > 0) {
      --forced_push_not_ready_;
      ++state_->async_push_not_ready;
      return OpResult::NotReady;
    }
    std::lock_guard lock(mutex_);
    if (pending_) {
      ++state_->async_push_not_ready;
      return OpResult::NotReady;
    }
    const auto input = floats(ctx.inputs[0]);
    pending_.emplace(input.begin(), input.end());
    return OpResult::Ok;
  }

  OpResult try_pop(core::AsyncPopCtx &ctx) override {
    ++state_->async_pop_calls;
    jitter(state_->async_pop_calls.load());
    if (forced_pop_not_ready_ > 0) {
      --forced_pop_not_ready_;
      ++state_->async_pop_not_ready;
      return OpResult::NotReady;
    }
    std::lock_guard lock(mutex_);
    if (!pending_) {
      ++state_->async_pop_not_ready;
      return OpResult::NotReady;
    }
    auto output = floats(ctx.outputs[0]);
    std::copy(pending_->begin(), pending_->end(), output.begin());
    pending_.reset();
    return OpResult::Ok;
  }

private:
  void jitter(uint32_t call) const {
    if (max_yields_ == 0) {
      return;
    }
    auto value = jitter_seed_ ^ (call * 1664525U + 1013904223U);
    for (uint32_t i = 0; i < value % (max_yields_ + 1); ++i) {
      std::this_thread::yield();
    }
  }

  std::shared_ptr<MathState>        state_;
  uint32_t                          jitter_seed_;
  uint32_t                          max_yields_;
  uint32_t                          forced_push_not_ready_;
  uint32_t                          forced_pop_not_ready_;
  std::mutex                        mutex_;
  std::optional<std::vector<float>> pending_;
};

class VectorSourceFactory final : public core::ISyncTaskFactory {
public:
  VectorSourceFactory(std::vector<float> values, std::shared_ptr<MathState> state,
                      MemLoc mem_loc = MemLoc::Host)
      : values_(std::move(values)), state_(std::move(state)), mem_loc_(mem_loc) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("source has no inputs");
    }
    return {{}, {TDesc({values_.size()}, DType::F32, mem_loc_)}, {}, {}, {false}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &ctx) const override {
    state_->last_sync_stream = ctx.stream;
    return std::make_unique<VectorSourceTask>(values_, state_);
  }

private:
  std::vector<float>         values_;
  std::shared_ptr<MathState> state_;
  MemLoc                     mem_loc_;
};

class SequenceSourceFactory final : public core::ISyncTaskFactory {
public:
  SequenceSourceFactory(std::vector<float> base, std::shared_ptr<MathState> state)
      : base_(std::move(base)), state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("sequence source has no inputs");
    }
    return {{}, {TDesc({base_.size()}, DType::F32, MemLoc::Host)}, {}, {}, {false}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &) const override {
    return std::make_unique<SequenceSourceTask>(base_, state_);
  }

private:
  std::vector<float>         base_;
  std::shared_ptr<MathState> state_;
};

class AddFactory final : public core::ISyncTaskFactory {
public:
  explicit AddFactory(std::shared_ptr<MathState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 2 || inputs[0].shape != inputs[1].shape ||
        inputs[0].mem_loc != inputs[1].mem_loc) {
      throw std::invalid_argument("add requires matching inputs");
    }
    return {
        {inputs.begin(), inputs.end()}, {inputs[0]}, {}, {false, false}, {false}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &ctx) const override {
    state_->last_sync_stream = ctx.stream;
    return std::make_unique<AddTask>(state_);
  }

private:
  std::shared_ptr<MathState> state_;
};

class ScaleFactory final : public core::ISyncTaskFactory {
public:
  ScaleFactory(float factor, std::shared_ptr<MathState> state)
      : factor_(factor), state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("scale requires one input");
    }
    return {{inputs[0]}, {inputs[0]}, {}, {false}, {false}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &ctx) const override {
    state_->last_sync_stream = ctx.stream;
    return std::make_unique<ScaleTask>(factor_, state_);
  }

private:
  float                      factor_;
  std::shared_ptr<MathState> state_;
};

class CollectFactory final : public core::ISyncTaskFactory {
public:
  explicit CollectFactory(std::shared_ptr<MathState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("sink requires one input");
    }
    return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &ctx) const override {
    state_->last_sync_stream = ctx.stream;
    return std::make_unique<CollectTask>(state_);
  }

private:
  std::shared_ptr<MathState> state_;
};

class HistoryCollectFactory final : public core::ISyncTaskFactory {
public:
  HistoryCollectFactory(size_t frame_count, std::shared_ptr<MathState> state)
      : frame_count_(frame_count), state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("history sink requires one input");
    }
    return {{inputs[0]}, {}, {}, {false}, {}, TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                          const core::SyncCreateCtx &) const override {
    return std::make_unique<HistoryCollectTask>(frame_count_, state_);
  }

private:
  size_t                     frame_count_;
  std::shared_ptr<MathState> state_;
};

class AsyncBridgeFactory final : public core::IAsyncTaskFactory {
public:
  explicit AsyncBridgeFactory(std::shared_ptr<MathState> state, uint32_t jitter_seed = 0,
                              uint32_t max_yields = 0, uint32_t forced_push_not_ready = 0,
                              uint32_t forced_pop_not_ready = 0)
      : state_(std::move(state)), jitter_seed_(jitter_seed), max_yields_(max_yields),
        forced_push_not_ready_(forced_push_not_ready), forced_pop_not_ready_(forced_pop_not_ready) {
  }

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("bridge requires one input");
    }
    return {{inputs[0]}, {inputs[0]}, {}, {false}, {false}, TaskKind::Async};
  }

  std::unique_ptr<core::IAsyncTask> create(std::span<const TDesc>, const nlohmann::json &,
                                           const core::AsyncCreateCtx &ctx) const override {
    state_->producer_stream = ctx.producer_stream;
    state_->consumer_stream = ctx.consumer_stream;
    return std::make_unique<AsyncBridgeTask>(state_, jitter_seed_, max_yields_,
                                             forced_push_not_ready_, forced_pop_not_ready_);
  }

private:
  std::shared_ptr<MathState> state_;
  uint32_t                   jitter_seed_;
  uint32_t                   max_yields_;
  uint32_t                   forced_push_not_ready_;
  uint32_t                   forced_pop_not_ready_;
};

} // namespace holoflow::test
