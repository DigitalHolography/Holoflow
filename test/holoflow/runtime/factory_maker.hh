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

#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holoflow::test {

inline core::TDesc make_desc(std::initializer_list<size_t> shape = {1},
                             core::DType dtype                           = core::DType::U8,
                             core::MemLoc mem_loc                        = core::MemLoc::Host) {
  return core::TDesc{std::vector<size_t>(shape), dtype, mem_loc};
}

inline core::InferResult make_infer_result(core::TaskKind kind,
                                           std::vector<core::TDesc> inputs,
                                           std::vector<core::TDesc> outputs,
                                           std::vector<bool> owned_inputs  = {},
                                           std::vector<bool> owned_outputs = {},
                                           std::vector<core::InPlace> inplace = {}) {
  core::InferResult result;
  result.kind         = kind;
  result.input_descs  = std::move(inputs);
  result.output_descs = std::move(outputs);
  result.in_place     = std::move(inplace);

  if (owned_inputs.empty()) {
    result.owned_inputs.assign(result.input_descs.size(), false);
  } else {
    result.owned_inputs = std::move(owned_inputs);
  }

  if (owned_outputs.empty()) {
    result.owned_outputs.assign(result.output_descs.size(), false);
  } else {
    result.owned_outputs = std::move(owned_outputs);
  }

  return result;
}

class StubSyncTask : public core::ISyncTask {
public:
  explicit StubSyncTask(core::SyncCreateCtx ctx) : ctx_(ctx) {}

  const core::SyncCreateCtx &ctx() const noexcept { return ctx_; }

  std::optional<core::TView> acquire_input(int) override { return std::nullopt; }
  void release_output(int) override {}
  core::OpResult execute(core::SyncCtx &) override { return core::OpResult::Ok; }

private:
  core::SyncCreateCtx ctx_;
};

class StubAsyncTask : public core::IAsyncTask {
public:
  explicit StubAsyncTask(core::AsyncCreateCtx ctx) : ctx_(ctx) {}

  const core::AsyncCreateCtx &ctx() const noexcept { return ctx_; }

  std::optional<core::TView> acquire_input(int) override { return std::nullopt; }
  void release_output(int) override {}
  core::OpResult try_push(core::AsyncPushCtx &) override { return core::OpResult::Ok; }
  core::OpResult try_pop(core::AsyncPopCtx &) override { return core::OpResult::Ok; }

private:
  core::AsyncCreateCtx ctx_;
};

struct SyncFactorySpec {
  std::function<core::InferResult(std::span<const core::TDesc>, const nlohmann::json &)> infer;
  std::function<std::unique_ptr<core::ISyncTask>(std::span<const core::TDesc>, const nlohmann::json &,
                                                 const core::SyncCreateCtx &)> create;
  std::function<std::unique_ptr<core::ISyncTask>(std::unique_ptr<core::ISyncTask>,
                                                 std::span<const core::TDesc>, const nlohmann::json &,
                                                 const core::SyncCreateCtx &)> update;
};

struct AsyncFactorySpec {
  std::function<core::InferResult(std::span<const core::TDesc>, const nlohmann::json &)> infer;
  std::function<std::unique_ptr<core::IAsyncTask>(std::span<const core::TDesc>, const nlohmann::json &,
                                                  const core::AsyncCreateCtx &)> create;
  std::function<std::unique_ptr<core::IAsyncTask>(std::unique_ptr<core::IAsyncTask>,
                                                  std::span<const core::TDesc>, const nlohmann::json &,
                                                  const core::AsyncCreateCtx &)> update;
};

class RecordingSyncFactory : public core::ISyncTaskFactory {
public:
  struct InferCall {
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
  };

  struct CreateCall {
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
    core::SyncCreateCtx      ctx;
  };

  struct UpdateCall {
    core::ISyncTask         *previous;
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
    core::SyncCreateCtx      ctx;
  };

  explicit RecordingSyncFactory(SyncFactorySpec spec = {}) : spec_(std::move(spec)) {}

  const std::vector<InferCall> &infer_calls() const noexcept { return infer_calls_; }
  const std::vector<CreateCall> &create_calls() const noexcept { return create_calls_; }
  const std::vector<UpdateCall> &update_calls() const noexcept { return update_calls_; }

  core::InferResult infer(std::span<const core::TDesc> input_descs,
                          const nlohmann::json &settings) const override {
    infer_calls_.push_back({{input_descs.begin(), input_descs.end()}, settings});
    if (spec_.infer) {
      return spec_.infer(input_descs, settings);
    }
    auto copied = std::vector<core::TDesc>(input_descs.begin(), input_descs.end());
    return make_infer_result(core::TaskKind::Sync, std::move(copied), {make_desc()});
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc> input_descs,
                                          const nlohmann::json &settings,
                                          const core::SyncCreateCtx &ctx) const override {
    create_calls_.push_back({{input_descs.begin(), input_descs.end()}, settings, ctx});
    if (spec_.create) {
      return spec_.create(input_descs, settings, ctx);
    }
    return std::make_unique<StubSyncTask>(ctx);
  }

  std::unique_ptr<core::ISyncTask> update(std::unique_ptr<core::ISyncTask> old_task,
                                          std::span<const core::TDesc> input_descs,
                                          const nlohmann::json &settings,
                                          const core::SyncCreateCtx &ctx) const override {
    update_calls_.push_back({old_task.get(), {input_descs.begin(), input_descs.end()}, settings, ctx});
    if (spec_.update) {
      return spec_.update(std::move(old_task), input_descs, settings, ctx);
    }
    if (spec_.create) {
      return spec_.create(input_descs, settings, ctx);
    }
    return std::make_unique<StubSyncTask>(ctx);
  }

private:
  mutable std::vector<InferCall>  infer_calls_;
  mutable std::vector<CreateCall> create_calls_;
  mutable std::vector<UpdateCall> update_calls_;
  SyncFactorySpec                 spec_;
};

class RecordingAsyncFactory : public core::IAsyncTaskFactory {
public:
  struct InferCall {
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
  };

  struct CreateCall {
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
    core::AsyncCreateCtx     ctx;
  };

  struct UpdateCall {
    core::IAsyncTask        *previous;
    std::vector<core::TDesc> input_descs;
    nlohmann::json           settings;
    core::AsyncCreateCtx     ctx;
  };

  explicit RecordingAsyncFactory(AsyncFactorySpec spec = {}) : spec_(std::move(spec)) {}

  const std::vector<InferCall> &infer_calls() const noexcept { return infer_calls_; }
  const std::vector<CreateCall> &create_calls() const noexcept { return create_calls_; }
  const std::vector<UpdateCall> &update_calls() const noexcept { return update_calls_; }

  core::InferResult infer(std::span<const core::TDesc> input_descs,
                          const nlohmann::json &settings) const override {
    infer_calls_.push_back({{input_descs.begin(), input_descs.end()}, settings});
    if (spec_.infer) {
      return spec_.infer(input_descs, settings);
    }
    auto copied = std::vector<core::TDesc>(input_descs.begin(), input_descs.end());
    return make_infer_result(core::TaskKind::Async, std::move(copied), {make_desc()});
  }

  std::unique_ptr<core::IAsyncTask> create(std::span<const core::TDesc> input_descs,
                                           const nlohmann::json &settings,
                                           const core::AsyncCreateCtx &ctx) const override {
    create_calls_.push_back({{input_descs.begin(), input_descs.end()}, settings, ctx});
    if (spec_.create) {
      return spec_.create(input_descs, settings, ctx);
    }
    return std::make_unique<StubAsyncTask>(ctx);
  }

  std::unique_ptr<core::IAsyncTask> update(std::unique_ptr<core::IAsyncTask> old_task,
                                           std::span<const core::TDesc> input_descs,
                                           const nlohmann::json &settings,
                                           const core::AsyncCreateCtx &ctx) const override {
    update_calls_.push_back({old_task.get(), {input_descs.begin(), input_descs.end()}, settings, ctx});
    if (spec_.update) {
      return spec_.update(std::move(old_task), input_descs, settings, ctx);
    }
    if (spec_.create) {
      return spec_.create(input_descs, settings, ctx);
    }
    return std::make_unique<StubAsyncTask>(ctx);
  }

private:
  mutable std::vector<InferCall>  infer_calls_;
  mutable std::vector<CreateCall> create_calls_;
  mutable std::vector<UpdateCall> update_calls_;
  AsyncFactorySpec                spec_;
};

} // namespace holoflow::test
