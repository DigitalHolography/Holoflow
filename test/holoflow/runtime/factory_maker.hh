#pragma once
#include "holoflow/core/tasks.hh"

namespace holoflow::test {

struct DummyTaskConfig {
  int num_inputs  = 1;
  int num_outputs = 1;
  bool inplace    = false;
  bool owned_inputs = false;
  bool owned_outputs = false;
  bool always_throw = false;
  core::TaskKind kind = core::TaskKind::Sync;
};

/// Generic dummy sync task
class DummySyncTask : public core::ISyncTask {
public:
  // Linker requires these overrides
  std::optional<core::TView> acquire_input(int) override {
    return std::nullopt; // no-op for tests
  }
  void release_output(int) override {
  }

  core::OpResult execute(core::SyncCtx &) override {
    return core::OpResult::Ok;
  }
};

/// Generic dummy async task
class DummyAsyncTask : public core::IAsyncTask {
public:
  std::optional<core::TView> acquire_input(int) override {
    return std::nullopt;
  }
  void release_output(int) override {
  }

  core::OpResult try_push(core::AsyncPushCtx &) override { return core::OpResult::Ok; }
  core::OpResult try_pop(core::AsyncPopCtx &) override { return core::OpResult::Ok; }
};

/// Meta-factory that can act as either ISyncTaskFactory or IAsyncTaskFactory
template <typename BaseFactory>
class DummyTaskFactory;

template <>
class DummyTaskFactory<core::ISyncTaskFactory> : public core::ISyncTaskFactory {
public:
  explicit DummyTaskFactory(DummyTaskConfig cfg = {}) : cfg_(cfg) {}

  core::InferResult infer(std::span<const core::TDesc> input_descs,
                          const nlohmann::json &) const override {
    if (cfg_.always_throw) {
      throw std::invalid_argument("DummyTaskFactory configured to throw");
    }
    core::InferResult ir;
    ir.kind = core::TaskKind::Sync;

    // Inputs
    ir.input_descs = input_descs.empty() ? std::vector<core::TDesc>(cfg_.num_inputs) 
                                         : std::vector<core::TDesc>(input_descs.begin(), input_descs.end());
    ir.owned_inputs.assign(ir.input_descs.size(), cfg_.owned_inputs);

    // Outputs
    ir.output_descs = std::vector<core::TDesc>(cfg_.num_outputs);
    ir.owned_outputs.assign(ir.output_descs.size(), cfg_.owned_outputs);

    if (cfg_.inplace && !ir.input_descs.empty() && !ir.output_descs.empty()) {
      ir.in_place.push_back({0, 0});
    }
    return ir;
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc>,
                                          const nlohmann::json &,
                                          const core::SyncCreateCtx &) const override {
    return std::make_unique<DummySyncTask>();
  }

private:
  DummyTaskConfig cfg_;
};

template <>
class DummyTaskFactory<core::IAsyncTaskFactory> : public core::IAsyncTaskFactory {
public:
  explicit DummyTaskFactory(DummyTaskConfig cfg = {}) : cfg_(cfg) {}

  core::InferResult infer(std::span<const core::TDesc> input_descs,
                          const nlohmann::json &) const override {
    if (cfg_.always_throw) {
      throw std::invalid_argument("DummyTaskFactory configured to throw");
    }
    core::InferResult ir;
    ir.kind = core::TaskKind::Async;

    // Inputs
    ir.input_descs = input_descs.empty() ? std::vector<core::TDesc>(cfg_.num_inputs) 
                                         : std::vector<core::TDesc>(input_descs.begin(), input_descs.end());
    ir.owned_inputs.assign(ir.input_descs.size(), cfg_.owned_inputs);

    // Outputs
    ir.output_descs = std::vector<core::TDesc>(cfg_.num_outputs);
    ir.owned_outputs.assign(ir.output_descs.size(), cfg_.owned_outputs);

    if (cfg_.inplace && !ir.input_descs.empty() && !ir.output_descs.empty()) {
      ir.in_place.push_back({0, 0});
    }
    return ir;
  }

  std::unique_ptr<core::IAsyncTask> create(std::span<const core::TDesc>,
                                           const nlohmann::json &,
                                           const core::AsyncCreateCtx &) const override {
    return std::make_unique<DummyAsyncTask>();
  }

private:
  DummyTaskConfig cfg_;
};

} // namespace holoflow::test
