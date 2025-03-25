#include "holoflow/model.hh"

#include <glog/logging.h>
#include <nvtx3/nvToolsExt.h>
#include <optional>
#include <stack>

#include "curaii/curaii.hh"
#include "holoflow/model_builder.hh"

namespace dh {

// ==========================================================================
//                     TensorSlot Implementation
// ==========================================================================

Model::TensorSlot::TensorSlot(TensorMeta meta, unique_host_ptr<uint8_t> h,
                              unique_device_ptr<uint8_t> d)
    : meta(meta), host_data(std::move(h)), device_data(std::move(d)),
      data(nullptr) {
  switch (meta.memory_location()) {
  case MemoryLocation::HOST:
    data = host_data.get();
    break;
  case MemoryLocation::DEVICE:
    data = device_data.get();
    break;
  }
}

TensorView Model::TensorSlot::view() { return TensorView(data, meta); }

// ==========================================================================
//                     ModelNode Implementation
// ==========================================================================

ModelNode::ModelNode(const std::string &kind, const std::string &name,
                     const json &params, cudaStream_t stream)
    : kind_(kind), name_(name), params_(params), stream_(stream) {}

std::string &ModelNode::name() { return name_; }

const std::string &ModelNode::name() const { return name_; }

std::string &ModelNode::kind() { return kind_; }

const std::string &ModelNode::kind() const { return kind_; }

json &ModelNode::params() { return params_; }

const json &ModelNode::params() const { return params_; }

cudaStream_t ModelNode::stream() const { return stream_; }

void ModelNode::add_child(ModelNode &child) { children_.push_back(child); }

std::span<ModelNode::Child> ModelNode::children() { return children_; }

std::span<const ModelNode::Child> ModelNode::children() const {
  return children_;
}

// ==========================================================================
//                     TaskNode Implementation
// ==========================================================================

TaskNode::TaskNode(const std::string &kind, const std::string &name,
                   const json &params, int itens_id, int otens_id,
                   cudaStream_t stream, std::unique_ptr<Task> task,
                   const TaskMeta &task_meta)
    : ModelNode(kind, name, params, stream), task_(std::move(task)),
      task_meta_(task_meta), itens_id_(itens_id), otens_id_(otens_id) {}

void TaskNode::accept(ModelVisitor &visitor) { visitor.visit(*this); }

Task &TaskNode::task() { return *task_; }

const Task &TaskNode::task() const { return *task_; }

TaskMeta &TaskNode::task_meta() { return task_meta_; }

const TaskMeta &TaskNode::task_meta() const { return task_meta_; }

int &TaskNode::itens_id() { return itens_id_; }

const int &TaskNode::itens_id() const { return itens_id_; }

int &TaskNode::otens_id() { return otens_id_; }

const int &TaskNode::otens_id() const { return otens_id_; }

// ==========================================================================
//                     AccumulatorNode Implementation
// ==========================================================================

AccumulatorNode::AccumulatorNode(const std::string &kind,
                                 const std::string &name, const json &params,
                                 int itens_id, int otens_id,
                                 cudaStream_t stream,
                                 std::unique_ptr<Accumulator> accumulator,
                                 const AccumulatorMeta &accumulator_meta)
    : ModelNode(kind, name, params, stream),
      accumulator_(std::move(accumulator)), accumulator_meta_(accumulator_meta),
      itens_id_(itens_id), otens_id_(otens_id) {}

void AccumulatorNode::accept(ModelVisitor &visitor) { visitor.visit(*this); }

Accumulator &AccumulatorNode::accumulator() { return *accumulator_; }

const Accumulator &AccumulatorNode::accumulator() const {
  return *accumulator_;
}

AccumulatorMeta &AccumulatorNode::accumulator_meta() {
  return accumulator_meta_;
}

const AccumulatorMeta &AccumulatorNode::accumulator_meta() const {
  return accumulator_meta_;
}

int &AccumulatorNode::itens_id() { return itens_id_; }

const int &AccumulatorNode::itens_id() const { return itens_id_; }

int &AccumulatorNode::otens_id() { return otens_id_; }

const int &AccumulatorNode::otens_id() const { return otens_id_; }

// ==========================================================================
//                     SourceNode Implementation
// ==========================================================================

SourceNode::SourceNode(const std::string &kind, const std::string &name,
                       const json &params, int otens_id, cudaStream_t stream,
                       std::unique_ptr<Source> source,
                       const SourceMeta &source_meta)
    : ModelNode(kind, name, params, stream), source_(std::move(source)),
      source_meta_(source_meta), otens_id_(otens_id) {}

void SourceNode::accept(ModelVisitor &visitor) { visitor.visit(*this); }

Source &SourceNode::source() { return *source_; }

const Source &SourceNode::source() const { return *source_; }

SourceMeta &SourceNode::source_meta() { return source_meta_; }

const SourceMeta &SourceNode::source_meta() const { return source_meta_; }

int &SourceNode::otens_id() { return otens_id_; }

const int &SourceNode::otens_id() const { return otens_id_; }

// ==========================================================================
//                     SinkNode Implementation
// ==========================================================================

SinkNode::SinkNode(const std::string &kind, const std::string &name,
                   const json &params, int itens_id, cudaStream_t stream,
                   std::unique_ptr<Sink> sink, const SinkMeta &sink_meta)
    : ModelNode(kind, name, params, stream), sink_(std::move(sink)),
      sink_meta_(sink_meta), itens_id_(itens_id) {}

void SinkNode::accept(ModelVisitor &visitor) { visitor.visit(*this); }

Sink &SinkNode::sink() { return *sink_; }

const Sink &SinkNode::sink() const { return *sink_; }

SinkMeta &SinkNode::sink_meta() { return sink_meta_; }

const SinkMeta &SinkNode::sink_meta() const { return sink_meta_; }

int &SinkNode::itens_id() { return itens_id_; }

const int &SinkNode::itens_id() const { return itens_id_; }

// ==========================================================================
//                     Model Implementation
// ==========================================================================

namespace {

class AllocatePES : public ModelVisitor {
public:
  AllocatePES(std::unordered_map<int, Model::TensorSlot> &tensors,
              ModelNode &root, std::atomic<bool> &stop)
      : tensors_(tensors), root_(root), stop_(stop) {}

  void visit(TaskNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }

    auto &itens = tensors_.at(node.itens_id());
    CHECK_NOTNULL(itens.data);

    auto &otens = tensors_.at(node.itens_id());
    CHECK_NOTNULL(otens.data);
  }

  void visit(AccumulatorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (&node == &root_) {
      auto &otens = tensors_.at(node.otens_id());

      auto result = node.accumulator().read_tensor();
      CHECK(result);
      auto view = result.value();

      while (!view && !stop_) {
        result = node.accumulator().read_tensor();
        CHECK(result);
        view = result.value();
      }

      if (stop_) {
        LOG(INFO) << "Stop signal received.";
        return;
      }

      otens.data = (uint8_t *)view.value().data();
    }

    else {
      auto &itens = tensors_.at(node.itens_id());

      auto result = node.accumulator().write_tensor();
      CHECK(result);
      auto view = result.value();

      while (!view && !stop_) {
        result = node.accumulator().write_tensor();
        CHECK(result);
        view = result.value();
      }

      if (stop_) {
        LOG(INFO) << "Stop signal received.";
        return;
      }

      itens.data = (uint8_t *)view.value().data();
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }

    auto &otens = tensors_.at(node.otens_id());
    CHECK_NOTNULL(otens.data);
  }

  void visit(SinkNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    CHECK(node.children().empty());

    auto &itens = tensors_.at(node.itens_id());
    CHECK_NOTNULL(itens.data);
  }

private:
  std::unordered_map<int, Model::TensorSlot> &tensors_;
  ModelNode &root_;
  std::atomic<bool> &stop_;
};

class FreePES : public ModelVisitor {
public:
  FreePES(std::unordered_map<int, Model::TensorSlot> &tensors, ModelNode &root,
          std::atomic<bool> &stop)
      : tensors_(tensors), root_(root), stop_(stop) {}

  void visit(TaskNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(AccumulatorNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (&node == &root_) {
      auto result = node.accumulator().commit_read();
      CHECK(result);
    } else {
      auto result = node.accumulator().commit_write();
      CHECK(result);
    }

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SourceNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    for (auto child : node.children()) {
      child.get().accept(*this);
    }
  }

  void visit(SinkNode &node) override {
    VLOG(2) << "visiting: " << node.name();
  }

private:
  std::unordered_map<int, Model::TensorSlot> &tensors_;
  ModelNode &root_;
  std::atomic<bool> &stop_;
};

class ExecPES : public ModelVisitor {
public:
  ExecPES(std::unordered_map<int, Model::TensorSlot> &tensors, ModelNode &root,
          std::atomic<bool> &stop)
      : tensors_(tensors), root_(root), stop_(stop) {}

  void visit(TaskNode &node) override {
    if (stop_) {
      LOG(INFO) << "Stop signal received.";
      return;
    }

    auto itens = tensors_.at(node.itens_id()).view();
    auto otens = tensors_.at(node.otens_id()).view();
    CHECK_NOTNULL(itens.data());
    CHECK_NOTNULL(otens.data());
    VLOG(3) << "Running " << node.name();
    CHECK(node.task().run(itens, otens));

    // Do non-inlined children.
    for (auto child : node.children()) {
      auto *task = dynamic_cast<TaskNode *>(&child.get());
      if (!task || !task->task_meta().inlined()) {
        child.get().accept(*this);
      }
    }

    // Do inlined child next.
    for (auto &child : node.children()) {
      auto *task = dynamic_cast<TaskNode *>(&child.get());
      if (task && task->task_meta().inlined()) {
        child.get().accept(*this);
      }
    }
  }

  void visit(AccumulatorNode &node) override {
    if (stop_) {
      LOG(INFO) << "Stop signal received.";
      return;
    }

    if (&node == &root_) {
      nvtxRangePushA("PES");
      // Do non-inlined children first.
      for (auto child : node.children()) {
        auto *task = dynamic_cast<TaskNode *>(&child.get());
        if (!task || !task->task_meta().inlined()) {
          child.get().accept(*this);
        }
      }

      // Do inlined child next.
      for (auto &child : node.children()) {
        auto *task = dynamic_cast<TaskNode *>(&child.get());
        if (task && task->task_meta().inlined()) {
          child.get().accept(*this);
        }
      }

      CHECK(cudaStreamSynchronize(node.stream()) == cudaSuccess);
      nvtxRangePop();
    }
  }

  void visit(SourceNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (stop_) {
      LOG(INFO) << "Stop signal received.";
      return;
    }

    auto otens = tensors_.at(node.otens_id()).view();
    CHECK_NOTNULL(otens.data());
    VLOG(3) << "Running " << node.name();
    CHECK(node.source().run(otens));

    // Do non-inlined children.
    for (auto child : node.children()) {
      auto *task = dynamic_cast<TaskNode *>(&child.get());
      if (!task || !task->task_meta().inlined()) {
        child.get().accept(*this);
      }
    }

    // Do inlined child next.
    for (auto &child : node.children()) {
      auto *task = dynamic_cast<TaskNode *>(&child.get());
      if (task && task->task_meta().inlined()) {
        child.get().accept(*this);
      }
    }
  }

  void visit(SinkNode &node) override {
    VLOG(2) << "visiting: " << node.name();

    if (stop_) {
      LOG(INFO) << "Stop signal received.";
      return;
    }

    auto itens = tensors_.at(node.itens_id()).view();
    CHECK_NOTNULL(itens.data());
    VLOG(3) << "Running " << node.name();
    CHECK(node.sink().run(itens));
  }

private:
  std::unordered_map<int, Model::TensorSlot> &tensors_;
  ModelNode &root_;
  std::atomic<bool> &stop_;
};

} // namespace

Model::Model(std::vector<std::unique_ptr<ModelNode>> nodes,
             std::unordered_map<int, TensorSlot> tensors,
             std::vector<std::reference_wrapper<ModelNode>> pes,
             std::vector<unique_cuda_stream> streams, ModelNode &root)
    : nodes_(std::move(nodes)), tensors_(std::move(tensors)),
      pes_(std::move(pes)), streams_(std::move(streams)), root_(root),
      state_(State::STOPPED), stop_flag_(false) {}

tl::expected<std::unique_ptr<Model>, Error>
Model::from_descriptor(const ModelDescriptor &descriptor) {
  ModelBuilder builder;
  return builder.build(descriptor);
}

void Model::run() {
  std::vector<std::thread> threads;

  for (auto &pes : pes_) {
    threads.emplace_back([this, pes]() {
      AllocatePES allocate_pes(tensors_, pes, stop_flag_);
      ExecPES exec_pes(tensors_, pes, stop_flag_);
      FreePES free_pes(tensors_, pes, stop_flag_);

      while (!stop_flag_) {
        pes.get().accept(allocate_pes);
        pes.get().accept(exec_pes);
        pes.get().accept(free_pes);
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void Model::start() {
  CHECK(state_ == State::STOPPED);
  state_ = State::RUNNING;
  stop_flag_ = false;

  model_thread_ = std::thread([this]() { run(); });
}

void Model::stop() {
  CHECK(state_ == State::STOPPED);
  stop_flag_ = true;

  model_thread_.join();
  state_ = State::STOPPED;
}

} // namespace dh