#include "holoflow/model.hh"

#include <glog/logging.h>
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
    : meta(meta), host_data(std::move(h)), device_data(std::move(d)) {}

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

Model::Model(std::vector<std::unique_ptr<ModelNode>> nodes,
             std::unordered_map<int, TensorSlot> tensors,
             std::vector<std::reference_wrapper<ModelNode>> pes,
             std::vector<unique_cuda_stream> streams, ModelNode &root)
    : nodes_(std::move(nodes)), tensors_(std::move(tensors)),
      pes_(std::move(pes)), streams_(std::move(streams)), root_(root) {}

tl::expected<std::unique_ptr<Model>, Error>
Model::from_descriptor(const ModelDescriptor &descriptor) {

  ModelBuilder builder;
  return builder.build(descriptor);
}

} // namespace dh