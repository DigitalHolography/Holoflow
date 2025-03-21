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
                     const json &params, int itens_id, int otens_id,
                     cudaStream_t stream)
    : kind_(kind), name_(name), params_(params), itens_id_(itens_id),
      otens_id_(otens_id), stream_(stream) {}

std::string &ModelNode::name() { return name_; }

const std::string &ModelNode::name() const { return name_; }

std::string &ModelNode::kind() { return kind_; }

const std::string &ModelNode::kind() const { return kind_; }

json &ModelNode::params() { return params_; }

const json &ModelNode::params() const { return params_; }

int &ModelNode::itens_id() { return itens_id_; }

const int &ModelNode::itens_id() const { return itens_id_; }

int &ModelNode::otens_id() { return otens_id_; }

const int &ModelNode::otens_id() const { return otens_id_; }

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
    : ModelNode(kind, name, params, itens_id, otens_id, stream),
      task_(std::move(task)), task_meta_(task_meta) {}

void TaskNode::accept(ModelVisitor &visitor) { visitor.visit(*this); }

Task &TaskNode::task() { return *task_; }

const Task &TaskNode::task() const { return *task_; }

TaskMeta &TaskNode::task_meta() { return task_meta_; }

const TaskMeta &TaskNode::task_meta() const { return task_meta_; }

// ==========================================================================
//                     AccumulatorNode Implementation
// ==========================================================================

AccumulatorNode::AccumulatorNode(const std::string &kind,
                                 const std::string &name, const json &params,
                                 int itens_id, int otens_id,
                                 cudaStream_t stream,
                                 std::unique_ptr<Accumulator> accumulator,
                                 const AccumulatorMeta &accumulator_meta)
    : ModelNode(kind, name, params, itens_id, otens_id, stream),
      accumulator_(std::move(accumulator)),
      accumulator_meta_(accumulator_meta) {}

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

// ==========================================================================
//                     Model Implementation
// ==========================================================================

Model::Model(std::vector<std::unique_ptr<ModelNode>> nodes,
             std::unordered_map<int, TensorSlot> tensors,
             std::vector<std::reference_wrapper<ModelNode>> pes,
             ModelNode &root)
    : nodes_(std::move(nodes)), tensors_(std::move(tensors)),
      pes_(std::move(pes)), root_(root) {}

tl::expected<std::unique_ptr<Model>, Error>
Model::from_descriptor(const ModelDescriptor &descriptor,
                       const TensorMeta &imeta) {

  ModelBuilder builder;
  return builder.build(descriptor, imeta);
}

} // namespace dh