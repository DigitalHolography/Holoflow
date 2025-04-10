#include "holoflow/v2/model_transaction.hh"

#include <memory>
#include <tl/expected.hpp>

#include "bug_buster/bug_buster.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v2/error.hh"

namespace dh::v2 {

// ==========================================================================
//                     AssignCudaStreamVisitor Implementation
// ==========================================================================

ModelTransaction::AssignCudaStreamVisitor::AssignCudaStreamVisitor(
    std::vector<CudaStream> &streams, std::vector<Error> &errors)
    : streams_(streams), errors_(errors) {}

void ModelTransaction::AssignCudaStreamVisitor::visit(Model::TaskNode &node) {
  holoflow_logger()->trace("[AssignCudaStreamVisitor::visit] Visiting task "
                           "node: {}",
                           *node.name_);

  DH_CHECK(!stream_stack_.empty());
  node.stream_ = stream_stack_.top();

  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
}

void ModelTransaction::AssignCudaStreamVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace("[AssignCudaStreamVisitor::visit] Visiting "
                           "accumulator node: {}",
                           *node.name_);

  if (!node.stream_) {
    auto result = CudaStream::try_create();
    DH_CHECK(result);
    auto stream = std::move(result.value());
    node.stream_ = stream.ref();
    streams_.push_back(std::move(stream));
  }

  stream_stack_.push(*node.stream_);
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  stream_stack_.pop();
}

void ModelTransaction::AssignCudaStreamVisitor::visit(Model::SourceNode &node) {
  holoflow_logger()->trace("[AssignCudaStreamVisitor::visit] Visiting source "
                           "node: {}",
                           *node.name_);

  if (!node.stream_) {
    auto result = CudaStream::try_create();
    DH_CHECK(result);
    auto stream = std::move(result.value());
    node.stream_ = stream.ref();
    streams_.push_back(std::move(stream));
  }

  stream_stack_.push(*node.stream_);
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  stream_stack_.pop();
}

void ModelTransaction::AssignCudaStreamVisitor::visit(Model::SinkNode &node) {
  holoflow_logger()->trace("[AssignCudaStreamVisitor::visit] Visiting sink "
                           "node: {}",
                           *node.name_);

  DH_CHECK(!stream_stack_.empty());
  node.stream_ = stream_stack_.top();
}

// ==========================================================================
//                     TypeCheckVisitor Implementation
// ==========================================================================

ModelTransaction::TypeCheckVisitor::TypeCheckVisitor(
    std::vector<Error> &errors, const Model::TaskFactoryMap &task_factories_map,
    const Model::AccumulatorFactoryMap &accumulator_factories_map,
    const Model::SourceFactoryMap &source_factories_map,
    const Model::SinkFactoryMap &sink_factories_map)
    : errors_(errors), task_factories_map_(task_factories_map),
      accumulator_factories_map_(accumulator_factories_map),
      source_factories_map_(source_factories_map),
      sink_factories_map_(sink_factories_map) {}

void ModelTransaction::TypeCheckVisitor::visit(Model::TaskNode &node) {
  holoflow_logger()->trace("[TypeCheckVisitor::visit] Visiting task node: {}",
                           *node.name_);

  DH_CHECK(!imetas_stack_.empty());
  DH_CHECK(task_factories_map_.contains(*node.kind_));

  auto &factory = task_factories_map_.at(*node.kind_);
  auto result = factory.get().type_check(imetas_stack_.top(), *node.params_);
  if (!result) {
    holoflow_logger()->warn("[TypeCheckVisitor::visit] Type checking failed at "
                            "node: {}",
                            *node.name_);

    // TODO: Use error from result
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Type checking failed at node: {}",
                                  *node.name_));
    return;
  }

  node.task_meta_ = result.value();
  imetas_stack_.push(result.value().ometa());
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  imetas_stack_.pop();
}

void ModelTransaction::TypeCheckVisitor::visit(Model::AccumulatorNode &node) {
  holoflow_logger()->trace("[TypeCheckVisitor::visit] Visiting accumulator "
                           "node: {}",
                           *node.name_);

  DH_CHECK(!imetas_stack_.empty());
  DH_CHECK(accumulator_factories_map_.contains(*node.kind_));

  auto &factory = accumulator_factories_map_.at(*node.kind_);
  auto result = factory.get().type_check(imetas_stack_.top(), *node.params_);
  if (!result) {
    holoflow_logger()->warn("[TypeCheckVisitor::visit] Type checking failed at "
                            "node: {}",
                            *node.name_);

    // TODO: Use error from result
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Type checking failed at node: {}",
                                  *node.name_));
    return;
  }

  node.accumulator_meta_ = result.value();
  imetas_stack_.push(result.value().ometa());
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  imetas_stack_.pop();
}

void ModelTransaction::TypeCheckVisitor::visit(Model::SourceNode &node) {
  holoflow_logger()->trace("[TypeCheckVisitor::visit] Visiting source node: {}",
                           *node.name_);

  DH_CHECK(imetas_stack_.empty());
  DH_CHECK(source_factories_map_.contains(*node.kind_));

  auto &factory = source_factories_map_.at(*node.kind_);
  auto result = factory.get().type_check(*node.params_);
  if (!result) {
    holoflow_logger()->warn("[TypeCheckVisitor::visit] Type checking failed at "
                            "node: {}",
                            *node.name_);

    // TODO: Use error from result
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Type checking failed at node: {}",
                                  *node.name_));
  }

  node.source_meta_ = result.value();
  imetas_stack_.push(result.value().ometa());
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  imetas_stack_.pop();
}

void ModelTransaction::TypeCheckVisitor::visit(Model::SinkNode &node) {
  holoflow_logger()->trace("[TypeCheckVisitor::visit] Visiting sink node: {}",
                           *node.name_);

  DH_CHECK(!imetas_stack_.empty());
  DH_CHECK(sink_factories_map_.contains(*node.kind_));

  auto &factory = sink_factories_map_.at(*node.kind_);
  auto result = factory.get().type_check(imetas_stack_.top(), *node.params_);
  if (!result) {
    holoflow_logger()->warn("[TypeCheckVisitor::visit] Type checking failed "
                            "at node: {}",
                            *node.name_);

    // TODO: Use error from result
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Type checking failed at node: {}",
                                  *node.name_));
    return;
  }

  node.sink_meta_ = result.value();
}

// ==========================================================================
//                     NonInlinedChildCheckVisitor Implementation
// ==========================================================================

ModelTransaction::NonInlinedChildCheckVisitor::NonInlinedChildCheckVisitor(
    std::vector<Error> &errors)
    : errors_(errors) {}

void ModelTransaction::NonInlinedChildCheckVisitor::visit(
    Model::TaskNode &node) {
  holoflow_logger()->trace("[NonInlinedChildCheckVisitor::visit] Visiting task "
                           "node: {}",
                           *node.name_);

  DH_CHECK(!seen_non_inlined_.empty());

  if (!node.is_inlined()) {
    seen_non_inlined_.push(true);
  } else {
    seen_non_inlined_.push(seen_non_inlined_.top());
  }

  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  seen_non_inlined_.pop();
}

void ModelTransaction::NonInlinedChildCheckVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace("[NonInlinedChildCheckVisitor::visit] Visiting "
                           "accumulator node: {}",
                           *node.name_);

  DH_CHECK(!seen_non_inlined_.empty());

  if (!seen_non_inlined_.top()) {
    holoflow_logger()->warn("[NonInlinedChildCheckVisitor::visit] Non inlined "
                            "task check failed at node: {}",
                            *node.name_);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Non inlined task check failed at node: {}",
                                  *node.name_));
    return;
  }

  seen_non_inlined_.push(false);
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  seen_non_inlined_.pop();
}

void ModelTransaction::NonInlinedChildCheckVisitor::visit(
    Model::SourceNode &node) {
  holoflow_logger()->trace("[NonInlinedChildCheckVisitor::visit] Visiting "
                           "source node: {}",
                           *node.name_);

  DH_CHECK(seen_non_inlined_.empty());

  seen_non_inlined_.push(true);
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  seen_non_inlined_.pop();
}

void ModelTransaction::NonInlinedChildCheckVisitor::visit(
    Model::SinkNode &node) {
  holoflow_logger()->trace("[NonInlinedChildCheckVisitor::visit] Visiting "
                           "sink node: {}",
                           *node.name_);
}

// ==========================================================================
//                     AssignAccumulatorTensorsVisitor Implementation
// ==========================================================================

ModelTransaction::AssignAccumulatorTensorsVisitor::
    AssignAccumulatorTensorsVisitor(int &next_id)
    : next_id_(next_id) {}

void ModelTransaction::AssignAccumulatorTensorsVisitor::visit(
    Model::TaskNode &node) {
  holoflow_logger()->trace("[AssignAccumulatorTensorsVisitor::visit] Visiting "
                           "task node: {}",
                           *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignAccumulatorTensorsVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace(
      "[AssignAccumulatorTensorsVisitor::visit] Visiting accumulator node: {}",
      *node.name_);

  // Set current
  if (!node.itens_id_) {
    node.itens_id_ = next_id_++;
    holoflow_logger()->trace(
        "[AssignAccumulatorTensorsVisitor::visit] {} <= itens_id: {}",
        *node.name_, *node.itens_id_);
  }
  if (!node.otens_id_) {
    node.otens_id_ = next_id_++;
    holoflow_logger()->trace(
        "[AssignAccumulatorTensorsVisitor::visit] {} <= otens_id: {}",
        *node.name_, *node.otens_id_);
  }

  // Set parent
  DH_CHECK(!parents_stack_.empty());
  auto parent = parents_stack_.top();
  if (auto *task = dynamic_cast<Model::TaskNode *>(&parent.get())) {
    task->otens_id_ = node.itens_id_;
    holoflow_logger()->trace(
        "[AssignAccumulatorTensorsVisitor::visit] {} <= otens_id: {}",
        *task->name_, *task->otens_id_);
  } else if (auto *source = dynamic_cast<Model::SourceNode *>(&parent.get())) {
    source->otens_id_ = node.itens_id_;
    holoflow_logger()->trace(
        "[AssignAccumulatorTensorsVisitor::visit] {} <= otens_id: {}",
        *source->name_, *source->otens_id_);
  } else {
    DH_BUG("Expected parent to be TaskNode or SourceNode, but got {}",
           typeid(parent.get()).name());
  }

  // Set children
  for (auto &child : node.children_) {
    if (auto *task = dynamic_cast<Model::TaskNode *>(&child.get())) {
      task->itens_id_ = node.otens_id_;
      holoflow_logger()->trace(
          "[AssignAccumulatorTensorsVisitor::visit] {} <= itens_id: {}",
          *task->name_, *task->itens_id_);
    } else if (auto *sink = dynamic_cast<Model::SinkNode *>(&child.get())) {
      sink->itens_id_ = node.otens_id_;
      holoflow_logger()->trace(
          "[AssignAccumulatorTensorsVisitor::visit] {} <= itens_id: {}",
          *sink->name_, *sink->itens_id_);
    } else {
      DH_BUG("Expected child to be TaskNode or SinkNode, but got {}",
             typeid(child.get()).name());
    }
  }

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignAccumulatorTensorsVisitor::visit(
    Model::SourceNode &node) {
  holoflow_logger()->trace("[AssignAccumulatorTensorsVisitor::visit] Visiting "
                           "source node: {}",
                           *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignAccumulatorTensorsVisitor::visit(
    Model::SinkNode &node) {
  holoflow_logger()->trace("[AssignAccumulatorTensorsVisitor::visit] Visiting "
                           "sink node: {}",
                           *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

// ==========================================================================
//                     AssignInlinedTaskTensorsVisitor Implementation
// ==========================================================================

ModelTransaction::AssignInlinedTaskTensorsVisitor::
    AssignInlinedTaskTensorsVisitor(int &next_id)
    : next_id_(next_id) {}

void ModelTransaction::AssignInlinedTaskTensorsVisitor::visit(
    Model::TaskNode &node) {
  holoflow_logger()->trace(
      "[AssignInlinedTaskTensorsVisitor::visit] Visiting task node: {}",
      *node.name_);

  // Forward propagation (otens_id <= itens_id)
  if (node.is_inlined() && node.itens_id_ && node.otens_id_ != node.itens_id_) {
    // Set current
    DH_CHECK(!node.otens_id_);
    node.otens_id_ = node.itens_id_;
    holoflow_logger()->trace(
        "[AssignInlinedTaskTensorsVisitor::visit] {} <= otens_id: {}",
        *node.name_, *node.otens_id_);

    // Set children
    for (auto &child : node.children_) {
      if (auto *task = dynamic_cast<Model::TaskNode *>(&child.get())) {
        task->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *task->name_, *task->itens_id_);
      } else if (auto *sink = dynamic_cast<Model::SinkNode *>(&child.get())) {
        sink->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *sink->name_, *sink->itens_id_);
      } else {
        DH_BUG("Expected child to be TaskNode or SinkNode, but got {}",
               typeid(child.get()).name());
      }
    }
  }

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();

  // Backward propagation (itens_id <= otens_id)
  if (node.is_inlined() && node.otens_id_ && node.itens_id_ != node.otens_id_) {
    // Set current
    DH_CHECK(!node.itens_id_);
    node.itens_id_ = node.otens_id_;
    holoflow_logger()->trace(
        "[AssignInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
        *node.name_, *node.itens_id_);

    // Set parent
    DH_CHECK(!parents_stack_.empty());
    auto parent = parents_stack_.top();
    if (auto *task = dynamic_cast<Model::TaskNode *>(&parent.get())) {
      task->otens_id_ = node.itens_id_;
      holoflow_logger()->trace(
          "[AssignInlinedTaskTensorsVisitor::visit] {} <= otens_id: {}",
          *task->name_, *task->otens_id_);
    } else if (auto *source =
                   dynamic_cast<Model::SourceNode *>(&parent.get())) {
      source->otens_id_ = node.itens_id_;
      holoflow_logger()->trace(
          "[AssignInlinedTaskTensorsVisitor::visit] {} <= otens_id: {}",
          *source->name_, *source->otens_id_);
    } else {
      DH_BUG("Expected parent to be TaskNode or SourceNode, but got {}",
             typeid(parent.get()).name());
    }
  }
}

void ModelTransaction::AssignInlinedTaskTensorsVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace(
      "[AssignInlinedTaskTensorsVisitor::visit] Visiting accumulator node: {}",
      *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignInlinedTaskTensorsVisitor::visit(
    Model::SourceNode &node) {
  holoflow_logger()->trace(
      "[AssignInlinedTaskTensorsVisitor::visit] Visiting source node: {}",
      *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignInlinedTaskTensorsVisitor::visit(
    Model::SinkNode &node) {
  holoflow_logger()->trace(
      "[AssignInlinedTaskTensorsVisitor::visit] Visiting sink node: {}",
      *node.name_);
}

// ==========================================================================
//                     AssignNonInlinedTaskTensorsVisitor Implementation
// ==========================================================================

ModelTransaction::AssignNonInlinedTaskTensorsVisitor::
    AssignNonInlinedTaskTensorsVisitor(int &next_id)
    : next_id_(next_id) {}

void ModelTransaction::AssignNonInlinedTaskTensorsVisitor::visit(
    Model::TaskNode &node) {
  holoflow_logger()->trace(
      "[AssignNonInlinedTaskTensorsVisitor::visit] Visiting task node: {}",
      *node.name_);

  if (!node.is_inlined() && !node.otens_id_) {
    // Set current
    node.otens_id_ = next_id_++;
    holoflow_logger()->trace(
        "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= otens_id: {}",
        *node.name_, *node.otens_id_);

    // Set children
    for (auto &child : node.children_) {
      if (auto *task = dynamic_cast<Model::TaskNode *>(&child.get())) {
        task->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *task->name_, *task->itens_id_);
      } else if (auto *sink = dynamic_cast<Model::SinkNode *>(&child.get())) {
        sink->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *sink->name_, *sink->itens_id_);
      } else {
        DH_BUG("Expected child to be TaskNode or SinkNode, but got {}",
               typeid(child.get()).name());
      }
    }
  }

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignNonInlinedTaskTensorsVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace(
      "[AssignNonInlinedTaskTensorsVisitor::visit] Visiting accumulator node: "
      "{}",
      *node.name_);

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignNonInlinedTaskTensorsVisitor::visit(
    Model::SourceNode &node) {
  holoflow_logger()->trace(
      "[AssignNonInlinedTaskTensorsVisitor::visit] Visiting source node: {}",
      *node.name_);

  if (!node.otens_id_) {
    // Set current
    node.otens_id_ = next_id_++;
    holoflow_logger()->trace(
        "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= otens_id: {}",
        *node.name_, *node.otens_id_);

    // Set children
    for (auto &child : node.children_) {
      if (auto *task = dynamic_cast<Model::TaskNode *>(&child.get())) {
        task->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *task->name_, *task->itens_id_);
      } else if (auto *sink = dynamic_cast<Model::SinkNode *>(&child.get())) {
        sink->itens_id_ = node.otens_id_;
        holoflow_logger()->trace(
            "[AssignNonInlinedTaskTensorsVisitor::visit] {} <= itens_id: {}",
            *sink->name_, *sink->itens_id_);
      } else {
        DH_BUG("Expected child to be TaskNode or SinkNode, but got {}",
               typeid(child.get()).name());
      }
    }
  }

  parents_stack_.push(std::ref(node));
  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
  parents_stack_.pop();
}

void ModelTransaction::AssignNonInlinedTaskTensorsVisitor::visit(
    Model::SinkNode &node) {
  holoflow_logger()->trace(
      "[AssignNonInlinedTaskTensorsVisitor::visit] Visiting sink node: {}",
      *node.name_);
}

// ==========================================================================
//                     CallFactoriesVisitor Implementation
// ==========================================================================

ModelTransaction::CallFactoriesVisitor::CallFactoriesVisitor(
    const Model::TaskFactoryMap &task_factories_map,
    const Model::AccumulatorFactoryMap &accumulator_factories_map,
    const Model::SourceFactoryMap &source_factories_map,
    const Model::SinkFactoryMap &sink_factories_map, std::vector<Error> &errors)
    : task_factories_map_(task_factories_map),
      accumulator_factories_map_(accumulator_factories_map),
      source_factories_map_(source_factories_map),
      sink_factories_map_(sink_factories_map), errors_(errors) {}

void ModelTransaction::CallFactoriesVisitor::visit(Model::TaskNode &node) {
  holoflow_logger()->trace(
      "[CallFactoriesVisitor::visit] Visiting task node: {}", *node.name_);

  DH_CHECK(task_factories_map_.contains(*node.kind_));
  auto &factory = task_factories_map_.at(*node.kind_);
  auto result = factory.get().create((*node.task_meta_).imeta(), *node.params_,
                                     *node.stream_);
  if (!result) {
    holoflow_logger()->warn(
        "[CallFactoriesVisitor::visit] Factory call failed at node: {}",
        *node.name_);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory call failed at node: {}",
                                  *node.name_));
    return;
  }

  node.task_ = std::move(result.value());

  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
}

void ModelTransaction::CallFactoriesVisitor::visit(
    Model::AccumulatorNode &node) {
  holoflow_logger()->trace(
      "[CallFactoriesVisitor::visit] Visiting accumulator node: {}",
      *node.name_);

  DH_CHECK(accumulator_factories_map_.contains(*node.kind_));
  auto &factory = accumulator_factories_map_.at(*node.kind_);
  auto result = factory.get().create((*node.accumulator_meta_).imeta(),
                                     *node.params_, *node.stream_);
  if (!result) {
    holoflow_logger()->warn(
        "[CallFactoriesVisitor::visit] Factory call failed at node: {}",
        *node.name_);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory call failed at node: {}",
                                  *node.name_));
    return;
  }

  node.accumulator_ = std::move(result.value());

  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
}

void ModelTransaction::CallFactoriesVisitor::visit(Model::SourceNode &node) {
  holoflow_logger()->trace(
      "[CallFactoriesVisitor::visit] Visiting source node: {}", *node.name_);

  DH_CHECK(source_factories_map_.contains(*node.kind_));
  auto &factory = source_factories_map_.at(*node.kind_);
  auto result = factory.get().create(*node.params_, *node.stream_);
  if (!result) {
    holoflow_logger()->warn(
        "[CallFactoriesVisitor::visit] Factory call failed at node: {}",
        *node.name_);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory call failed at node: {}",
                                  *node.name_));
    return;
  }

  node.source_ = std::move(result.value());

  for (auto &child : node.children_) {
    child.get().accept(*this);
  }
}

void ModelTransaction::CallFactoriesVisitor::visit(Model::SinkNode &node) {
  holoflow_logger()->trace(
      "[CallFactoriesVisitor::visit] Visiting sink node: {}", *node.name_);

  DH_CHECK(sink_factories_map_.contains(*node.kind_));
  auto &factory = sink_factories_map_.at(*node.kind_);
  auto result = factory.get().create((*node.sink_meta_).imeta(), *node.params_,
                                     *node.stream_);
  if (!result) {
    holoflow_logger()->warn(
        "[CallFactoriesVisitor::visit] Factory call failed at node: {}",
        *node.name_);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory call failed at node: {}",
                                  *node.name_));
    return;
  }

  node.sink_ = std::move(result.value());
}

// ==========================================================================
//                     Model Implementation
// ==========================================================================

ModelTransaction::ModelTransaction(Model &model) : model_(model) {}

ModelTransaction &ModelTransaction::add_source(const std::string &name,
                                               const std::string &kind,
                                               const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_source] Adding source: {}",
                           name);

  if (!model_.has_source_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_source] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_source] Source {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Source {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<Model::SourceNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  model_.nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_sink(const std::string &name,
                                             const std::string &kind,
                                             const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_sink] Adding sink: {}",
                           name);

  if (!model_.has_sink_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_sink] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_sink] Sink {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Sink {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<Model::SinkNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  model_.nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_task(const std::string &name,
                                             const std::string &kind,
                                             const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_task] Adding task: {}",
                           name);

  if (!model_.has_task_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_task] Factory {} is "
                            "not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_task] Task {} is "
                            "already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Task {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<Model::TaskNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  model_.nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::add_accumulator(const std::string &name,
                                                    const std::string &kind,
                                                    const json &params) {
  holoflow_logger()->trace("[ModelTransaction::add_accumulator] Adding "
                           "accumulator: {}",
                           name);

  if (!model_.has_accumulator_factory(kind)) {
    holoflow_logger()->warn("[ModelTransaction::add_accumulator] Factory {} "
                            "is not registered",
                            kind);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Factory {} is not registered", kind));
    return *this;
  }

  if (has_node(name)) {
    holoflow_logger()->warn("[ModelTransaction::add_accumulator] "
                            "Accumulator {} is already declared",
                            name);
    errors_.push_back(Error::make(ErrorType::InternalError,
                                  "Accumulator {} is already declared", name));
    return *this;
  }

  auto node = std::make_unique<Model::AccumulatorNode>();
  node->name_ = name;
  node->kind_ = kind;
  node->params_ = params;

  model_.nodes_.push_back(std::move(node));
  return *this;
}

ModelTransaction &ModelTransaction::update_node_params(const std::string &name,
                                                       const json &params) {
  holoflow_logger()->trace("[ModelTransaction::update_node_params] Updating "
                           "node params: {}",
                           name);

  for (auto &node : model_.nodes_) {
    if (node->name_ == name) {
      node->params_ = params;
      return *this;
    }
  }

  holoflow_logger()->warn("[ModelTransaction::update_node_params] Node {} "
                          "not found",
                          name);
  errors_.push_back(
      Error::make(ErrorType::NotFound, "Node {} not found", name));
  return *this;
}

ModelTransaction &ModelTransaction::remove_node(const std::string &name,
                                                RemoveNodeBehavior behavior) {
  holoflow_logger()->trace("[ModelTransaction::remove_node] Removing node: {}",
                           name);

  for (auto it = model_.nodes_.begin(); it != model_.nodes_.end(); ++it) {
    if ((*it)->name_ == name) {
      if (behavior == RemoveNodeBehavior::RemoveSubtree) {
        // Remove children
        for (auto &child : (*it)->children_) {
          remove_node(*child.get().name_, behavior);
        }

        // Unlink the node from its parent
        for (auto &parent : model_.nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        model_.nodes_.erase(it);
        return *this;
      }

      if (behavior == RemoveNodeBehavior::OrphanChildren) {
        // Unlink the node from its parent
        for (auto &parent : model_.nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        model_.nodes_.erase(it);
        return *this;
      }

      if (behavior == RemoveNodeBehavior::ReparentChildren) {
        // Add children to the parent
        for (auto &parent : model_.nodes_) {
          if (std::any_of(parent->children_.begin(), parent->children_.end(),
                          [&name](const auto &child) {
                            return child.get().name_ == name;
                          })) {
            for (auto &child : (*it)->children_) {
              parent->children_.push_back(child);
            }
          }
        }

        // Unlink the node from its parent
        for (auto &parent : model_.nodes_) {
          parent->children_.erase(
              std::remove_if(parent->children_.begin(), parent->children_.end(),
                             [&name](const auto &child) {
                               return child.get().name_ == name;
                             }),
              parent->children_.end());
        }

        // Remove the node itself
        model_.nodes_.erase(it);
        return *this;
      }
    }
  }

  holoflow_logger()->warn("[ModelTransaction::remove_node] Node {} not found",
                          name);
  errors_.push_back(
      Error::make(ErrorType::NotFound, "Node {} not found", name));
  return *this;
}

ModelTransaction &ModelTransaction::connect(const std::string &parent_name,
                                            const std::string &child_name) {
  holoflow_logger()->trace("[ModelTransaction::connect] Connecting {} to {}",
                           child_name, parent_name);

  auto parent_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(),
      [&parent_name](const auto &node) { return node->name_ == parent_name; });

  auto child_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(),
      [&child_name](const auto &node) { return node->name_ == child_name; });

  if (parent_it != model_.nodes_.end() && child_it != model_.nodes_.end()) {
    (*parent_it)->children_.push_back(**child_it);
  } else {
    holoflow_logger()->warn("[ModelTransaction::connect] Node {} or {} not "
                            "found",
                            parent_name, child_name);
    errors_.push_back(Error::make(ErrorType::NotFound,
                                  "Node {} or {} not found", parent_name,
                                  child_name));
  }

  return *this;
}

ModelTransaction &
ModelTransaction::disconnect(const std::string &parent_name,
                             const std::string &child_name,
                             DisconnectNodeBehavior behavior) {
  holoflow_logger()->trace("[ModelTransaction::disconnect] Disconnecting {} "
                           "from {}",
                           child_name, parent_name);

  auto parent_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(),
      [&parent_name](const auto &node) { return node->name_ == parent_name; });

  auto child_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(),
      [&child_name](const auto &node) { return node->name_ == child_name; });

  if (parent_it != model_.nodes_.end() && child_it != model_.nodes_.end()) {
    if (behavior == DisconnectNodeBehavior::OrphanChild) {
      // Remove the child from the parent
      (*parent_it)
          ->children_.erase(std::remove_if((*parent_it)->children_.begin(),
                                           (*parent_it)->children_.end(),
                                           [&child_name](const auto &child) {
                                             return child.get().name_ ==
                                                    child_name;
                                           }),
                            (*parent_it)->children_.end());
    } else if (behavior == DisconnectNodeBehavior::ReparentChildren) {
      // Reparent children to the parent
      for (auto &child : (*child_it)->children_) {
        (*parent_it)->children_.push_back(child);
      }

      // Remove the child from the parent
      (*parent_it)
          ->children_.erase(std::remove_if((*parent_it)->children_.begin(),
                                           (*parent_it)->children_.end(),
                                           [&child_name](const auto &child) {
                                             return child.get().name_ ==
                                                    child_name;
                                           }),
                            (*parent_it)->children_.end());
    }
  } else {
    holoflow_logger()->warn("[ModelTransaction::disconnect] Node {} or {} not "
                            "found",
                            parent_name, child_name);
    errors_.push_back(Error::make(ErrorType::NotFound,
                                  "Node {} or {} not found", parent_name,
                                  child_name));
  }
  return *this;
}

bool ModelTransaction::has_source(const std::string &name) const {
  for (const auto &node : model_.nodes_) {
    if (node->name_ == name && model_.has_source_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_sink(const std::string &name) const {
  for (const auto &node : model_.nodes_) {
    if (node->name_ == name && model_.has_sink_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_task(const std::string &name) const {
  for (const auto &node : model_.nodes_) {
    if (node->name_ == name && model_.has_task_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_accumulator(const std::string &name) const {
  for (const auto &node : model_.nodes_) {
    if (node->name_ == name && model_.has_accumulator_factory(*node->kind_)) {
      return true;
    }
  }

  return false;
}

bool ModelTransaction::has_node(const std::string &name) const {
  for (const auto &node : model_.nodes_) {
    if (node->name_ == name) {
      return true;
    }
  }

  return false;
}

Model::Node *ModelTransaction::get_root() {
  // Find the source node in the model.
  auto source_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(), [this](const auto &node) {
        return model_.has_source_factory(*node->kind_);
      });

  if (source_it == model_.nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::get_root] No source node "
                            "found");
    errors_.push_back(Error::make(ErrorType::NotFound, "No source node found"));
    return nullptr;
  }

  // Return the source node found.
  return source_it->get();
}

void ModelTransaction::validate_orphan_nodes() {
  holoflow_logger()->debug(
      "[ModelTransaction::validateOrphanNodes] Validating orphan nodes");

  for (const auto &node : model_.nodes_) {
    // Check if there is any parent node that lists this node as a child.
    auto parent_it = std::find_if(
        model_.nodes_.begin(), model_.nodes_.end(),
        [&node](const auto &parent) {
          return std::any_of(parent->children_.begin(), parent->children_.end(),
                             [&node](const auto &child) {
                               return child.get().name_ == node->name_;
                             });
        });
    // If no parent is found and the node is not a source, record an error.
    if (parent_it == model_.nodes_.end() &&
        !model_.has_source_factory(*node->kind_)) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateOrphanNodes] Node {} has no parent",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Node {} has no parent", *node->name_));
    }
  }
}

void ModelTransaction::validate_childless_nodes() {
  holoflow_logger()->debug(
      "[ModelTransaction::validateChildlessNodes] Validating childless nodes");

  for (const auto &node : model_.nodes_) {
    // Check if the node has no children and is not a sink.
    if (node->children_.empty() && !model_.has_sink_factory(*node->kind_)) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateChildlessNodes] Node {} has no "
          "children",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Node {} has no children", *node->name_));
    }
  }
}

void ModelTransaction::validate_source_nodes() {
  holoflow_logger()->debug(
      "[ModelTransaction::validateSourceNodes] Validating source nodes");

  // Find the first source node.
  auto source_it = std::find_if(
      model_.nodes_.begin(), model_.nodes_.end(), [this](const auto &node) {
        return model_.has_source_factory(*node->kind_);
      });

  if (source_it == model_.nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] No "
                            "source node found");
    errors_.push_back(Error::make(ErrorType::NotFound, "No source node found"));
  }

  // Ensure there is only one source node.
  if (std::find_if(source_it + 1, model_.nodes_.end(),
                   [this](const auto &node) {
                     return model_.has_source_factory(*node->kind_);
                   }) != model_.nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] Multiple "
                            "source nodes found");
    errors_.push_back(
        Error::make(ErrorType::ConnectionError, "Multiple source nodes found"));
  }

  // Check that the source node has no parent.
  if (std::find_if(model_.nodes_.begin(), model_.nodes_.end(),
                   [&source_it](const auto &node) {
                     return std::any_of(
                         node->children_.begin(), node->children_.end(),
                         [&source_it](const auto &child) {
                           return child.get().name_ == (*source_it)->name_;
                         });
                   }) != model_.nodes_.end()) {
    holoflow_logger()->warn("[ModelTransaction::validateSourceNodes] Source "
                            "node {} has a parent",
                            *(*source_it)->name_);
    errors_.push_back(Error::make(ErrorType::ConnectionError,
                                  "Source node {} has a parent",
                                  *(*source_it)->name_));
  }
}

void ModelTransaction::validate_sink_nodes() {
  holoflow_logger()->debug(
      "[ModelTransaction::validateSinkNodes] Validating sink nodes");

  for (const auto &node : model_.nodes_) {
    // Check if the node is a sink and has children.
    if (model_.has_sink_factory(*node->kind_) && !node->children_.empty()) {
      holoflow_logger()->warn(
          "[ModelTransaction::validateSinkNodes] Sink node {} has children",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::ConnectionError,
                                    "Sink node {} has children", *node->name_));
    }
  }
}

void ModelTransaction::assign_cuda_streams() {
  holoflow_logger()->debug("[ModelTransaction::assign_cuda_streams] "
                           "Assigning CUDA streams");

  AssignCudaStreamVisitor visitor(model_.streams_, errors_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::perform_type_checking() {
  holoflow_logger()->debug("[ModelTransaction::perform_type_checking] "
                           "Performing type checking");

  TypeCheckVisitor visitor(
      errors_, model_.task_factories_map_, model_.accumulator_factories_map_,
      model_.source_factories_map_, model_.sink_factories_map_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::perform_single_inlined_child_checking() {
  holoflow_logger()->debug(
      "[ModelTransaction::perform_single_inlined_child_checking] "
      "Performing single inlined child checking");

  for (const auto &node : model_.nodes_) {
    size_t nb_inlined_children = std::count_if(
        node->children_.begin(), node->children_.end(),
        [](const auto &child) { return child.get().is_inlined(); });

    if (nb_inlined_children > 1) {
      holoflow_logger()->warn(
          "[ModelTransaction::perform_single_inlined_child_checking] "
          "Single inlined child check failed at node: {}",
          *node->name_);
      errors_.push_back(Error::make(ErrorType::InternalError,
                                    "Single inlined child check failed at "
                                    "node: {}",
                                    *node->name_));
    }
  }
}

void ModelTransaction::perform_non_inlined_child_checking() {
  holoflow_logger()->debug(
      "[ModelTransaction::perform_non_inlined_child_checking] "
      "Performing non inlined child checking");

  NonInlinedChildCheckVisitor visitor(errors_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::reset_non_accumulator_tensors() {
  holoflow_logger()->debug(
      "[ModelTransaction::reset_non_accumulator_tensors] Resetting non "
      "accumulator tensors");

  for (auto &node : model_.nodes_) {
    if (auto *task = dynamic_cast<Model::TaskNode *>(node.get())) {
      task->itens_id_ = std::nullopt;
      task->otens_id_ = std::nullopt;
    } else if (auto *source = dynamic_cast<Model::SourceNode *>(node.get())) {
      source->otens_id_ = std::nullopt;
    } else if (auto *sink = dynamic_cast<Model::SinkNode *>(node.get())) {
      sink->itens_id_ = std::nullopt;
    }
  }
}

void ModelTransaction::assign_accumulator_tensors() {
  holoflow_logger()->debug("[ModelTransaction::assign_accumulator_tensors] "
                           "Assigning accumulator tensors");

  AssignAccumulatorTensorsVisitor visitor(model_.next_id_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::assign_inlined_task_tensors() {
  holoflow_logger()->debug("[ModelTransaction::assign_inlined_task_tensors] "
                           "Assigning inlined task tensors");

  AssignInlinedTaskTensorsVisitor visitor(model_.next_id_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::assign_non_inlined_task_tensors() {
  holoflow_logger()->debug(
      "[ModelTransaction::assign_non_inlined_task_tensors] Assigning "
      "non-inlined task tensors");

  AssignNonInlinedTaskTensorsVisitor visitor(model_.next_id_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

void ModelTransaction::call_factories() {
  holoflow_logger()->debug("[ModelTransaction::call_factories] Calling "
                           "factories");

  CallFactoriesVisitor visitor(
      model_.task_factories_map_, model_.accumulator_factories_map_,
      model_.source_factories_map_, model_.sink_factories_map_, errors_);
  auto root = get_root();
  DH_CHECK(root);
  root->accept(visitor);
}

tl::expected<void, Error> ModelTransaction::commit() {
  holoflow_logger()->debug("[ModelTransaction::commit] Committing changes");

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  validate_orphan_nodes();
  validate_childless_nodes();
  validate_source_nodes();
  validate_sink_nodes();

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  assign_cuda_streams();
  perform_type_checking();

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  perform_single_inlined_child_checking();
  perform_non_inlined_child_checking();

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  reset_non_accumulator_tensors();
  assign_accumulator_tensors();
  assign_inlined_task_tensors();
  assign_non_inlined_task_tensors();
  assign_inlined_task_tensors();
  call_factories();

  if (!errors_.empty()) {
    return tl::unexpected(Error::aggregate(
        ErrorType::TransactionError, "Transaction errors occurred", errors_));
  }

  holoflow_logger()->debug(
      "[ModelTransaction::commit] Transaction committed successfully");

  return {};
}

} // namespace dh::v2