#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "holoflow/accumulator.hh"
#include "holoflow/model.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class ModelBuilderVisitor;

class ModelNodeBuilder {
public:
  using Child = std::reference_wrapper<ModelNodeBuilder>;

  virtual ~ModelNodeBuilder() = default;

  const std::string &name() const;

  const std::string &kind() const;

  const json &params() const;

  cudaStream_t stream() const;

  std::span<Child> children();

  std::span<const Child> children() const;

  void set_name(const std::string &name);

  void set_kind(const std::string &kind);

  void set_params(const json &params);

  void set_stream(cudaStream_t stream);

  void add_child(ModelNodeBuilder &child);
  virtual void accept(ModelBuilderVisitor &visitor) = 0;

  virtual std::unique_ptr<ModelNode> build() = 0;

protected:
  std::optional<std::string> name_;
  std::optional<std::string> kind_;
  std::optional<json> params_;
  std::optional<cudaStream_t> stream_;
  std::vector<std::reference_wrapper<ModelNodeBuilder>> children_;
};

class TaskNodeBuilder : public ModelNodeBuilder {
public:
  int itens_id() const;

  int otens_id() const;

  Task &task() const;

  const TaskMeta &task_meta() const;

  std::optional<int> get_itens_id() const;

  std::optional<int> get_otens_id() const;

  void set_task(std::unique_ptr<Task> task);

  void set_task_meta(const TaskMeta &task_meta);

  void set_itens_id(int id);

  void set_otens_id(int id);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Task> task_;
  std::optional<TaskMeta> task_meta_;
  std::optional<int> itens_id_;
  std::optional<int> otens_id_;
};

class AccumulatorNodeBuilder : public ModelNodeBuilder {
public:
  int itens_id() const;

  int otens_id() const;

  Accumulator &accumulator() const;

  const AccumulatorMeta &accumulator_meta() const;

  std::optional<int> get_itens_id() const;

  std::optional<int> get_otens_id() const;

  void set_accumulator(std::unique_ptr<Accumulator> accumulator);

  void set_accumulator_meta(const AccumulatorMeta &accumulator_meta);

  void set_itens_id(int id);

  void set_otens_id(int id);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Accumulator> accumulator_;
  std::optional<AccumulatorMeta> accumulator_meta_;
  std::optional<int> itens_id_;
  std::optional<int> otens_id_;
};

class SourceNodeBuilder : public ModelNodeBuilder {
public:
  int otens_id() const;

  Source &source() const;

  const SourceMeta &source_meta() const;

  std::optional<int> get_otens_id() const;

  void set_source(std::unique_ptr<Source> source);

  void set_source_meta(const SourceMeta &source_meta);

  void set_otens_id(int id);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Source> source_;
  std::optional<SourceMeta> source_meta_;
  std::optional<int> otens_id_;
};

class SinkNodeBuilder : public ModelNodeBuilder {
public:
  int itens_id() const;

  Sink &sink() const;

  const SinkMeta &sink_meta() const;

  std::optional<int> get_itens_id() const;

  void set_sink(std::unique_ptr<Sink> sink);

  void set_sink_meta(const SinkMeta &sink_meta);

  void set_itens_id(int id);

  std::unique_ptr<ModelNode> build() override;

  void accept(ModelBuilderVisitor &visitor);

private:
  std::unique_ptr<Sink> sink_;
  std::optional<SinkMeta> sink_meta_;
  std::optional<int> itens_id_;
};

class ModelBuilderVisitor {
public:
  virtual ~ModelBuilderVisitor() = default;

  virtual void visit(TaskNodeBuilder &node) = 0;

  virtual void visit(AccumulatorNodeBuilder &node) = 0;

  virtual void visit(SourceNodeBuilder &node) = 0;

  virtual void visit(SinkNodeBuilder &node) = 0;
};

class ModelBuilder {
public:
  tl::expected<std::unique_ptr<Model>, Error>
  build(const ModelDescriptor &descriptor, const TensorMeta &imeta);
};

} // namespace dh