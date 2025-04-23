#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class MemcpyTask : public Task {
public:
  void run(TensorView itens, TensorView otens) override;

  friend class MemcpyTaskFactory;

private:
  MemcpyTask(const TaskMeta &meta, cudaStream_t stream, cudaMemcpyKind kind);

  cudaMemcpyKind kind_;
};

class MemcpyTaskFactory : public TaskFactory {
public:
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               cudaStream_t stream) override;

private:
  struct Params {
    std::string kind;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, kind);
  };
};

} // namespace dh
