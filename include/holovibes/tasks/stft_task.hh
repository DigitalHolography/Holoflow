#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/v2/cufft.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class STFTTask : public Task {
  void run(TensorView input, TensorView output) override;

  friend class STFTTaskFactory;

private:
  STFTTask(const TaskMeta &meta, cudaStream_t stream,
           curaii::cufft::Handle handle);

  curaii::cufft::Handle handle_;
};

class STFTTaskFactory : public TaskFactory {
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               cudaStream_t stream) override;
};

} // namespace dh