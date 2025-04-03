#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/cufft.hh"
#include "curaii/curaii.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class STFTTask : public Task {
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class STFTTaskFactory;

private:
  STFTTask(const TaskMeta &meta, cudaStream_t stream, CufftHandle handle);

  CufftHandle handle_;
};

class STFTTaskFactory : public TaskFactory {
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params);

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream);
};

} // namespace dh