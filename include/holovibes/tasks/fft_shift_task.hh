#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class FFTShiftTask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class FFTShiftTaskFactory;

private:
  FFTShiftTask(const TaskMeta &meta, CudaStreamRef stream);
};

class FFTShiftTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params) override;

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params,
         CudaStreamRef stream) override;
};

} // namespace dh