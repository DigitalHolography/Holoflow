#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/curaii.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class IdentityTask : public Task {
public:
  IdentityTask() = default;

  void run(TensorView input, TensorView output) override;

  friend class IdentityTaskFactory;

private:
  IdentityTask(const TaskMeta &meta, CudaStreamRef stream);
};

class IdentityTaskFactory : public TaskFactory {
public:
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               CudaStreamRef stream) override;
};

} // namespace dh