#include "holovibes/tasks/identity_task.hh"

#include "curaii/v2/cuda.hh"
#include "holovibes/holovibes.hh"

namespace dh {

IdentityTask::IdentityTask(const TaskMeta &meta, cudaStream_t stream)
    : Task(meta, stream) {}

void IdentityTask::run(TensorView input, TensorView output) {
  auto kind = input.memory_location() == MemoryLocation::HOST
                  ? cudaMemcpyHostToHost
                  : cudaMemcpyDeviceToDevice;
  CUDA_CHECK(cudaMemcpyAsync(output.data(), input.data(), input.size_in_bytes(),
                             kind, stream_));
}

TaskMeta IdentityTaskFactory::type_check(const TensorMeta &imeta,
                                         const json &) {
  return TaskMeta(imeta, imeta, false);
}

std::unique_ptr<Task> IdentityTaskFactory::create(const TensorMeta &imeta,
                                                  const json &jparams,
                                                  cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);

  // 2) Assemble task
  auto *task = new IdentityTask(meta, stream);
  return std::unique_ptr<IdentityTask>(task);
}

} // namespace dh