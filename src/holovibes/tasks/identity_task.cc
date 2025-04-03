#include "holovibes/tasks/identity_task.hh"

#include "holovibes/holovibes.hh"

namespace dh {

IdentityTask::IdentityTask(const TaskMeta &meta, cudaStream_t stream)
    : Task(meta, stream) {}

tl::expected<void, Error> IdentityTask::run(TensorView input,
                                            TensorView output) {
  cudaMemcpyAsync(output.data(), input.data(), input.size_in_bytes(),
                  cudaMemcpyDeviceToDevice, stream_);
  return {};
}

tl::expected<TaskMeta, Error>
IdentityTaskFactory::type_check(const TensorMeta &imeta, const json &) {
  return TaskMeta(imeta, imeta, false);
}

tl::expected<std::unique_ptr<Task>, Error>
IdentityTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                            cudaStream_t stream) {

  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();

  auto *task = new IdentityTask(meta, stream);
  return std::unique_ptr<IdentityTask>(task);
}

} // namespace dh