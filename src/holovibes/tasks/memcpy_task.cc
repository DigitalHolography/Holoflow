#include "holovibes/tasks/memcpy_task.hh"

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"

namespace dh {

// ==========================================================================
//                     MemcpyTask Implementation
// ==========================================================================

MemcpyTask::MemcpyTask(const TaskMeta &meta, cudaStream_t stream,
                       cudaMemcpyKind kind)
    : Task(meta, stream), kind_(kind) {}

void MemcpyTask::run(TensorView input, TensorView output) {
  CUDA_CHECK(cudaMemcpyAsync(output.data(), input.data(), input.size_in_bytes(),
                             kind_, stream_));
}

// ==========================================================================
//                     MemcpyTaskFactory Implementation
// ==========================================================================

TaskMeta MemcpyTaskFactory::type_check(const TensorMeta &imeta,
                                       const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.kind == "HOST_TO_HOST" || params.kind == "HOST_TO_DEVICE" ||
            params.kind == "DEVICE_TO_HOST" ||
            params.kind == "DEVICE_TO_DEVICE",
        "memcpy kind invalid");

  // 2) Tensor meta sanity
  check(((params.kind == "HOST_TO_HOST" || params.kind == "HOST_TO_DEVICE")
             ? (imeta.memory_location() == MemoryLocation::HOST)
             : (imeta.memory_location() == MemoryLocation::DEVICE)),
        "invalid memcpy: tensor is in the wrong memory for this kind");

  // 3) Success
  MemoryLocation location;
  if (params.kind == "HOST_TO_HOST" || params.kind == "DEVICE_TO_HOST") {
    location = MemoryLocation::HOST;
  } else {
    location = MemoryLocation::DEVICE;
  }

  TensorMeta ometa(imeta.data_type(), location, imeta.shape());
  return TaskMeta(imeta, ometa, false);
}

std::unique_ptr<Task> MemcpyTaskFactory::create(const TensorMeta &imeta,
                                                const json &jparams,
                                                cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  // 2) Extract memcpy kind
  cudaMemcpyKind kind;
  if (params.kind == "HOST_TO_HOST") {
    kind = cudaMemcpyHostToHost;
  } else if (params.kind == "HOST_TO_DEVICE") {
    kind = cudaMemcpyHostToDevice;
  } else if (params.kind == "DEVICE_TO_HOST") {
    kind = cudaMemcpyDeviceToHost;
  } else if (params.kind == "DEVICE_TO_DEVICE") {
    kind = cudaMemcpyDeviceToDevice;
  } else {
    DH_BUG("unreachable statement reached");
  }

  // 3) Assemble task
  auto *task = new MemcpyTask(meta, stream, kind);
  return std::unique_ptr<MemcpyTask>(task);
}

} // namespace dh