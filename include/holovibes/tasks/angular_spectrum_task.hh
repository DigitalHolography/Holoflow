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

class AngularSpectrumTask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class AngularSpectrumTaskFactory;

private:
  AngularSpectrumTask(const TaskMeta &meta, CudaStreamRef stream, float lambda,
                      float z, float pixel_size,
                      unique_device_ptr<cuFloatComplex> lens,
                      CufftHandle handle);
  float lambda_;
  float z_;
  float pixel_size_;
  unique_device_ptr<cuFloatComplex> lens_;
  CufftHandle handle_;
};

class AngularSpectrumTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params) override;

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params,
         CudaStreamRef stream) override;

private:
  struct Params {
    float lambda;
    float z;
    float pixel_size;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, lambda, z, pixel_size);
  };
};

} // namespace dh