#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/curaii.hh"
#include "curaii/v2/cufft.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class FresnelDiffractionTask : public Task {
public:
  ~FresnelDiffractionTask() = default;

  void run(TensorView input, TensorView output) override;

  friend class FresnelDiffractionTaskFactory;

private:
  FresnelDiffractionTask(const TaskMeta &meta, CudaStreamRef stream,
                         float lambda, float z, float pixel_size,
                         bool skip_phase_shift,
                         unique_device_ptr<cuFloatComplex> lens,
                         curaii::cufft::Handle handle);

  float lambda_;
  float z_;
  float pixel_size_;
  bool skip_phase_shift_;
  unique_device_ptr<cuFloatComplex> lens_;
  curaii::cufft::Handle handle_;
};

class FresnelDiffractionTaskFactory : public TaskFactory {
public:
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               CudaStreamRef stream) override;

private:
  struct Params {
    float lambda;
    float z;
    float pixel_size;
    bool skip_phase_shift;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, lambda, z, pixel_size,
                                   skip_phase_shift);
  };
};

} // namespace dh