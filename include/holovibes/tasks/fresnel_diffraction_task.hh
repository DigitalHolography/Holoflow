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

class FresnelDiffractionTask : public Task {
public:
  ~FresnelDiffractionTask() = default;

  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class FresnelDiffractionTaskFactory;

private:
  FresnelDiffractionTask(const TaskMeta &meta, cudaStream_t stream,
                         float lambda, float z, float pixel_size,
                         bool skip_phase_shift,
                         unique_device_ptr<cuFloatComplex> lens,
                         CufftHandle handle);

  float lambda_;
  float z_;
  float pixel_size_;
  bool skip_phase_shift_;
  unique_device_ptr<cuFloatComplex> lens_;
  CufftHandle handle_;
};

class FresnelDiffractionTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params);

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream);

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