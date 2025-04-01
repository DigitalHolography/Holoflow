#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/curaii.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class PercentileClipTask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class PercentileClipTaskFactory;

private:
  PercentileClipTask(const TaskMeta &meta, cudaStream_t stream,
                     unique_device_ptr<float> lower_threshold,
                     unique_device_ptr<float> upper_threshold,
                     unique_device_ptr<uint8_t> d_temp_storage,
                     size_t temp_storage_bytes);

  unique_device_ptr<float> lower_threshold_;
  unique_device_ptr<float> upper_threshold_;
  unique_device_ptr<uint8_t> d_temp_storage_;
  size_t temp_storage_bytes_;
};

class PercentileClipTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params);

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream);
};

} // namespace dh