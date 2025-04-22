#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/v2/cuda.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class ConvertTask : public Task {
public:
  enum class Conversion {
    U8_CF32_REAL,
    U16_CF32_REAL,
    F32_U8_SCALED,
    F32_U16_SCALED,
    CF32_F32_MODU,
    CF32_F32_ARGU
  };

  ConvertTask(const TaskMeta &meta, cudaStream_t stream, Conversion conv,
              size_t min_temp_storage_bytes,
              curaii::cuda::unique_device_ptr<uint8_t> d_min_temp_storage,
              curaii::cuda::unique_device_ptr<uint8_t> d_min,
              size_t max_temp_storage_bytes,
              curaii::cuda::unique_device_ptr<uint8_t> d_max_temp_storage,
              curaii::cuda::unique_device_ptr<uint8_t> d_max);

  void run(TensorView input, TensorView output) override;

private:
  Conversion conversion_;
  size_t min_temp_storage_bytes_;
  curaii::cuda::unique_device_ptr<uint8_t> d_min_temp_storage_;
  curaii::cuda::unique_device_ptr<uint8_t> d_min_;
  size_t max_temp_storage_bytes_;
  curaii::cuda::unique_device_ptr<uint8_t> d_max_temp_storage_ = nullptr;
  curaii::cuda::unique_device_ptr<uint8_t> d_max_ = nullptr;
};

class ConvertTaskFactory : public TaskFactory {
public:
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               cudaStream_t stream) override;

private:
  struct Params {
    std::string conversion;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, conversion);
  };
};

} // namespace dh