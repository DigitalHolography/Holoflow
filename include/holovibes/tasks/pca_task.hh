#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/cublas.hh"
#include "curaii/curaii.hh"
#include "curaii/cusolver_dn.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class PCATask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class PCATaskFactory;

private:
  PCATask(const TaskMeta &meta, cudaStream_t stream, CublasHandle cublas_handle,
          CusolverDnHandle cusolver_handle, CusolverDnParams cusolver_params,
          unique_device_ptr<cuFloatComplex> d_cov_matrix,
          unique_device_ptr<float> d_eigenvalues, unique_device_ptr<int> d_info,
          unique_host_ptr<uint8_t> h_workspace,
          unique_device_ptr<uint8_t> d_workspace, size_t h_workspace_size,
          size_t d_workspace_size, size_t begin, size_t end);

  CublasHandle cublas_handle_;
  CusolverDnHandle cusolver_handle_;
  CusolverDnParams cusolver_params_;
  unique_device_ptr<cuFloatComplex> d_cov_matrix_;
  unique_device_ptr<float> d_eigenvalues_;
  unique_device_ptr<int> d_info_;
  unique_host_ptr<uint8_t> h_workspace_;
  unique_device_ptr<uint8_t> d_workspace_;
  size_t h_workspace_size_;
  size_t d_workspace_size_;
  size_t begin_;
  size_t end_;
};

class PCATaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params);

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream);

private:
  struct Params {
    size_t begin;
    size_t end;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, begin, end);
  };
};

} // namespace dh