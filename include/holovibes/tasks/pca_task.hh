#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/v2/cublas.hh"
#include "curaii/v2/cuda.hh"
#include "curaii/v2/cusolver_common.hh"
#include "curaii/v2/cusolver_dn.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class PCATask : public Task {
public:
  void run(TensorView input, TensorView output) override;

  friend class PCATaskFactory;

private:
  PCATask(const TaskMeta &meta, cudaStream_t stream,
          curaii::cublas::Handle cublas_handle,
          curaii::cusolverdn::Handle cusolver_handle,
          curaii::cusolverdn::Params cusolver_params,
          curaii::cuda::unique_device_ptr<uint8_t> d_cov_matrix,
          curaii::cuda::unique_device_ptr<float> d_eigenvalues,
          curaii::cuda::unique_device_ptr<int> d_info,
          curaii::cuda::unique_host_ptr<uint8_t> h_workspace,
          curaii::cuda::unique_device_ptr<uint8_t> d_workspace,
          size_t h_workspace_size, size_t d_workspace_size, size_t begin,
          size_t end, bool is_complex);

  curaii::cublas::Handle cublas_handle_;
  curaii::cusolverdn::Handle cusolver_handle_;
  curaii::cusolverdn::Params cusolver_params_;
  curaii::cuda::unique_device_ptr<uint8_t> d_cov_matrix_;
  curaii::cuda::unique_device_ptr<float> d_eigenvalues_;
  curaii::cuda::unique_device_ptr<int> d_info_;
  curaii::cuda::unique_host_ptr<uint8_t> h_workspace_;
  curaii::cuda::unique_device_ptr<uint8_t> d_workspace_;
  size_t h_workspace_size_;
  size_t d_workspace_size_;
  size_t begin_;
  size_t end_;
  bool is_complex_;
};

class PCATaskFactory : public TaskFactory {
public:
  TaskMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Task> create(const TensorMeta &imeta, const json &params,
                               cudaStream_t stream) override;

private:
  struct Params {
    size_t begin;
    size_t end;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, begin, end);
  };
};

} // namespace dh