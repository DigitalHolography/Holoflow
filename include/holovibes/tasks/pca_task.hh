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

using json                          = nlohmann::json;
template <typename T> using DevPtr  = curaii::cuda::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::cuda::unique_host_ptr<T>;

namespace dh {

class PCATask : public Task {
public:
  void run(TensorView input, TensorView output) override;

  friend class PCATaskFactory;

private:
  PCATask(const TaskMeta &meta, cudaStream_t stream,
          curaii::cublas::Handle     cublas_handle,
          curaii::cusolverdn::Handle cusolver_handle,
          curaii::cusolverdn::Params cusolver_params,
          DevPtr<uint8_t> d_cov_matrix, DevPtr<float> d_eigenvalues,
          DevPtr<int> d_info, HostPtr<uint8_t> h_workspace,
          DevPtr<uint8_t> d_workspace, size_t h_workspace_size,
          size_t d_workspace_size, size_t begin, size_t end, bool is_complex);

  // Config
  size_t begin_;
  size_t end_;
  bool   is_complex_;

  // Handles / Ressources
  curaii::cublas::Handle     cublas_handle_;
  curaii::cusolverdn::Handle cusolver_handle_;
  curaii::cusolverdn::Params cusolver_params_;

  // Buffers
  DevPtr<uint8_t>  d_cov_matrix_;
  DevPtr<float>    d_eigenvalues_;
  DevPtr<int>      d_info_;
  HostPtr<uint8_t> h_workspace_;
  DevPtr<uint8_t>  d_workspace_;
  size_t           h_workspace_size_;
  size_t           d_workspace_size_;
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