// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mkl_lapacke.h>
#include <nlohmann/json.hpp>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holovibes::tasks::syncs {

struct PcaSettings {
  int begin;
  int end;
};

void to_json(nlohmann::json &j, const PcaSettings &settings);
void from_json(const nlohmann::json &j, PcaSettings &settings);

class Pca : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Pca(const PcaSettings &settings, curaii::CublasHandle &&cublas_handle,
      curaii::CusolverDnHandle &&cusolver_handle, curaii::CusolverDnParams &&cusolver_params,
      DevPtr<cuFloatComplex> &&d_cov, DevPtr<float> &&d_eigvals, DevPtr<int> &&d_info,
      DevPtr<uint8_t> &&d_workspace, HostPtr<uint8_t> &&h_workspace,
      HostPtr<lapack_complex_float> &&h_cov, HostPtr<float> &&h_eigvals,
      HostPtr<lapack_complex_float> &&h_eigvecs, HostPtr<int64_t> &&h_meig,
      HostPtr<lapack_int> &&h_isuppz, size_t d_workspace_size, size_t h_workspace_size,
      cudaStream_t stream);

  friend class PcaFactory;

  PcaSettings                   settings_;
  curaii::CublasHandle          cublas_handle_;
  curaii::CusolverDnHandle      cusolver_handle_;
  curaii::CusolverDnParams      cusolver_params_;
  DevPtr<cuFloatComplex>        d_cov_;
  DevPtr<float>                 d_eigvals_;
  DevPtr<int>                   d_info_;
  DevPtr<uint8_t>               d_workspace_;
  HostPtr<uint8_t>              h_workspace_;
  HostPtr<lapack_complex_float> h_cov_;
  HostPtr<float>                h_eigvals_;
  HostPtr<lapack_complex_float> h_eigvecs_;
  HostPtr<int64_t>              h_meig_;
  HostPtr<lapack_int>           h_isuppz_;
  size_t                        d_workspace_size_;
  size_t                        h_workspace_size_;
  cudaStream_t                  stream_;

  static constexpr int cpu_heuristic_max_ = 32;
};

class PcaFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks::syncs
