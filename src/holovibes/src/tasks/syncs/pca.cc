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

#include "pca.hh"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks {

namespace {

#define LAPACKE_CHECK(expr)                                                                        \
  do {                                                                                             \
    lapack_int err__ = (expr);                                                                     \
    if (err__ != 0) {                                                                              \
      logger()->error("[Pca] LAPACKE error at {}:{} (code={})", __FILE__, __LINE__, err__);        \
      throw std::runtime_error("LAPACKE error");                                                   \
    }                                                                                              \
  } while (false)

} // namespace

void to_json(nlohmann::json &j, const PcaSettings &settings) {
  j = nlohmann::json{
      {"begin", settings.begin},
      {"end", settings.end},
  };
}

void from_json(const nlohmann::json &j, PcaSettings &settings) {
  j.at("begin").get_to(settings.begin);
  j.at("end").get_to(settings.end);
}

Pca::Pca(const PcaSettings &settings, curaii::CublasHandle &&cublas_handle,
         curaii::CusolverDnHandle &&cusolver_handle, curaii::CusolverDnParams &&cusolver_params,
         DevPtr<cuFloatComplex> &&d_cov, DevPtr<float> &&d_eigvals, DevPtr<int> &&d_info,
         DevPtr<std::uint8_t> &&d_workspace, HostPtr<std::uint8_t> &&h_workspace,
         HostPtr<lapack_complex_float> &&h_cov, HostPtr<float> &&h_eigvals,
         HostPtr<lapack_complex_float> &&h_eigvecs, HostPtr<std::int64_t> &&h_meig,
         HostPtr<lapack_int> &&h_isuppz, std::size_t d_workspace_size, std::size_t h_workspace_size,
         cudaStream_t stream)
    : settings_(settings), cublas_handle_(std::move(cublas_handle)),
      cusolver_handle_(std::move(cusolver_handle)), cusolver_params_(std::move(cusolver_params)),
      d_cov_(std::move(d_cov)), d_eigvals_(std::move(d_eigvals)), d_info_(std::move(d_info)),
      d_workspace_(std::move(d_workspace)), h_workspace_(std::move(h_workspace)),
      h_cov_(std::move(h_cov)), h_eigvals_(std::move(h_eigvals)), h_eigvecs_(std::move(h_eigvecs)),
      h_meig_(std::move(h_meig)), h_isuppz_(std::move(h_isuppz)),
      d_workspace_size_(d_workspace_size), h_workspace_size_(h_workspace_size), stream_(stream) {}

holoflow::core::OpResult Pca::execute(holoflow::core::SyncCtx &ctx) {
  auto &iview = ctx.inputs[0];
  auto &oview = ctx.outputs[0];
  auto *idata = reinterpret_cast<cuFloatComplex *>(iview.data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(oview.data);

  const auto   &idesc      = iview.desc;
  const int     n_features = static_cast<int>(idesc.shape.at(0));
  const int     height     = static_cast<int>(idesc.shape.at(1));
  const int     width      = static_cast<int>(idesc.shape.at(2));
  const int     n_samples  = height * width;
  const int     components = static_cast<int>(settings_.end - settings_.begin);
  const int64_t il         = static_cast<int64_t>(settings_.begin) + 1;
  const int64_t iu         = static_cast<int64_t>(settings_.end);

  const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
  const cuFloatComplex beta  = make_cuFloatComplex(0.0f, 0.0f);

  // Compute covariance matrix:
  // COV = I^H * I, where I is the input data matrix with dimensions
  // (n_samples x n_features). This operation yields an (n_features x
  // n_features) Hermitian matrix representing the covariance between each
  // pair of features (assuming the data has been centered).
  CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_C, CUBLAS_OP_N, n_features, n_features,
                            n_samples, &alpha, idata, CUDA_C_32F, n_samples, idata, CUDA_C_32F,
                            n_samples, &beta, d_cov_.get(), CUDA_C_32F, n_features,
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));

  // Compute eigenvectors:
  // Perform an eigen decomposition on the covariance matrix to extract both
  // eigenvalues and eigenvectors.
  if (n_features <= cpu_heuristic_max_) { // Use CPU for small problems
    const std::size_t elem_count = static_cast<std::size_t>(n_features) * n_features;
    auto              bytes      = elem_count * sizeof(cuFloatComplex);
    auto              kind       = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpyAsync(h_cov_.get(), d_cov_.get(), bytes, kind, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    lapack_int m = 0;
    LAPACKE_CHECK(LAPACKE_cheevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', n_features, h_cov_.get(),
                                 n_features, 0.0f, 0.0f, il, iu, 0.0f, &m, h_eigvals_.get(),
                                 h_eigvecs_.get(), n_features, h_isuppz_.get()));

    bool success = m == components;
    HOLOVIBES_CHECK(success, "[Pca] unexpected eigenvector count returned by LAPACKE_cheevr");
    bytes = static_cast<std::size_t>(n_features) * components * sizeof(cuFloatComplex);
    kind  = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpyAsync(d_cov_.get(), h_eigvecs_.get(), bytes, kind, stream_));
  }

  else { // Use GPU for large problems
    float vl = 0.0f;
    float vu = 0.0f;
    CUSOLVER_CHECK(
        cusolverDnXsyevdx(cusolver_handle_.get(), cusolver_params_.get(), CUSOLVER_EIG_MODE_VECTOR,
                          CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features, CUDA_C_32F,
                          d_cov_.get(), n_features, &vl, &vu, il, iu, h_meig_.get(), CUDA_R_32F,
                          d_eigvals_.get(), CUDA_C_32F, d_workspace_.get(), d_workspace_size_,
                          h_workspace_.get(), h_workspace_size_, d_info_.get()));
  }

  // Project data:
  // Multiply the input data by the eigenvector matrix
  // to transform the data into the principal component space. Each row of
  // the output corresponds to the projection of the original data onto the
  // new basis defined by the eigenvectors.
  cuFloatComplex *eigvecs = reinterpret_cast<cuFloatComplex *>(d_cov_.get());
  CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples, components,
                            n_features, &alpha, idata, CUDA_C_32F, n_samples, eigvecs, CUDA_C_32F,
                            n_features, &beta, odata, CUDA_C_32F, n_samples,
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult PcaFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[PcaFactory::infer] error: {}", msg);
      throw std::invalid_argument("PcaFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<PcaSettings>();

  // Validate
  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() == 3, "expected input rank 3");
  check(idesc.dtype == holoflow::core::DType::CF32, "expected input dtype CF32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");
  check(settings.end <= idesc.shape.at(0), "expected end <= n_features");

  // Success
  auto odesc     = idesc;
  odesc.shape[0] = settings.end - settings.begin;
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  // Validate
  this->infer(input_descs, jsettings);
  auto        settings   = jsettings.get<PcaSettings>();
  const auto &idesc      = input_descs[0];
  const int   n_features = static_cast<int>(idesc.shape.at(0));

  // Cov matrix
  curaii::CublasHandle cublas_handle;
  CUBLAS_CHECK(cublasSetStream(cublas_handle.get(), ctx.stream));
  auto num_elems = static_cast<std::size_t>(n_features) * n_features;
  auto d_cov     = curaii::make_unique_device_ptr<cuFloatComplex>(num_elems);

  // GPU eigenvecs
  curaii::CusolverDnHandle cusolver_handle;
  CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle.get(), ctx.stream));
  CUSOLVER_CHECK(cusolverDnSetDeterministicMode(cusolver_handle.get(),
                                                CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS));

  curaii::CusolverDnParams cusolver_params;
  auto                     d_eigvals        = curaii::make_unique_device_ptr<float>(n_features);
  auto                     d_info           = curaii::make_unique_device_ptr<int>(1);
  auto                     h_meig           = curaii::make_unique_host_ptr<std::int64_t>(1);
  size_t                   d_workspace_size = 0;
  size_t                   h_workspace_size = 0;
  CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
      cusolver_handle.get(), cusolver_params.get(), CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_LOWER, n_features, CUDA_C_32F, d_cov.get(), n_features, nullptr, nullptr,
      static_cast<int64_t>(settings.begin) + 1, static_cast<int64_t>(settings.end), h_meig.get(),
      CUDA_R_32F, d_eigvals.get(), CUDA_C_32F, &d_workspace_size, &h_workspace_size));
  DevPtr<uint8_t>  d_workspace = curaii::make_unique_device_ptr<uint8_t>(d_workspace_size);
  HostPtr<uint8_t> h_workspace = curaii::make_unique_host_ptr<uint8_t>(h_workspace_size);

  // CPU eigenvecs
  auto h_cov     = curaii::make_unique_host_ptr<lapack_complex_float>(num_elems);
  auto h_eigvals = curaii::make_unique_host_ptr<float>(settings.end - settings.begin);
  auto h_eigvecs = curaii::make_unique_host_ptr<lapack_complex_float>(num_elems);
  auto h_isuppz  = curaii::make_unique_host_ptr<lapack_int>(2 * n_features);

  // Success
  auto *task = new Pca(
      settings, std::move(cublas_handle), std::move(cusolver_handle), std::move(cusolver_params),
      std::move(d_cov), std::move(d_eigvals), std::move(d_info), std::move(d_workspace),
      std::move(h_workspace), std::move(h_cov), std::move(h_eigvals), std::move(h_eigvecs),
      std::move(h_meig), std::move(h_isuppz), d_workspace_size, h_workspace_size, ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks
