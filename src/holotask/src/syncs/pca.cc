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

#include "holotask/syncs/pca.hh"

#include <limits>
#include <mkl_lapacke.h>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>
#include <string>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"

#include "bug.hh"
#include "logger.hh"

namespace holotask::syncs {

namespace {

#define LAPACKE_CHECK(expr)                                                                        \
  do {                                                                                             \
    lapack_int err__ = (expr);                                                                     \
    if (err__ != 0) {                                                                              \
      logger()->error("[Pca] LAPACKE error at {}:{} (code={})", __FILE__, __LINE__, err__);        \
      throw std::runtime_error("LAPACKE error");                                                   \
    }                                                                                              \
  } while (false)

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

// ----------------------------------------------------------------------------
// Type Traits for Real (float) vs Complex (cuFloatComplex) dispatch
// ----------------------------------------------------------------------------

template <typename T> struct PcaTraits;

template <> struct PcaTraits<float> {
  static constexpr auto cuda_type    = CUDA_R_32F;
  static constexpr auto compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  using lapack_type                  = float;

  static float alpha() { return 1.0f; }
  static float beta() { return 0.0f; }

  static lapack_int eigen(int matrix_layout, char jobz, char range, char uplo, lapack_int n,
                          lapack_type *a, lapack_int lda, float vl, float vu, lapack_int il,
                          lapack_int iu, float abstol, lapack_int *m, float *w, lapack_type *z,
                          lapack_int ldz, lapack_int *isuppz) {
    return LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w,
                          z, ldz, isuppz);
  }
};

template <> struct PcaTraits<cuFloatComplex> {
  static constexpr auto cuda_type    = CUDA_C_32F;
  static constexpr auto compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  using lapack_type                  = lapack_complex_float;

  static cuFloatComplex alpha() { return make_cuFloatComplex(1.0f, 0.0f); }
  static cuFloatComplex beta() { return make_cuFloatComplex(0.0f, 0.0f); }

  static lapack_int eigen(int matrix_layout, char jobz, char range, char uplo, lapack_int n,
                          lapack_type *a, lapack_int lda, float vl, float vu, lapack_int il,
                          lapack_int iu, float abstol, lapack_int *m, float *w, lapack_type *z,
                          lapack_int ldz, lapack_int *isuppz) {
    return LAPACKE_cheevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w,
                          z, ldz, isuppz);
  }
};

// ----------------------------------------------------------------------------
// PCA Task Implementation
// ----------------------------------------------------------------------------

template <typename T> class PcaTask : public holoflow::core::ISyncTask {
public:
  PcaTask(const PcaSettings &settings, const holoflow::core::SyncCreateCtx &ctx, int n_features)
      : settings_(settings), stream_(ctx.stream), n_features_(n_features), iteration_count_(0) {

    // Initialize handles
    CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(cusolver_handle_.get(),
                                                  CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS));

    const std::size_t num_elems = static_cast<std::size_t>(n_features_) * n_features_;

    // Allocate core buffers
    d_cov_     = curaii::make_unique_device_ptr<T>(num_elems);
    d_eigvals_ = curaii::make_unique_device_ptr<float>(n_features_);
    d_info_    = curaii::make_unique_device_ptr<int>(1);
    h_meig_    = curaii::make_unique_host_ptr<int64_t>(1);

    // Query and allocate workspace for cuSOLVER
    CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
        cusolver_handle_.get(), cusolver_params_.get(), CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features_, PcaTraits<T>::cuda_type,
        d_cov_.get(), n_features_, nullptr, nullptr, static_cast<int64_t>(settings_.begin) + 1,
        static_cast<int64_t>(settings_.end), h_meig_.get(), CUDA_R_32F, d_eigvals_.get(),
        PcaTraits<T>::cuda_type, &d_workspace_size_, &h_workspace_size_));

    d_workspace_ = curaii::make_unique_device_ptr<uint8_t>(d_workspace_size_);
    h_workspace_ = curaii::make_unique_host_ptr<uint8_t>(h_workspace_size_);

    // Allocate CPU buffers for fallback (LAPACKE)
    h_cov_     = curaii::make_unique_host_ptr<typename PcaTraits<T>::lapack_type>(num_elems);
    h_eigvals_ = curaii::make_unique_host_ptr<float>(settings_.components());
    h_eigvecs_ = curaii::make_unique_host_ptr<typename PcaTraits<T>::lapack_type>(num_elems);
    h_isuppz_  = curaii::make_unique_host_ptr<lapack_int>(2 * n_features_);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range r("PCA Sync Task");
    auto               &iview = ctx.inputs[0];
    auto               &oview = ctx.outputs[0];
    auto               *idata = reinterpret_cast<T *>(iview.data());
    auto               *odata = reinterpret_cast<T *>(oview.data());

    const auto   &idesc      = iview.desc;
    const int     height     = static_cast<int>(idesc.shape.at(1));
    const int     width      = static_cast<int>(idesc.shape.at(2));
    const int     n_samples  = height * width;
    const int     components = settings_.components();
    const int64_t il         = static_cast<int64_t>(settings_.begin) + 1;
    const int64_t iu         = static_cast<int64_t>(settings_.end);

    const T alpha = PcaTraits<T>::alpha();
    const T beta  = PcaTraits<T>::beta();

    bool should_update = (iteration_count_ % settings_.update_rate == 0);

    if (should_update) {
      // 1. Compute covariance matrix: COV = I^H * I
      // Yields an (n_features x n_features) matrix representing covariance between features.
      CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_C, CUBLAS_OP_N, n_features_,
                                n_features_, n_samples, &alpha, idata, PcaTraits<T>::cuda_type,
                                n_samples, idata, PcaTraits<T>::cuda_type, n_samples, &beta,
                                d_cov_.get(), PcaTraits<T>::cuda_type, n_features_,
                                PcaTraits<T>::compute_type, CUBLAS_GEMM_DEFAULT));

      // 2. Compute eigenvectors
      if (n_features_ <= cpu_heuristic_max_) {
        // Use CPU for small problems
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        nvtx3::scoped_range r_cpu("CPU eigen decomposition");
        const std::size_t   bytes = static_cast<std::size_t>(n_features_) * n_features_ * sizeof(T);
        CUDA_CHECK(
            cudaMemcpyAsync(h_cov_.get(), d_cov_.get(), bytes, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        lapack_int m = 0;
        LAPACKE_CHECK(PcaTraits<T>::eigen(
            LAPACK_COL_MAJOR, 'V', 'I', 'L', n_features_, h_cov_.get(), n_features_, 0.0f, 0.0f, il,
            iu, 0.0f, &m, h_eigvals_.get(), h_eigvecs_.get(), n_features_, h_isuppz_.get()));

        HOLOVIBES_CHECK(m == components, "[Pca] unexpected eigenvector count from LAPACKE");

        const std::size_t vec_bytes =
            static_cast<std::size_t>(n_features_) * components * sizeof(T);
        CUDA_CHECK(cudaMemcpyAsync(d_cov_.get(), h_eigvecs_.get(), vec_bytes,
                                   cudaMemcpyHostToDevice, stream_));
      } else {
        // Use GPU for large problems
        nvtx3::scoped_range r_gpu("GPU eigen decomposition");
        float               vl = 0.0f, vu = 0.0f;
        CUSOLVER_CHECK(cusolverDnXsyevdx(
            cusolver_handle_.get(), cusolver_params_.get(), CUSOLVER_EIG_MODE_VECTOR,
            CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features_, PcaTraits<T>::cuda_type,
            d_cov_.get(), n_features_, &vl, &vu, il, iu, h_meig_.get(), CUDA_R_32F,
            d_eigvals_.get(), PcaTraits<T>::cuda_type, d_workspace_.get(), d_workspace_size_,
            h_workspace_.get(), h_workspace_size_, d_info_.get()));
      }
    }

    // 3. Project data: Multiply input data by the eigenvector matrix
    T *eigvecs = reinterpret_cast<T *>(d_cov_.get());
    CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples, components,
                              n_features_, &alpha, idata, PcaTraits<T>::cuda_type, n_samples,
                              eigvecs, PcaTraits<T>::cuda_type, n_features_, &beta, odata,
                              PcaTraits<T>::cuda_type, n_samples, PcaTraits<T>::compute_type,
                              CUBLAS_GEMM_DEFAULT));

    iteration_count_++;
    return holoflow::core::OpResult::Ok;
  }

private:
  PcaSettings  settings_;
  cudaStream_t stream_;
  int          n_features_;
  uint64_t     iteration_count_;

  curaii::CublasHandle     cublas_handle_;
  curaii::CusolverDnHandle cusolver_handle_;
  curaii::CusolverDnParams cusolver_params_;

  DevPtr<T>        d_cov_;
  DevPtr<float>    d_eigvals_;
  DevPtr<int>      d_info_;
  DevPtr<uint8_t>  d_workspace_;
  HostPtr<uint8_t> h_workspace_;

  HostPtr<typename PcaTraits<T>::lapack_type> h_cov_;
  HostPtr<float>                              h_eigvals_;
  HostPtr<typename PcaTraits<T>::lapack_type> h_eigvecs_;
  HostPtr<int64_t>                            h_meig_;
  HostPtr<lapack_int>                         h_isuppz_;

  size_t d_workspace_size_{0};
  size_t h_workspace_size_{0};

  static constexpr int cpu_heuristic_max_ = 64;
};

} // namespace

// ----------------------------------------------------------------------------
// JSON Serialization
// ----------------------------------------------------------------------------

void to_json(nlohmann::json &j, const PcaSettings &settings) {
  j = nlohmann::json{
      {"begin", settings.begin},
      {"end", settings.end},
      {"update_rate", settings.update_rate},
  };
}

void from_json(const nlohmann::json &j, PcaSettings &settings) {
  j.at("begin").get_to(settings.begin);
  j.at("end").get_to(settings.end);
  settings.update_rate = j.value("update_rate", 1);
}

// ----------------------------------------------------------------------------
// Factory Methods
// ----------------------------------------------------------------------------

holoflow::core::InferResult PcaFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[PcaFactory::infer] error: {}", msg);
      throw std::invalid_argument("PcaFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<PcaSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc = input_descs[0];
  check(idesc.rank() == 3, "expected input rank 3");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32,
        "expected input dtype CF32 or F32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");
  check(settings.end <= idesc.shape.at(0), "expected end <= n_features");
  check(settings.update_rate >= 1, "expected update_rate >= 1");

  auto odesc     = idesc;
  odesc.shape[0] = settings.components();

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

  this->infer(input_descs, jsettings); // Validate bounds & types

  auto        settings = jsettings.get<PcaSettings>();
  const auto &idesc    = input_descs[0];
  const int   n_feats  = static_cast<int>(idesc.shape.at(0));

  // Dispatch based on datatype
  if (idesc.dtype == holoflow::core::DType::F32) {
    return std::make_unique<PcaTask<float>>(settings, ctx, n_feats);
  } else {
    return std::make_unique<PcaTask<cuFloatComplex>>(settings, ctx, n_feats);
  }
}

} // namespace holotask::syncs