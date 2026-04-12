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
  // n_features: size of the feature dimension (shape[-3])
  // n_batch:    product of all leading dimensions (1 for rank-3 input)
  PcaTask(const PcaSettings &settings, const holoflow::core::TDesc &idesc,
          const holoflow::core::SyncCreateCtx &ctx, int n_features, size_t n_batch)
      : settings_(settings), idesc_(idesc), stream_(ctx.stream), n_features_(n_features),
        n_batch_(n_batch), iteration_count_(0) {

    // Initialize handles
    CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnSetDeterministicMode(cusolver_handle_.get(),
                                                  CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS));

    // Per-batch covariance matrix size
    auto cov_elems = n_batch_ * n_features_ * n_features_;

    // Allocate core buffers (scaled by n_batch)
    d_cov_     = curaii::make_unique_device_ptr<T>(cov_elems);
    d_eigvals_ = curaii::make_unique_device_ptr<float>(n_batch_ * n_features_);

    // Allocate info and meig arrays scaled by n_batch to prevent race conditions in the loop
    d_info_ = curaii::make_unique_device_ptr<int>(n_batch_);
    h_meig_ = curaii::make_unique_host_ptr<int64_t>(n_batch_);

    // Query workspace size for a single n_features x n_features problem (reused for all batches)
    const int64_t il = static_cast<int64_t>(settings_.begin) + 1;
    const int64_t iu = static_cast<int64_t>(settings_.end);
    CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
        cusolver_handle_.get(), cusolver_params_.get(), CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features_, PcaTraits<T>::cuda_type,
        d_cov_.get(), n_features_, nullptr, nullptr, il, iu, h_meig_.get(), CUDA_R_32F,
        d_eigvals_.get(), PcaTraits<T>::cuda_type, &d_workspace_size_, &h_workspace_size_));

    d_workspace_ = curaii::make_unique_device_ptr<uint8_t>(d_workspace_size_);
    h_workspace_ = curaii::make_unique_host_ptr<uint8_t>(h_workspace_size_);

    // CPU fallback buffers (n_batch full covariance matrices and eigenvector outputs)
    h_cov_     = curaii::make_unique_host_ptr<typename PcaTraits<T>::lapack_type>(cov_elems);
    h_eigvals_ = curaii::make_unique_host_ptr<float>(n_batch_ * settings_.components());
    h_eigvecs_ = curaii::make_unique_host_ptr<typename PcaTraits<T>::lapack_type>(cov_elems);
    h_isuppz_  = curaii::make_unique_host_ptr<lapack_int>(n_batch_ * 2 * n_features_);
  }

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const PcaSettings           &get_settings() const { return settings_; }

  void update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_.get(), stream_));
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range r("PCA Sync Task");
    auto               &iview = ctx.inputs[0];
    auto               &oview = ctx.outputs[0];
    auto               *idata = reinterpret_cast<T *>(iview.data());
    auto               *odata = reinterpret_cast<T *>(oview.data());

    const auto   &idesc      = iview.desc;
    const int     rank       = static_cast<int>(idesc.shape.size());
    const int     height     = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 2)));
    const int     width      = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 1)));
    const int     n_samples  = height * width;
    const int     components = settings_.components();
    const int64_t il         = static_cast<int64_t>(settings_.begin) + 1;
    const int64_t iu         = static_cast<int64_t>(settings_.end);

    const T alpha = PcaTraits<T>::alpha();
    const T beta  = PcaTraits<T>::beta();

    // Strides (in elements) between consecutive batch slices
    const long long stride_I = static_cast<long long>(n_features_) * n_samples;
    const long long stride_C = static_cast<long long>(n_features_) * n_features_;
    const long long stride_O = static_cast<long long>(n_samples) * components;

    bool should_update = (iteration_count_ % settings_.update_rate == 0);

    if (should_update) {
      // 1. Compute covariance matrix: COV = I^H * I
      // Yields an (n_features x n_features) matrix representing covariance between features.
      // Looping standard GEMM allows cuBLAS to utilize Split-K for massive n_samples.
      for (int b = 0; b < n_batch_; ++b) {
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle_.get(), CUBLAS_OP_C, CUBLAS_OP_N, n_features_, n_features_, n_samples,
            &alpha, idata + b * stride_I, PcaTraits<T>::cuda_type, n_samples, idata + b * stride_I,
            PcaTraits<T>::cuda_type, n_samples, &beta, d_cov_.get() + b * stride_C,
            PcaTraits<T>::cuda_type, n_features_, PcaTraits<T>::compute_type, CUBLAS_GEMM_DEFAULT));
      }

      // 2. Per-batch eigendecomposition
      if (n_features_ <= cpu_heuristic_max_) {
        // Use CPU (LAPACKE) for small problems
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        nvtx3::scoped_range r_cpu("CPU eigen decomposition");
        auto                D2H       = cudaMemcpyDeviceToHost;
        auto                cov_bytes = n_batch_ * n_features_ * n_features_ * sizeof(T);
        CUDA_CHECK(cudaMemcpyAsync(h_cov_.get(), d_cov_.get(), cov_bytes, D2H, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        for (size_t b = 0; b < n_batch_; ++b) {
          auto cov_off    = b * n_features_ * n_features_;
          auto eigval_off = b * components;
          auto isuppz_off = b * 2 * n_features_;

          lapack_int m = 0;
          LAPACKE_CHECK(PcaTraits<T>::eigen(
              LAPACK_COL_MAJOR, 'V', 'I', 'L', n_features_, h_cov_.get() + cov_off, n_features_,
              0.0f, 0.0f, il, iu, 0.0f, &m, h_eigvals_.get() + eigval_off,
              h_eigvecs_.get() + cov_off, n_features_, h_isuppz_.get() + isuppz_off));

          HOLOVIBES_CHECK(m == components, "[Pca] unexpected eigenvector count from LAPACKE");
        }

        // Copy all eigenvector matrices back to GPU in one go.
        auto H2D       = cudaMemcpyHostToDevice;
        auto vec_bytes = n_batch_ * n_features_ * n_features_ * sizeof(T);
        CUDA_CHECK(cudaMemcpyAsync(d_cov_.get(), h_eigvecs_.get(), vec_bytes, H2D, stream_));
      }

      else {
        // Use GPU (cuSOLVER) for large problems
        nvtx3::scoped_range r_gpu("GPU eigen decomposition");
        float               vl = 0.0f, vu = 0.0f;
        for (size_t b = 0; b < n_batch_; ++b) {
          auto cov_off    = b * n_features_ * n_features_;
          auto eigval_off = b * n_features_;
          CUSOLVER_CHECK(cusolverDnXsyevdx(
              cusolver_handle_.get(), cusolver_params_.get(), CUSOLVER_EIG_MODE_VECTOR,
              CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features_, PcaTraits<T>::cuda_type,
              d_cov_.get() + cov_off, n_features_, &vl, &vu, il, iu, h_meig_.get() + b, CUDA_R_32F,
              d_eigvals_.get() + eigval_off, PcaTraits<T>::cuda_type, d_workspace_.get(),
              d_workspace_size_, h_workspace_.get(), h_workspace_size_, d_info_.get() + b));
        }
      }
    }

    // 3. Project data: Multiply input data by the eigenvector matrix
    T *eigvecs = reinterpret_cast<T *>(d_cov_.get());
    for (int b = 0; b < n_batch_; ++b) {
      CUBLAS_CHECK(cublasGemmEx(
          cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples, components, n_features_,
          &alpha, idata + b * stride_I, PcaTraits<T>::cuda_type, n_samples, eigvecs + b * stride_C,
          PcaTraits<T>::cuda_type, n_features_, &beta, odata + b * stride_O,
          PcaTraits<T>::cuda_type, n_samples, PcaTraits<T>::compute_type, CUBLAS_GEMM_DEFAULT));
    }

    iteration_count_++;
    return holoflow::core::OpResult::Ok;
  }

private:
  PcaSettings           settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  int                   n_features_;
  size_t                n_batch_;
  uint64_t              iteration_count_;

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
  check(idesc.rank() >= 3, "expected input rank >= 3");
  check(idesc.dtype == holoflow::core::DType::CF32 || idesc.dtype == holoflow::core::DType::F32,
        "expected input dtype CF32 or F32");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");

  const int feat_dim = static_cast<int>(idesc.rank()) - 3;
  check(settings.end <= static_cast<int>(idesc.shape.at(static_cast<size_t>(feat_dim))),
        "expected end <= n_features");
  check(settings.update_rate >= 1, "expected update_rate >= 1");

  // auto odesc                                 = idesc;
  // odesc.shape[static_cast<size_t>(feat_dim)] = static_cast<size_t>(settings.components());
  auto oshape                              = idesc.shape;
  oshape.at(static_cast<size_t>(feat_dim)) = static_cast<size_t>(settings.components());
  holoflow::core::TDesc odesc(oshape, idesc.dtype, idesc.mem_loc);

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
  const int   feat_dim = static_cast<int>(idesc.rank()) - 3;
  const int   n_feats  = static_cast<int>(idesc.shape.at(static_cast<size_t>(feat_dim)));

  // Flatten all leading dimensions into a single batch count
  int n_batch = 1;
  for (int i = 0; i < feat_dim; ++i) {
    n_batch *= static_cast<int>(idesc.shape[static_cast<size_t>(i)]);
  }

  // Dispatch based on datatype
  if (idesc.dtype == holoflow::core::DType::F32) {
    return std::make_unique<PcaTask<float>>(settings, idesc, ctx, n_feats, n_batch);
  } else {
    return std::make_unique<PcaTask<cuFloatComplex>>(settings, idesc, ctx, n_feats, n_batch);
  }
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {

  if (input_descs.size() == 1) {
    const auto  new_settings = jsettings.get<PcaSettings>();
    const auto &new_idesc    = input_descs[0];

    // Branch logic based on the expected templated type
    if (new_idesc.dtype == holoflow::core::DType::F32) {
      auto *old_pca = dynamic_cast<PcaTask<float> *>(old_task.get());
      if (old_pca != nullptr) {
        const auto &old_idesc = old_pca->get_idesc();
        if ((new_settings == old_pca->get_settings()) && (new_idesc.shape == old_idesc.shape) &&
            (new_idesc.strides == old_idesc.strides) && (new_idesc.dtype == old_idesc.dtype) &&
            (new_idesc.mem_loc == old_idesc.mem_loc)) {

          old_pca->update_stream(ctx.stream);
          return old_task;
        }
      }
    } else if (new_idesc.dtype == holoflow::core::DType::CF32) {
      auto *old_pca = dynamic_cast<PcaTask<cuFloatComplex> *>(old_task.get());
      if (old_pca != nullptr) {
        const auto &old_idesc = old_pca->get_idesc();
        if ((new_settings == old_pca->get_settings()) && (new_idesc.shape == old_idesc.shape) &&
            (new_idesc.strides == old_idesc.strides) && (new_idesc.dtype == old_idesc.dtype) &&
            (new_idesc.mem_loc == old_idesc.mem_loc)) {

          old_pca->update_stream(ctx.stream);
          return old_task;
        }
      }
    }
  }

  // Fallback: Recreate the task entirely.
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs