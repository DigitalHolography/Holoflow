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

#include <nvtx3/nvtx3.hpp>
#include <stdexcept>
#include <string>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"

#include "logger.hh"
#include "pca_cusolverdx.hh"

namespace holotask::syncs {

namespace {

// -------------------------------------------------------------------------------------------------
// PCA task implementation
// -------------------------------------------------------------------------------------------------

class PcaTask : public holoflow::core::ISyncTask {
public:
  // n_features: size of the feature dimension (shape[-3])
  // n_batch:    product of all leading dimensions (1 for rank-3 input)
  PcaTask(const PcaSettings &settings, const holoflow::core::TDesc &idesc,
          const holoflow::core::SyncCreateCtx &ctx, int n_features, size_t n_batch)
      : settings_(settings), idesc_(idesc), stream_(ctx.stream), n_features_(n_features),
        n_batch_(n_batch) {

    CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    heev_kernel_ = std::make_unique<detail::PcaHeevKernel>(n_features_, n_batch_, stream_);

    const auto cov_elems = n_batch_ * static_cast<size_t>(n_features_) * n_features_;
    d_cov_               = curaii::make_unique_device_ptr<float>(cov_elems);
    d_eigvals_           = curaii::make_unique_device_ptr<float>(n_batch_ * n_features_);
    d_info_              = curaii::make_unique_device_ptr<int>(n_batch_);
  }

  const holoflow::core::TDesc &get_idesc() const { return idesc_; }
  const PcaSettings           &get_settings() const { return settings_; }
  bool                         is_compatible_stream(cudaStream_t stream) const {
    return heev_kernel_->is_compatible_stream(stream);
  }

  void update_settings(const PcaSettings &settings, cudaStream_t stream) {
    settings_ = settings;
    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_handle_.get(), stream_));
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    nvtx3::scoped_range r("PCA Sync Task");
    auto               &iview = ctx.inputs[0];
    auto               &oview = ctx.outputs[0];
    auto               *idata = reinterpret_cast<float *>(iview.data());
    auto               *odata = reinterpret_cast<float *>(oview.data());

    const auto     &idesc      = iview.desc;
    const int       rank       = static_cast<int>(idesc.shape.size());
    const int       height     = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 2)));
    const int       width      = static_cast<int>(idesc.shape.at(static_cast<size_t>(rank - 1)));
    const int       n_samples  = height * width;
    const int       components = settings_.components();
    constexpr float alpha      = 1.0f;
    constexpr float beta       = 0.0f;

    // Strides (in elements) between consecutive batch slices
    const long long stride_I = static_cast<long long>(n_features_) * n_samples;
    const long long stride_C = static_cast<long long>(n_features_) * n_features_;
    const long long stride_O = static_cast<long long>(n_samples) * components;

    {
      nvtx3::scoped_range covariance_range("PCA covariance");
      // Independent GEMMs preserve cuBLAS Split-K selection for the large sample dimension.
      for (size_t b = 0; b < n_batch_; ++b) {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_T, CUBLAS_OP_N, n_features_,
                                  n_features_, n_samples, &alpha, idata + b * stride_I, CUDA_R_32F,
                                  n_samples, idata + b * stride_I, CUDA_R_32F, n_samples, &beta,
                                  d_cov_.get() + b * stride_C, CUDA_R_32F, n_features_,
                                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
      }
    }

    {
      nvtx3::scoped_range eigendecomposition_range("PCA cuSolverDx eigendecomposition");
      heev_kernel_->launch(d_cov_.get(), d_eigvals_.get(), d_info_.get(), n_batch_, stream_);
    }

    {
      nvtx3::scoped_range projection_range("PCA projection");
      const auto eigenvector_offset = static_cast<long long>(settings_.begin) * n_features_;
      for (size_t b = 0; b < n_batch_; ++b) {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples,
                                  components, n_features_, &alpha, idata + b * stride_I, CUDA_R_32F,
                                  n_samples, d_cov_.get() + b * stride_C + eigenvector_offset,
                                  CUDA_R_32F, n_features_, &beta, odata + b * stride_O, CUDA_R_32F,
                                  n_samples, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
      }
    }

    return holoflow::core::OpResult::Ok;
  }

private:
  PcaSettings           settings_;
  holoflow::core::TDesc idesc_;
  cudaStream_t          stream_;
  int                   n_features_;
  size_t                n_batch_;

  curaii::CublasHandle                   cublas_handle_;
  std::unique_ptr<detail::PcaHeevKernel> heev_kernel_;
  curaii::unique_device_ptr<float>       d_cov_;
  curaii::unique_device_ptr<float>       d_eigvals_;
  curaii::unique_device_ptr<int>         d_info_;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

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

// -------------------------------------------------------------------------------------------------
// Factory methods
// -------------------------------------------------------------------------------------------------

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
  check(idesc.dtype == holoflow::core::DType::F32,
        "cuSolverDx PCA currently supports F32 input only");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "expected input in device memory");
  check(settings.begin < settings.end, "expected begin < end");
  check(settings.begin >= 0, "expected begin >= 0");

  const int feat_dim = static_cast<int>(idesc.rank()) - 3;
  check(settings.end <= static_cast<int>(idesc.shape.at(static_cast<size_t>(feat_dim))),
        "expected end <= n_features");

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
  size_t n_batch = 1;
  for (int i = 0; i < feat_dim; ++i) {
    n_batch *= idesc.shape[static_cast<size_t>(i)];
  }

  return std::make_unique<PcaTask>(settings, idesc, ctx, n_feats, n_batch);
}

std::unique_ptr<holoflow::core::ISyncTask>
PcaFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {

  this->infer(input_descs, jsettings); // Validate settings before updating the existing task

  if (input_descs.size() == 1) {
    const auto  new_settings = jsettings.get<PcaSettings>();
    const auto &new_idesc    = input_descs[0];

    if (new_idesc.dtype == holoflow::core::DType::F32) {
      auto *old_pca = dynamic_cast<PcaTask *>(old_task.get());
      if (old_pca != nullptr) {
        const auto &old_idesc = old_pca->get_idesc();
        if ((new_idesc.shape == old_idesc.shape) && (new_idesc.strides == old_idesc.strides) &&
            (new_idesc.dtype == old_idesc.dtype) && (new_idesc.mem_loc == old_idesc.mem_loc) &&
            old_pca->is_compatible_stream(ctx.stream)) {

          old_pca->update_settings(new_settings, ctx.stream);
          return old_task;
        }
      }
    }
  }

  // Fallback: Recreate the task entirely.
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
