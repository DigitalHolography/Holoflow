#include "holovibes/tasks/pca_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "bug_buster/bug_buster.hh"
#include "curaii/cublas.hh"
#include "curaii/curaii.hh"
#include "curaii/cusolver_dn.hh"
#include "curaii/v2/cuda.hh"
#include "holoflow/error.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     PCATask Implementation
// ==========================================================================

PCATask::PCATask(const TaskMeta &meta, CudaStreamRef stream,
                 CublasHandle cublas_handle, CusolverDnHandle cusolver_handle,
                 CusolverDnParams cusolver_params,
                 unique_device_ptr<cuFloatComplex> d_cov_matrix,
                 unique_device_ptr<float> d_eigenvalues,
                 unique_device_ptr<int> d_info,
                 unique_host_ptr<uint8_t> h_workspace,
                 unique_device_ptr<uint8_t> d_workspace,
                 size_t h_workspace_size, size_t d_workspace_size, size_t begin,
                 size_t end)
    : Task(meta, stream), cublas_handle_(std::move(cublas_handle)),
      cusolver_handle_(std::move(cusolver_handle)),
      cusolver_params_(std::move(cusolver_params)),
      d_cov_matrix_(std::move(d_cov_matrix)),
      d_eigenvalues_(std::move(d_eigenvalues)), d_info_(std::move(d_info)),
      h_workspace_(std::move(h_workspace)),
      d_workspace_(std::move(d_workspace)), h_workspace_size_(h_workspace_size),
      d_workspace_size_(d_workspace_size), begin_(begin), end_(end) {}

void PCATask::run(TensorView input, TensorView output) {
  // Note:
  // This PCA implementation does not normalize input as this does
  // not seems to make a difference in our application.

  // 1) Aliases
  size_t batch = input.meta().shape().at(0);
  size_t height = input.meta().shape().at(1);
  size_t width = input.meta().shape().at(2);
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();
  int n_features = static_cast<int>(batch);
  int n_samples = static_cast<int>(height * width);
  cuComplex alpha = make_cuComplex(1.0f, 0.0f);
  cuComplex beta = make_cuComplex(0.0f, 0.0f);

  // 2) Compute covariance matrix:
  // COV = I^H * I, where I is the input data matrix with dimensions
  // (n_samples x n_features). This operation yields an (n_features x
  // n_features) Hermitian matrix representing the covariance between each
  // pair of features (assuming the data has been centered).
  if (auto result = cublas_handle_.try_gemm_ex(
          CublasOperation(CUBLAS_OP_C), CublasOperation(CUBLAS_OP_N),
          n_features, n_features, n_samples, &alpha, idata,
          CudaDataType(CUDA_C_32F), n_samples, idata, CudaDataType(CUDA_C_32F),
          n_samples, &beta, d_cov_matrix_.get(), CudaDataType(CUDA_C_32F),
          n_features, CublasComputeType(CUBLAS_COMPUTE_32F_FAST_16F),
          CUBLAS_GEMM_DEFAULT);
      !result) {
    throw std::runtime_error(
        fmt::format("[PCATask::run] failed with error \"{}\"", result.error()));
  }

  // 3) Compute eigenvectors:
  // Perform an eigen decomposition on the covariance matrix to extract both
  // eigenvalues and eigenvectors. The eigenvectors (principal components) are
  // stored in the same memory as the covariance matrix. These eigenvectors
  // represent the directions of maximum variance in the data.
  cuFloatComplex *eigenvecs = d_cov_matrix_.get();

  int64_t h_meig = 0;
  float vl = 0;
  float vu = 0;
  if (auto result = cusolver_handle_.try_x_syevdx(
          cusolver_params_.ref(), CusolverEigMode(CUSOLVER_EIG_MODE_VECTOR),
          CusolverEigRange(CUSOLVER_EIG_RANGE_I),
          CublasFillMode(CUBLAS_FILL_MODE_LOWER), n_features,
          CudaDataType(CUDA_C_32F), d_cov_matrix_.get(), n_features, &vl, &vu,
          begin_ + 1, end_, &h_meig, CudaDataType(CUDA_R_32F),
          d_eigenvalues_.get(), CudaDataType(CUDA_C_32F), d_workspace_.get(),
          d_workspace_size_, h_workspace_.get(), h_workspace_size_,
          d_info_.get());
      !result) {
    throw std::runtime_error(
        fmt::format("[PCATask::run] failed with error \"{}\"", result.error()));
  }
  DH_CHECK(h_meig == (int)end_ - (int)begin_);

  // 4) Project data:
  // Multiply the input data by the eigenvector matrix
  // to transform the data into the principal component space. Each row of the
  // output corresponds to the projection of the original data onto the new
  // basis defined by the eigenvectors.
  if (auto result = cublas_handle_.try_gemm_ex(
          CublasOperation(CUBLAS_OP_N), CublasOperation(CUBLAS_OP_N), n_samples,
          (int)(end_ - begin_), n_features, &alpha, idata,
          CudaDataType(CUDA_C_32F), n_samples, eigenvecs,
          CudaDataType(CUDA_C_32F), n_features, &beta, odata,
          CudaDataType(CUDA_C_32F), n_samples,
          CublasComputeType(CUBLAS_COMPUTE_32F_FAST_16F), CUBLAS_GEMM_DEFAULT);
      !result) {
    throw std::runtime_error(
        fmt::format("[PCATask::run] failed with error \"{}\"", result.error()));
  }
}

// ==========================================================================
//                     PCATaskFactory Implementation
// ==========================================================================

TaskMeta PCATaskFactory::type_check(const TensorMeta &imeta,
                                    const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.data_type() == DataType::CF32, "tensor data_type != CF32");
  check(params.end <= imeta.shape().at(0), "end > tensor dim 0");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  // 2) Success
  auto oshape = imeta.shape();
  oshape.at(0) = (params.end - params.begin);
  TensorMeta ometa(imeta.data_type(), imeta.memory_location(), oshape);
  return TaskMeta(imeta, ometa, false);
}

std::unique_ptr<Task> PCATaskFactory::create(const TensorMeta &imeta,
                                             const json &jparams,
                                             CudaStreamRef stream) {

  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  size_t batch = imeta.shape().at(0);
  int n_features = static_cast<int>(batch);

  // CublasHandle
  auto cublas_handle_result = CublasHandle::try_create();
  if (!cublas_handle_result) {
    throw std::runtime_error(
        fmt::format("[PCATaskFactory::create] failed with error \"{}\"",
                    cublas_handle_result.error()));
  }
  auto cublas_handle = std::move(cublas_handle_result.value());

  if (auto result = cublas_handle.try_set_stream(stream); !result) {
    throw std::runtime_error(fmt::format(
        "[PCATaskFactory::create] failed with error \"{}\"", result.error()));
  }

  // CusolverDnHandle
  auto cusolver_handle_result = CusolverDnHandle::try_create();
  if (!cusolver_handle_result) {
    throw std::runtime_error(
        fmt::format("[PCATaskFactory::create] failed with error \"{}\"",
                    cusolver_handle_result.error()));
  }
  auto cusolver_handle = std::move(cusolver_handle_result.value());

  if (auto result = cusolver_handle.try_set_stream(stream); !result) {
    throw std::runtime_error(fmt::format(
        "[PCATaskFactory::create] failed with error \"{}\"", result.error()));
  }

  if (auto result = cusolver_handle.try_set_deterministic_mode(
          CusolverDeterministicMode(CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS));
      !result) {
    throw std::runtime_error(fmt::format(
        "[PCATaskFactory::create] failed with error \"{}\"", result.error()));
  }

  // CusolverDnParams
  auto cusolver_params_result = CusolverDnParams::try_create();
  if (!cusolver_params_result) {
    throw std::runtime_error(
        fmt::format("[PCATaskFactory::create] failed with error \"{}\"",
                    cusolver_params_result.error()));
  }
  auto cusolver_params = std::move(cusolver_params_result.value());

  // Covariance matrix
  auto d_cov_matrix = make_unique_device_ptr<cuFloatComplex>(
      n_features * n_features, stream.stream());

  // Eigenvalues
  auto d_eigenvalues = make_unique_device_ptr<float>(
      (params.end - params.begin), stream.stream());

  // Info
  auto d_info = make_unique_device_ptr<int>(1, stream.stream());

  // Workspace sizes
  size_t d_workspace_size = 0;
  size_t h_workspace_size = 0;
  int64_t h_meig = 0;
  if (auto result = cusolver_handle.try_x_syevdx_buffer_size(
          cusolver_params.ref(), CusolverEigMode(CUSOLVER_EIG_MODE_VECTOR),
          CusolverEigRange(CUSOLVER_EIG_RANGE_I),
          CublasFillMode(CUBLAS_FILL_MODE_LOWER), n_features,
          CudaDataType(CUDA_C_32F), d_cov_matrix.get(), n_features, nullptr,
          nullptr, params.begin + 1, params.end, &h_meig,
          CudaDataType(CUDA_R_32F), d_eigenvalues.get(),
          CudaDataType(CUDA_C_32F), &d_workspace_size, &h_workspace_size);
      !result) {
    throw std::runtime_error(fmt::format(
        "[PCATaskFactory::create] failed with error \"{}\"", result.error()));
  }

  // Device workspace
  auto d_workspace =
      make_unique_device_ptr<uint8_t>(d_workspace_size, stream.stream());

  // Host workspace
  auto h_workspace = make_unique_host_ptr<uint8_t>(h_workspace_size);

  // Return
  auto *task = new PCATask(
      meta, stream, std::move(cublas_handle), std::move(cusolver_handle),
      std::move(cusolver_params), std::move(d_cov_matrix),
      std::move(d_eigenvalues), std::move(d_info), std::move(h_workspace),
      std::move(d_workspace), h_workspace_size, d_workspace_size, params.begin,
      params.end);

  return std::unique_ptr<PCATask>(task);
}

} // namespace dh