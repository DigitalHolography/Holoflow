#include "holovibes/tasks/pca_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <tl/expected.hpp>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holoflow/error.hh"
#include "holovibes/holovibes.hh"

namespace {

__global__ void abs_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fabsf(input[idx]);
  }
}

} // namespace

namespace dh {

// ==========================================================================
//                     PCATask Implementation
// ==========================================================================

PCATask::PCATask(const TaskMeta &meta, cudaStream_t stream,
                 curaii::cublas::Handle cublas_handle,
                 curaii::cusolverdn::Handle cusolver_handle,
                 curaii::cusolverdn::Params cusolver_params,
                 curaii::cuda::unique_device_ptr<uint8_t> d_cov_matrix,
                 curaii::cuda::unique_device_ptr<float> d_eigenvalues,
                 curaii::cuda::unique_device_ptr<int> d_info,
                 curaii::cuda::unique_host_ptr<uint8_t> h_workspace,
                 curaii::cuda::unique_device_ptr<uint8_t> d_workspace,
                 size_t h_workspace_size, size_t d_workspace_size, size_t begin,
                 size_t end, bool is_complex)
    : Task(meta, stream), cublas_handle_(std::move(cublas_handle)),
      cusolver_handle_(std::move(cusolver_handle)),
      cusolver_params_(std::move(cusolver_params)),
      d_cov_matrix_(std::move(d_cov_matrix)),
      d_eigenvalues_(std::move(d_eigenvalues)), d_info_(std::move(d_info)),
      h_workspace_(std::move(h_workspace)),
      d_workspace_(std::move(d_workspace)), h_workspace_size_(h_workspace_size),
      d_workspace_size_(d_workspace_size), begin_(begin), end_(end),
      is_complex_(is_complex) {}

void PCATask::run(TensorView input, TensorView output) {
  // Note:
  // This PCA implementation does not normalize input as this does
  // not seems to make a difference in our application.

  // 1) Aliases
  size_t batch   = input.meta().shape().at(0);
  size_t height  = input.meta().shape().at(1);
  size_t width   = input.meta().shape().at(2);
  int n_features = static_cast<int>(batch);
  int n_samples  = static_cast<int>(height * width);

  if (is_complex_) {
    auto idata      = reinterpret_cast<cuFloatComplex *>(input.data());
    auto odata      = reinterpret_cast<cuFloatComplex *>(output.data());
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta  = make_cuComplex(0.0f, 0.0f);

    // 2) Compute covariance matrix:
    // COV = I^H * I, where I is the input data matrix with dimensions
    // (n_samples x n_features). This operation yields an (n_features x
    // n_features) Hermitian matrix representing the covariance between each
    // pair of features (assuming the data has been centered).
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_.get(), CUBLAS_OP_C, CUBLAS_OP_N, n_features, n_features,
        n_samples, &alpha, idata, CUDA_C_32F, n_samples, idata, CUDA_C_32F,
        n_samples, &beta, d_cov_matrix_.get(), CUDA_C_32F, n_features,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));

    // 3) Compute eigenvectors:
    // Perform an eigen decomposition on the covariance matrix to extract both
    // eigenvalues and eigenvectors. The eigenvectors (principal components) are
    // stored in the same memory as the covariance matrix. These eigenvectors
    // represent the directions of maximum variance in the data.
    auto eigenvecs = reinterpret_cast<cuFloatComplex *>(d_cov_matrix_.get());

    int64_t h_meig = 0;
    float vl       = 0;
    float vu       = 0;
    CUSOLVER_CHECK(cusolverDnXsyevdx(
        cusolver_handle_.get(), cusolver_params_.get(),
        CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER,
        n_features, CUDA_C_32F, d_cov_matrix_.get(), n_features, &vl, &vu,
        begin_ + 1, end_, &h_meig, CUDA_R_32F, d_eigenvalues_.get(), CUDA_C_32F,
        d_workspace_.get(), d_workspace_size_, h_workspace_.get(),
        h_workspace_size_, d_info_.get()));

    // 4) Project data:
    // Multiply the input data by the eigenvector matrix
    // to transform the data into the principal component space. Each row of the
    // output corresponds to the projection of the original data onto the new
    // basis defined by the eigenvectors.
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples,
        (int)(end_ - begin_), n_features, &alpha, idata, CUDA_C_32F, n_samples,
        eigenvecs, CUDA_C_32F, n_features, &beta, odata, CUDA_C_32F, n_samples,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
  }

  else {
    auto idata  = reinterpret_cast<float *>(input.data());
    auto odata  = reinterpret_cast<float *>(output.data());
    float alpha = 1.0f;
    float beta  = 0.0f;

    // 2) Compute covariance matrix:
    // COV = I^T * I, where I is the input data matrix with dimensions
    // (n_samples x n_features). This operation yields an (n_features x
    // n_features) Hermitian matrix representing the covariance between each
    // pair of features (assuming the data has been centered).
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_.get(), CUBLAS_OP_T, CUBLAS_OP_N, n_features, n_features,
        n_samples, &alpha, idata, CUDA_R_32F, n_samples, idata, CUDA_R_32F,
        n_samples, &beta, d_cov_matrix_.get(), CUDA_R_32F, n_features,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // 3) Compute eigenvectors:
    // Perform an eigen decomposition on the covariance matrix to extract both
    // eigenvalues and eigenvectors. The eigenvectors (principal components) are
    // stored in the same memory as the covariance matrix. These eigenvectors
    // represent the directions of maximum variance in the data.
    auto eigenvecs = reinterpret_cast<float *>(d_cov_matrix_.get());

    int64_t h_meig = 0;
    float vl       = 0;
    float vu       = 0;
    CUSOLVER_CHECK(cusolverDnXsyevdx(
        cusolver_handle_.get(), cusolver_params_.get(),
        CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER,
        n_features, CUDA_R_32F, d_cov_matrix_.get(), n_features, &vl, &vu,
        begin_ + 1, end_, &h_meig, CUDA_R_32F, d_eigenvalues_.get(), CUDA_R_32F,
        d_workspace_.get(), d_workspace_size_, h_workspace_.get(),
        h_workspace_size_, d_info_.get()));

    // 4) Project data:
    // Multiply the input data by the eigenvector matrix
    // to transform the data into the principal component space. Each row of the
    // output corresponds to the projection of the original data onto the new
    // basis defined by the eigenvectors.
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_.get(), CUBLAS_OP_N, CUBLAS_OP_N, n_samples,
        (int)(end_ - begin_), n_features, &alpha, idata, CUDA_R_32F, n_samples,
        eigenvecs, CUDA_R_32F, n_features, &beta, odata, CUDA_R_32F, n_samples,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    int size    = output.size();
    int threads = 256;
    int blocks  = (size + threads - 1) / threads;
    abs_kernel<<<blocks, threads, 0, stream_>>>(odata, odata, size);
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
  std::set<DataType> valid_dtypes = {DataType::CF32, DataType::F32};
  check(valid_dtypes.contains(imeta.data_type()), "tensor data_type invalid");
  check(params.end <= imeta.shape().at(0), "end > tensor dim 0");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  // 2) Success
  auto oshape  = imeta.shape();
  oshape.at(0) = (params.end - params.begin);
  TensorMeta ometa(imeta.data_type(), imeta.memory_location(), oshape);
  return TaskMeta(imeta, ometa, false);
}

std::unique_ptr<Task> PCATaskFactory::create(const TensorMeta &imeta,
                                             const json &jparams,
                                             cudaStream_t stream) {

  // 1) Validate
  auto meta   = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  size_t batch    = imeta.shape().at(0);
  int n_features  = static_cast<int>(batch);
  bool is_complex = imeta.data_type() == DataType::CF32;

  // 2) Handles and params
  curaii::cublas::Handle cublas_handle;
  CUBLAS_CHECK(cublasSetStream(cublas_handle.get(), stream));

  curaii::cusolverdn::Handle cusolver_handle;
  CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle.get(), stream));
  CUSOLVER_CHECK(cusolverDnSetDeterministicMode(
      cusolver_handle.get(), CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS));

  curaii::cusolverdn::Params cusolver_params;

  // 3) Buffer allocations
  auto d_cov_matrix = curaii::cuda::make_unique_device_ptr<uint8_t>(
      n_features * n_features * size_of(imeta.data_type()), stream);

  auto d_eigenvalues = curaii::cuda::make_unique_device_ptr<float>(
      (params.end - params.begin), stream);

  auto d_info = curaii::cuda::make_unique_device_ptr<int>(1, stream);

  // 4) Workspace sizes
  size_t d_workspace_size = 0;
  size_t h_workspace_size = 0;
  int64_t h_meig          = 0;

  if (is_complex) {
    CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
        cusolver_handle.get(), cusolver_params.get(), CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features, CUDA_C_32F,
        d_cov_matrix.get(), n_features, nullptr, nullptr, params.begin + 1,
        params.end, &h_meig, CUDA_R_32F, d_eigenvalues.get(), CUDA_C_32F,
        &d_workspace_size, &h_workspace_size));
  }

  else {
    CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
        cusolver_handle.get(), cusolver_params.get(), CUSOLVER_EIG_MODE_VECTOR,
        CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n_features, CUDA_R_32F,
        d_cov_matrix.get(), n_features, nullptr, nullptr, params.begin + 1,
        params.end, &h_meig, CUDA_R_32F, d_eigenvalues.get(), CUDA_R_32F,
        &d_workspace_size, &h_workspace_size));
  }

  // 5) Workspace allocations
  auto d_workspace =
      curaii::cuda::make_unique_device_ptr<uint8_t>(d_workspace_size, stream);

  auto h_workspace =
      curaii::cuda::make_unique_host_ptr<uint8_t>(h_workspace_size);

  // 6) Assemble task
  auto *task = new PCATask(
      meta, stream, std::move(cublas_handle), std::move(cusolver_handle),
      std::move(cusolver_params), std::move(d_cov_matrix),
      std::move(d_eigenvalues), std::move(d_info), std::move(h_workspace),
      std::move(d_workspace), h_workspace_size, d_workspace_size, params.begin,
      params.end, is_complex);

  return std::unique_ptr<PCATask>(task);
}

} // namespace dh