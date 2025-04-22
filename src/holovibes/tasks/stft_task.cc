#include "holovibes/tasks/stft_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/curaii.hh"
#include "curaii/v2/cuda.hh"
#include "curaii/v2/cufft.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     STFTTask Implementation
// ==========================================================================

STFTTask::STFTTask(const TaskMeta &meta, CudaStreamRef stream,
                   curaii::cufft::Handle handle)
    : Task(meta, stream), handle_(std::move(handle)) {}

void STFTTask::run(TensorView input, TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();

  CUFFT_CHECK(cufftXtExec(handle_.get(), idata, odata, CUFFT_FORWARD));
}

// ==========================================================================
//                     STFTTaskFactory Implementation
// ==========================================================================

TaskMeta STFTTaskFactory::type_check(const TensorMeta &imeta, const json &) {
  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.data_type() == DataType::CF32, "tensor data_type != CF32");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  return TaskMeta(imeta, imeta, true);
}

std::unique_ptr<Task> STFTTaskFactory::create(const TensorMeta &imeta,
                                              const json &jparams,
                                              CudaStreamRef stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);

  const int B = static_cast<int>(imeta.shape()[0]);
  const int H = static_cast<int>(imeta.shape()[1]);
  const int W = static_cast<int>(imeta.shape()[2]);

  // 2) Initialize cufft plan
  int rank = 1;
  long long int n[1] = {B};
  long long int inembed[1] = {B};
  int istride = H * W;
  int idist = 1;
  cudaDataType inputtype = CUDA_C_32F;
  long long int onembed[1] = {B};
  int ostride = H * W;
  int odist = 1;
  cudaDataType outputtype = CUDA_C_32F;
  int batch = H * W;
  size_t work_size = 0;
  cudaDataType executiontype = CUDA_C_32F;

  curaii::cufft::Handle handle;
  CUFFT_CHECK(cufftSetStream(handle.get(), stream.stream()));
  CUFFT_CHECK(cufftXtGetSizeMany(handle.get(), rank, n, inembed, istride, idist,
                                 inputtype, onembed, ostride, odist, outputtype,
                                 batch, &work_size, executiontype));
  CUFFT_CHECK(cufftXtMakePlanMany(
      handle.get(), rank, n, inembed, istride, idist, inputtype, onembed,
      ostride, odist, outputtype, batch, &work_size, executiontype));

  // 3) Assemble task
  auto *task = new STFTTask(meta, stream, std::move(handle));
  return std::unique_ptr<STFTTask>(task);
}

} // namespace dh