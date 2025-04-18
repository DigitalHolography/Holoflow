#include "holovibes/tasks/stft_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/cufft.hh"
#include "curaii/curaii.hh"
#include "curaii/v2/cuda.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     STFTTask Implementation
// ==========================================================================

STFTTask::STFTTask(const TaskMeta &meta, CudaStreamRef stream,
                   CufftHandle handle)
    : Task(meta, stream), handle_(std::move(handle)) {}

void STFTTask::run(TensorView input, TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();

  if (auto result =
          handle_.try_xt_exec(idata, odata, CufftDirection(CUFFT_FORWARD));
      !result) {
    throw std::runtime_error(fmt::format(
        "[STFTTask::run] Fourier transform failed with error: \"{}\"",
        result.error()));
  }
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
  CudaDataType inputtype(CUDA_C_32F);
  long long int onembed[1] = {B};
  int ostride = H * W;
  int odist = 1;
  CudaDataType outputtype(CUDA_C_32F);
  int batch = H * W;
  CudaDataType executiontype(CUDA_C_32F);

  auto plan_result = CufftHandle::try_create();
  if (!plan_result) {
    throw std::runtime_error(
        fmt::format("[FresnelDiffractionTaskFactory::create] Fourrier "
                    "transform creation failed with error: \"{}\"",
                    plan_result.error()));
  }
  auto handle = std::move(plan_result.value());

  size_t work_size = 0;
  if (auto result = handle.try_xt_get_size_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    throw std::runtime_error(
        fmt::format("[FresnelDiffractionTaskFactory::create] Fourrier "
                    "transform creation failed with error: \"{}\"",
                    result.error()));
  }

  if (auto result = handle.try_xt_make_plan_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, batch, &work_size, executiontype);
      !result) {
    throw std::runtime_error(
        fmt::format("[fresneldiffractiontaskfactory::create] fourrier "
                    "transform creation failed with error: \"{}\"",
                    result.error()));
  }

  auto stream_result = handle.try_set_stream(stream);
  if (!stream_result) {
    throw std::runtime_error(
        fmt::format("[fresneldiffractiontaskfactory::create] fourrier "
                    "transform creation failed with error: \"{}\"",
                    stream_result.error()));
  }

  auto *task = new STFTTask(meta, stream, std::move(handle));
  return std::unique_ptr<STFTTask>(task);
}

} // namespace dh