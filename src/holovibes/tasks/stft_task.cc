#include "holovibes/tasks/stft_task.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/cufft.hh"
#include "curaii/curaii.hh"
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

tl::expected<void, Error> STFTTask::run(TensorView input, TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();

  if (auto result =
          handle_.try_xt_exec(idata, odata, CufftDirection(CUFFT_FORWARD));
      !result) {
    dh::holovibes_logger()->warn("[STFTTask::run] Fourier "
                                 "transform failed with error: \"{}\"",
                                 result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return {};
}

// ==========================================================================
//                     STFTTaskFactory Implementation
// ==========================================================================

tl::expected<TaskMeta, Error>
STFTTaskFactory::type_check(const TensorMeta &imeta, const json &) {
  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn("[STFTTaskFactory::type_check] invalid rank: {}",
                             (int)imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.memory_location() != MemoryLocation::DEVICE) {
    holovibes_logger()->warn(
        "[STFTTaskFactory::type_check] invalid memory location: {}",
        (int)imeta.memory_location());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::CF32) {
    holovibes_logger()->warn(
        "[STFTTaskFactory::type_check] invalid data type: {}",
        (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return TaskMeta(imeta, imeta, true);
}

tl::expected<std::unique_ptr<Task>, Error>
STFTTaskFactory::create(const TensorMeta &imeta, const json &jparams,
                        CudaStreamRef stream) {

  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("[PCATaskFactory::create] type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();

  int batch = static_cast<int>(imeta.shape().at(0));
  int height = static_cast<int>(imeta.shape().at(1));
  int width = static_cast<int>(imeta.shape().at(2));

  // Initialize cufft plan
  int rank = 1;
  long long int n[1] = {batch};
  long long int inembed[1] = {batch};
  int istride = height * width;
  int idist = 1;
  CudaDataType inputtype(CUDA_C_32F);
  long long int onembed[1] = {batch};
  int ostride = height * width;
  int odist = 1;
  CudaDataType outputtype(CUDA_C_32F);
  CudaDataType executiontype(CUDA_C_32F);

  auto plan_result = CufftHandle::try_create();
  if (!plan_result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             plan_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }
  auto handle = std::move(plan_result.value());

  size_t work_size = 0;
  if (auto result = handle.try_xt_get_size_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, height * width, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (auto result = handle.try_xt_make_plan_many(
          rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
          outputtype, height * width, &work_size, executiontype);
      !result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto stream_result = handle.try_set_stream(stream);
  if (!stream_result) {
    holovibes_logger()->warn("[FresnelDiffractionTaskFactory::create] Fourrier "
                             "transform creation failed with error: \"{}\"",
                             stream_result.error());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  auto *task = new STFTTask(meta, stream, std::move(handle));
  return std::unique_ptr<STFTTask>(task);
}

} // namespace dh