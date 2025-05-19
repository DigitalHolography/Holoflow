#include "holovibes/tasks/fresnel_diffraction_task.hh"

#include <cassert>
#include <cstdlib>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cufftXt.h>
#include <fstream>
#include <math_constants.h>
#include <numeric>
#include <nvrtc.h>
#include <spdlog/spdlog.h>

#include "curaii/v2/cuda.hh"
#include "curaii/v2/cufft.hh"
#include "holovibes/holovibes.hh"

#define NVRTC_SAFE_CALL(Name, x)                                               \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "\nerror: " << Name << " failed with error "                \
                << nvrtcGetErrorString(result) << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace dh {

// ==========================================================================
//                     FresnelDiffractionTask Implementation
// ==========================================================================

namespace {

__global__ void apply_lens_kernel(cuFloatComplex *idata, cuFloatComplex *odata,
                                  const cuFloatComplex *lens, int lens_size,
                                  int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= lens_size)
    return;

  cuFloatComplex val = lens[idx];
  for (int i = 0; i < batch_size; i++) {
    odata[i * lens_size + idx] = cuCmulf(idata[i * lens_size + idx], val);
  }
}

} // namespace

FresnelDiffractionTask::FresnelDiffractionTask(
    const TaskMeta &meta, cudaStream_t stream, float lambda, float z,
    float pixel_size, bool skip_phase_shift,
    curaii::cuda::unique_device_ptr<cuFloatComplex> lens,
    curaii::cufft::Handle handle)
    : Task(meta, stream), lambda_(lambda), z_(z), pixel_size_(pixel_size),
      skip_phase_shift_(skip_phase_shift), lens_(std::move(lens)),
      handle_(std::move(handle)) {}

void FresnelDiffractionTask::run(TensorView input, TensorView output) {
  auto idata = (cuFloatComplex *)input.data();
  auto odata = (cuFloatComplex *)output.data();
  // int batch_size = input.meta().shape().at(0);
  // int lens_size = input.meta().size() / batch_size;

  // dim3 block_size = 256;
  // dim3 grid_size = (lens_size + block_size.x - 1) / block_size.x;

  // apply_lens_kernel<<<grid_size, block_size, 0, stream_>>>(
  //     idata, odata, lens_.get(), lens_size, batch_size);

  CUFFT_CHECK(cufftXtExec(handle_.get(), odata, odata, CUFFT_FORWARD));
  // CUDA_CHECK(cudaPeekAtLastError());

  // TODO implement phase shift
}

// ==========================================================================
//                     FresnelDiffractionTaskFactory Implementation
// ==========================================================================

namespace {

void compile_file_to_lto(std::vector<char> &cubin_result,
                         const char *filename) {
  ///////////////
  // OPEN FILE //
  ///////////////
  std::ifstream inputFile(filename,
                          std::ios::in | std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  std::vector<char> memBlock(inputSize + 1);

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(memBlock.data(), inputSize);
  inputFile.close();
  memBlock[inputSize] = '\x0';

  const int num_params = 6;
  const char *compile_params[] = {
      "-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\include",
      "-arch=compute_86",
      "--std=c++20",
      "--relocatable-device-code=true",
      "-default-device",
      "-dlto"};

  /////////////
  // COMPILE //
  /////////////
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
      "nvrtcCreateProgram",
      nvrtcCreateProgram(&prog, memBlock.data(), filename, 0, NULL, NULL));
  nvrtcResult res = nvrtcCompileProgram(prog, num_params, compile_params);

  ///////////////
  // PRINT LOG //
  ///////////////
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize",
                  nvrtcGetProgramLogSize(prog, &logSize));
  std::vector<char> log(logSize + 1);
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log.data()));
  log[logSize] = '\x0';

  if (log.size() > 2) {
    std::cerr << "\n compilation log ---\n";
    std::string s(log.begin(), log.end());
    std::cerr << s;
    std::cerr << "\n end log ---\n";
  }

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

  size_t codeSize;
  NVRTC_SAFE_CALL("nvrtcGetLTOIRSize", nvrtcGetLTOIRSize(prog, &codeSize));
  std::vector<char> buffer(codeSize);
  NVRTC_SAFE_CALL("nvrtcGetNVVM", nvrtcGetLTOIR(prog, buffer.data()));
  cubin_result = buffer;
}

__global__ void quadratic_lens_kernel(cuFloatComplex *lens, int width,
                                      int height, float lambda, float z,
                                      float pixel_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height)
    return;

  int size = width > height ? width : height;

  // The intent with offsets is to support non-square images.
  // The are used to "center" the indexes as if the rectangle was extended to
  // a square.
  int offset_x = (size - width) / 2;
  int offset_y = (size - height) / 2;

  float x = ((col + offset_x) - size / 2.0f) * pixel_size;
  float y = ((row + offset_y) - size / 2.0f) * pixel_size;

  float phase = CUDART_PI_F / (lambda * z) * (x * x + y * y);
  float cos_phase = cosf(phase);
  float sin_phase = sinf(phase);

  lens[row * width + col] = make_cuComplex(cos_phase, sin_phase);
}

} // namespace

TaskMeta FresnelDiffractionTaskFactory::type_check(const TensorMeta &imeta,
                                                   const json &jparams) {
  // 0) Unpack parameters
  const Params params = jparams.get<Params>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Parameter sanity
  check(params.lambda > 0, "lambda <= 0");
  check(params.pixel_size > 0, "pixel_size <= 0");
  check(params.z > 0, "z <= 0");
  check(params.skip_phase_shift, "skip_phase_shift not yet implemented");

  // 2) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.data_type() == DataType::CF32, "tensor data_type != CF32");
  check(imeta.memory_location() == MemoryLocation::DEVICE,
        "tensor not in DEVICE memory");

  return TaskMeta(imeta, imeta, true);
}

std::unique_ptr<Task> FresnelDiffractionTaskFactory::create(
    const TensorMeta &imeta, const json &jparams, cudaStream_t stream) {

  std::vector<char> callback_buffer;
  compile_file_to_lto(
      callback_buffer,
      "C:\\Users\\yakutsk\\Documents\\Holoflow\\src\\holovibes\\tasks\\apply_"
      "lens_callback.cu");

  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<Params>();

  const int B = static_cast<int>(imeta.shape()[0]);
  const int H = static_cast<int>(imeta.shape()[1]);
  const int W = static_cast<int>(imeta.shape()[2]);

  // 2) Buffer sizes
  const int frame_size = imeta.size_in_bytes() / B;

  // 3) Allocations
  auto d_lens =
      curaii::cuda::make_unique_device_ptr<cuFloatComplex>(frame_size, stream);

  // 4) Compute lens
  dim3 block_size(16, 16);
  dim3 grid_size((W + block_size.x - 1) / block_size.x,
                 (H + block_size.y - 1) / block_size.y);

  quadratic_lens_kernel<<<grid_size, block_size, 0, stream>>>(
      d_lens.get(), W, H, params.lambda, params.z, params.pixel_size);

  CUDA_CHECK(cudaPeekAtLastError());

  auto *lens_ptr = d_lens.get();

  // 5) Initialize cufft plan
  int rank = 2;
  long long int n[2] = {H, W};
  long long int inembed[2] = {H, W};
  int istride = 1;
  int idist = H * W;
  cudaDataType inputtype = CUDA_C_32F;
  long long int onembed[2] = {H, W};
  int ostride = 1;
  int odist = H * W;
  cudaDataType outputtype = CUDA_C_32F;
  int batch = B;
  size_t work_size = 0;
  cudaDataType executiontype = CUDA_C_32F;

  curaii::cufft::Handle handle;
  CUFFT_CHECK(cufftXtSetJITCallback(
      handle.get(), "apply_lens_callback", (void *)callback_buffer.data(),
      callback_buffer.size(), CUFFT_CB_LD_COMPLEX, (void **)&lens_ptr));
  CUFFT_CHECK(cufftSetStream(handle.get(), stream));
  CUFFT_CHECK(cufftXtGetSizeMany(handle.get(), rank, n, inembed, istride, idist,
                                 inputtype, onembed, ostride, odist, outputtype,
                                 batch, &work_size, executiontype));
  CUFFT_CHECK(cufftXtMakePlanMany(
      handle.get(), rank, n, inembed, istride, idist, inputtype, onembed,
      ostride, odist, outputtype, batch, &work_size, executiontype));

  // 6) Assemble task
  auto *task = new FresnelDiffractionTask(meta, stream, params.lambda, params.z,
                                          params.pixel_size, true,
                                          std::move(d_lens), std::move(handle));
  return std::unique_ptr<FresnelDiffractionTask>(task);
}

} // namespace dh