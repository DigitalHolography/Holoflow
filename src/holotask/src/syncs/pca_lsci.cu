// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include "holotask/syncs/pca_lsci.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "holotask/syncs/spatial_contrast.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const PcaLsciSettings &s) {
  const char *map = s.map == PcaLsciMap::Original ? "original" :
                    s.map == PcaLsciMap::Static ? "static" :
                    s.map == PcaLsciMap::PcaFrame ? "pca_frame" : "dynamic";
  j = {{"window_size", s.window_size}, {"map", map}};
}

void from_json(const nlohmann::json &j, PcaLsciSettings &s) {
  s.window_size = j.value("window_size", 5);
  const auto map = j.value("map", std::string{"dynamic"});
  if (map == "original") s.map = PcaLsciMap::Original;
  else if (map == "static") s.map = PcaLsciMap::Static;
  else if (map == "dynamic") s.map = PcaLsciMap::Dynamic;
  else if (map == "pca_frame") s.map = PcaLsciMap::PcaFrame;
  else throw std::invalid_argument("PcaLsci map must be original, static, dynamic or pca_frame");
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

CUcontext context_for_stream(cudaStream_t stream) {
  CUcontext context = nullptr;
  const CUresult result = stream != nullptr
      ? cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context)
      : cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context == nullptr)
    throw std::runtime_error("PcaLsci requires an active CUDA context");
  return context;
}

__global__ void ones_kernel(float *values, int count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < count) values[p] = 1.0f;
}

template <typename T>
__global__ void prepare_kernel(const T *input, float *stack, double2 *moments, int *invalid,
                               int pixels, int frames) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  double sum = 0.0, squared = 0.0;
  for (int t = 0; t < frames; ++t) {
    const float value = static_cast<float>(input[static_cast<size_t>(t) * pixels + p]);
    if (!isfinite(value) || value < 0.0f) atomicExch(invalid, 1);
    stack[static_cast<size_t>(t) * pixels + p] = value;
    sum += static_cast<double>(value);
    squared += static_cast<double>(value) * value;
  }
  moments[p] = make_double2(sum, squared);
}

__global__ void center_kernel(float *stack, const float *means, int pixels, size_t count) {
  const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p < count) stack[p] -= means[p / pixels];
}

__global__ void symmetrize_kernel(const float *gram, double *covariance, int frames) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= frames * frames) return;
  const int row = p % frames;
  const int column = p / frames;
  covariance[p] = 0.5 * (static_cast<double>(gram[p]) + gram[row * frames + column]);
}

__global__ void static_moments_kernel(double2 *moments, const float *scores,
                                      const float2 *coefficients, int pixels, int components,
                                      double mean_sum, double mean_squared_sum) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  double sum = mean_sum, squared = mean_squared_sum;
  for (int c = 0; c < components; ++c) {
    const double score = scores[static_cast<size_t>(c) * pixels + p];
    const float2 coefficient = coefficients[c];
    sum += score * coefficient.x;
    squared += score * score + 2.0 * score * coefficient.y;
  }
  moments[p] = make_double2(sum, squared);
}

__global__ void reconstructed_frame_kernel(float *output, const float *scores,
                                           const float *basis, int pixels, int components,
                                           int frames, int center, float mean) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  float value = mean;
  for (int c = 0; c < components; ++c)
    value += scores[static_cast<size_t>(c) * pixels + p] * basis[c * frames + center];
  output[p] = value;
}

__global__ void subtract_kernel(float *output, const float *static_map, int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < pixels) output[p] -= static_map[p];
}

class PcaLsci final : public holoflow::core::ISyncTask {
public:
  PcaLsci(PcaLsciSettings settings, const holoflow::core::TDesc &desc, cudaStream_t stream)
      : settings_(settings), desc_(desc), stream_(stream),
        frames_(static_cast<int>(desc.shape[0])), height_(static_cast<int>(desc.shape[1])),
        width_(static_cast<int>(desc.shape[2])), pixels_(height_ * width_),
        stack_(curaii::make_unique_device_ptr<float>(static_cast<size_t>(frames_) * pixels_)),
        ones_(curaii::make_unique_device_ptr<float>(pixels_)),
        means_(curaii::make_unique_device_ptr<float>(frames_)),
        gram_(curaii::make_unique_device_ptr<float>(static_cast<size_t>(frames_) * frames_)),
        covariance_(curaii::make_unique_device_ptr<double>(static_cast<size_t>(frames_) * frames_)),
        eigenvalues_(curaii::make_unique_device_ptr<double>(frames_)),
        basis_(curaii::make_unique_device_ptr<float>(static_cast<size_t>(frames_) * frames_)),
        coefficients_(curaii::make_unique_device_ptr<float2>(frames_)),
        original_moments_(curaii::make_unique_device_ptr<double2>(pixels_)),
        static_moments_(curaii::make_unique_device_ptr<double2>(pixels_)),
        spatial_scratch_(curaii::make_unique_device_ptr<double2>(pixels_)),
        static_map_(curaii::make_unique_device_ptr<float>(pixels_)),
        invalid_(curaii::make_unique_device_ptr<int>(1)),
        solver_info_(curaii::make_unique_device_ptr<int>(1)) {
    context_ = context_for_stream(stream_);
    CUBLAS_CHECK(cublasSetStream(blas_.get(), stream_));
    CUBLAS_CHECK(cublasSetMathMode(blas_.get(), CUBLAS_PEDANTIC_MATH));
    CUSOLVER_CHECK(cusolverDnSetStream(solver_.get(), stream_));
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, frames_, covariance_.get(),
                                               frames_, eigenvalues_.get(), &work_size));
    solver_work_ = curaii::make_unique_device_ptr<double>(work_size);
    solver_work_size_ = work_size;
    tile_pixels_ = (std::min)(size_t{65536}, static_cast<size_t>(pixels_));
    for (;;) {
      try {
        scores_ = curaii::make_unique_device_ptr<float>(tile_pixels_ * frames_);
        break;
      } catch (const curaii::CudaError &error) {
        if (error.code() != cudaErrorMemoryAllocation || tile_pixels_ == 1) throw;
        tile_pixels_ = (std::max)(size_t{1}, tile_pixels_ / 2);
      }
    }
    ones_kernel<<<(pixels_ + 255) / 256, 256, 0, stream_>>>(ones_.get(), pixels_);
    CUDA_CHECK(cudaGetLastError());
  }

  [[nodiscard]] bool can_reuse(const holoflow::core::TDesc &desc, cudaStream_t stream) const {
    return desc.shape == desc_.shape && desc.strides == desc_.strides &&
           desc.dtype == desc_.dtype && desc.mem_loc == desc_.mem_loc &&
           desc.offset == desc_.offset && context_for_stream(stream) == context_;
  }

  void reconfigure(PcaLsciSettings settings, cudaStream_t stream) {
    settings_ = settings;
    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(blas_.get(), stream_));
      CUSOLVER_CHECK(cusolverDnSetStream(solver_.get(), stream_));
    }
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const int blocks = (pixels_ + 255) / 256;
    CUDA_CHECK(cudaMemsetAsync(invalid_.get(), 0, sizeof(int), stream_));
    const void *input = ctx.inputs[0].data();
    switch (desc_.dtype) {
    case holoflow::core::DType::U8:
      prepare_kernel<<<blocks, 256, 0, stream_>>>(static_cast<const uint8_t *>(input), stack_.get(),
          original_moments_.get(), invalid_.get(), pixels_, frames_);
      break;
    case holoflow::core::DType::U16:
      prepare_kernel<<<blocks, 256, 0, stream_>>>(static_cast<const uint16_t *>(input), stack_.get(),
          original_moments_.get(), invalid_.get(), pixels_, frames_);
      break;
    case holoflow::core::DType::F32:
      prepare_kernel<<<blocks, 256, 0, stream_>>>(static_cast<const float *>(input), stack_.get(),
          original_moments_.get(), invalid_.get(), pixels_, frames_);
      break;
    default: throw std::logic_error("PcaLsci unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    int invalid = 0;
    CUDA_CHECK(cudaMemcpyAsync(&invalid, invalid_.get(), sizeof(int), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (invalid) throw std::invalid_argument("PcaLsci requires finite nonnegative intensities");

    constexpr float one = 1.0f, zero = 0.0f;
    const float scale = 1.0f / pixels_;
    CUBLAS_CHECK(cublasSgemv(blas_.get(), CUBLAS_OP_T, pixels_, frames_, &scale, stack_.get(),
                             pixels_, ones_.get(), 1, &zero, means_.get(), 1));
    std::vector<float> means(frames_);
    CUDA_CHECK(cudaMemcpyAsync(means.data(), means_.get(), means.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const size_t count = static_cast<size_t>(pixels_) * frames_;
    center_kernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream_>>>(
        stack_.get(), means_.get(), pixels_, count);
    CUBLAS_CHECK(cublasSgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, frames_, frames_, pixels_,
                             &one, stack_.get(), pixels_, stack_.get(), pixels_, &zero,
                             gram_.get(), frames_));
    symmetrize_kernel<<<(frames_ * frames_ + 255) / 256, 256, 0, stream_>>>(
        gram_.get(), covariance_.get(), frames_);
    CUDA_CHECK(cudaGetLastError());
    CUSOLVER_CHECK(cusolverDnDsyevd(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER, frames_, covariance_.get(), frames_,
                                    eigenvalues_.get(), solver_work_.get(), solver_work_size_,
                                    solver_info_.get()));
    std::vector<double> eigenvalues(frames_), eigenvectors(static_cast<size_t>(frames_) * frames_);
    int info = 0;
    CUDA_CHECK(cudaMemcpyAsync(eigenvalues.data(), eigenvalues_.get(), frames_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(eigenvectors.data(), covariance_.get(),
                               eigenvectors.size() * sizeof(double), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&info, solver_info_.get(), sizeof(int), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (info != 0) throw std::runtime_error("PcaLsci eigendecomposition did not converge");

    double total = 0.0;
    for (double value : eigenvalues) {
      if (!std::isfinite(value)) throw std::runtime_error("PcaLsci covariance overflowed");
      total += (std::max)(0.0, static_cast<double>(value));
    }
    const double threshold = total / frames_;
    std::vector<float> basis;
    std::vector<float2> coefficients;
    if (total > 0.0) {
      for (int c = frames_ - 1; c >= 0; --c) {
        if ((std::max)(0.0, static_cast<double>(eigenvalues[c])) < threshold) continue;
        double sum = 0.0, mean_dot = 0.0;
        for (int t = 0; t < frames_; ++t) {
          const float v = static_cast<float>(eigenvectors[static_cast<size_t>(c) * frames_ + t]);
          basis.push_back(v);
          sum += v;
          mean_dot += v * means[t];
        }
        coefficients.push_back(make_float2(static_cast<float>(sum),
                                            static_cast<float>(mean_dot)));
      }
    }
    const int components = static_cast<int>(coefficients.size());
    logger()->debug("[PcaLsci] retained {}/{} static modes (threshold={})",
                    components, frames_, threshold);
    if (components) {
      CUDA_CHECK(cudaMemcpyAsync(basis_.get(), basis.data(), basis.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(coefficients_.get(), coefficients.data(),
                                 coefficients.size() * sizeof(float2), cudaMemcpyHostToDevice,
                                 stream_));
    }
    const double mean_sum = std::accumulate(means.begin(), means.end(), 0.0);
    double mean_squared_sum = 0.0;
    for (float mean : means) mean_squared_sum += static_cast<double>(mean) * mean;
    for (size_t first = 0; first < static_cast<size_t>(pixels_); first += tile_pixels_) {
      const int rows = static_cast<int>((std::min)(tile_pixels_, static_cast<size_t>(pixels_) - first));
      if (components) {
        CUBLAS_CHECK(cublasSgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, rows, components,
                                 frames_, &one, stack_.get() + first, pixels_, basis_.get(),
                                 frames_, &zero, scores_.get(), rows));
      }
      if (settings_.map == PcaLsciMap::PcaFrame) {
        reconstructed_frame_kernel<<<(rows + 255) / 256, 256, 0, stream_>>>(
            reinterpret_cast<float *>(ctx.outputs[0].data()) + first, scores_.get(),
            basis_.get(), rows, components, frames_, frames_ / 2, means[frames_ / 2]);
      } else {
        static_moments_kernel<<<(rows + 255) / 256, 256, 0, stream_>>>(
            static_moments_.get() + first, scores_.get(), coefficients_.get(), rows, components,
            mean_sum, mean_squared_sum);
      }
    }
    CUDA_CHECK(cudaGetLastError());

    if (settings_.map == PcaLsciMap::PcaFrame) return holoflow::core::OpResult::Ok;

    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
    if (settings_.map != PcaLsciMap::Static) {
      enqueue_spatial_contrast_from_moments(original_moments_.get(), spatial_scratch_.get(),
          output, width_, height_, frames_, settings_.window_size, stream_);
    }
    if (settings_.map != PcaLsciMap::Original) {
      float *static_output = settings_.map == PcaLsciMap::Static ? output : static_map_.get();
      enqueue_spatial_contrast_from_moments(static_moments_.get(), spatial_scratch_.get(),
          static_output, width_, height_, frames_, settings_.window_size, stream_);
      if (settings_.map == PcaLsciMap::Dynamic) {
        subtract_kernel<<<blocks, 256, 0, stream_>>>(output, static_map_.get(), pixels_);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    return holoflow::core::OpResult::Ok;
  }

private:
  PcaLsciSettings settings_;
  holoflow::core::TDesc desc_;
  cudaStream_t stream_;
  CUcontext context_;
  int frames_, height_, width_, pixels_;
  size_t tile_pixels_;
  int solver_work_size_;
  DevPtr<float> stack_, ones_, means_, gram_, basis_, scores_, static_map_;
  DevPtr<double> covariance_, eigenvalues_, solver_work_;
  DevPtr<float2> coefficients_;
  DevPtr<double2> original_moments_, static_moments_, spatial_scratch_;
  DevPtr<int> invalid_, solver_info_;
  curaii::CublasHandle blas_;
  curaii::CusolverDnHandle solver_;
};

} // namespace

holoflow::core::InferResult PcaLsciFactory::infer(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<PcaLsciSettings>();
  if (inputs.size() != 1) throw std::invalid_argument("PcaLsci expects one input");
  const auto &input = inputs.front();
  if (input.rank() != 3 || input.shape[0] < 2 || input.shape[0] > 512 ||
      input.shape[1] == 0 || input.shape[2] == 0 ||
      input.shape[1] > static_cast<size_t>((std::numeric_limits<int>::max)()) / input.shape[2] ||
      input.mem_loc != holoflow::core::MemLoc::Device ||
      (input.dtype != holoflow::core::DType::U8 &&
       input.dtype != holoflow::core::DType::U16 &&
       input.dtype != holoflow::core::DType::F32) ||
      input.strides != holoflow::core::TDesc(input.shape, input.dtype, input.mem_loc).strides)
    throw std::invalid_argument("PcaLsci requires contiguous device [2..512,H,W] U8/U16/F32 input");
  if (settings.window_size < 1 || settings.window_size > 63 || settings.window_size % 2 == 0)
    throw std::invalid_argument("PcaLsci spatial window must be odd and between 1 and 63");
  return {.input_descs = {input},
          .output_descs = {holoflow::core::TDesc({1, input.shape[1], input.shape[2]},
                                                holoflow::core::DType::F32, input.mem_loc)},
          .in_place = {}, .owned_inputs = {false}, .owned_outputs = {false},
          .kind = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask> PcaLsciFactory::create(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<PcaLsci>(settings.get<PcaLsciSettings>(), inputs.front(), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask> PcaLsciFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask> old_task,
    std::span<const holoflow::core::TDesc> inputs,
    const nlohmann::json &settings, const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  ctx.invalidate_execution();
  auto *pca_lsci = dynamic_cast<PcaLsci *>(old_task.get());
  if (pca_lsci != nullptr && pca_lsci->can_reuse(inputs.front(), ctx.stream)) {
    pca_lsci->reconfigure(settings.get<PcaLsciSettings>(), ctx.stream);
    return old_task;
  }
  old_task.reset();
  return create(inputs, settings, ctx);
}

} // namespace holotask::syncs
