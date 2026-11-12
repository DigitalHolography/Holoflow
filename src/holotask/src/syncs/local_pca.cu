// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0 (the "License");

#include "holotask/syncs/local_pca.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda.h>

#include "curaii/cuda.hh"
#include "curaii/cublas.hh"
#include "curaii/cusolver.hh"
#include "holotask/syncs/spatial_contrast.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const LocalPcaSettings &s) {
  const char *map = s.map == LocalPcaMap::Original ? "original" :
                    s.map == LocalPcaMap::Static ? "static" :
                    s.map == LocalPcaMap::Dynamic ? "dynamic" : "energy";
  j = {{"window_size", s.window_size}, {"stride", s.stride},
       {"begin", s.begin}, {"end", s.end}, {"automatic", s.automatic}, {"map", map}};
}

void from_json(const nlohmann::json &j, LocalPcaSettings &s) {
  s.window_size = j.value("window_size", 5);
  s.stride = j.value("stride", 1);
  s.begin = j.value("begin", 0);
  s.end = j.value("end", 15);
  s.automatic = j.value("automatic", false);
  const auto map = j.value("map", std::string{"energy"});
  if (map == "energy") s.map = LocalPcaMap::Energy;
  else if (map == "original") s.map = LocalPcaMap::Original;
  else if (map == "static") s.map = LocalPcaMap::Static;
  else if (map == "dynamic") s.map = LocalPcaMap::Dynamic;
  else throw std::invalid_argument("LocalPca map must be energy, original, static or dynamic");
}

namespace {

constexpr int kBatch = 1024;

__device__ int wrap(int x, int extent) {
  x %= extent;
  return x < 0 ? x + extent : x;
}

template <typename T>
__global__ void gather_centered_patches(const T *input, float *patches, float *means,
                                        int width, int height,
                                        int out_width, int stride, int window, int frames,
                                        int first, int count) {
  const int p = blockIdx.x;
  const int t = threadIdx.x;
  if (p >= count || t >= frames) return;
  const int center = first + p;
  const int x = (center % out_width) * stride;
  const int y = (center / out_width) * stride;
  const int radius = window / 2;
  const int pixels = window * window;
  double sum = 0.0;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      const int pixel = wrap(y + dy, height) * width + wrap(x + dx, width);
      sum += static_cast<double>(input[static_cast<size_t>(t) * width * height + pixel]);
    }
  const double mean = sum / pixels;
  if (means) means[p * frames + t] = static_cast<float>(mean);
  int j = 0;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      const int pixel = wrap(y + dy, height) * width + wrap(x + dx, width);
      patches[(static_cast<size_t>(p) * frames + t) * pixels + j++] =
          static_cast<float>(static_cast<double>(input[static_cast<size_t>(t) * width * height + pixel]) - mean);
    }
}

template <typename T>
__global__ void original_moments_kernel(const T *input, double2 *moments,
                                        int pixels, int frames) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  double sum = 0.0, squared = 0.0;
  for (int t = 0; t < frames; ++t) {
    const double value = static_cast<double>(input[static_cast<size_t>(t) * pixels + p]);
    sum += value;
    squared += value * value;
  }
  moments[p] = make_double2(sum, squared);
}

__global__ void reconstructed_moments_kernel(
    const float *patches, const float *means, const float *vectors,
    const float *eigenvalues, const int *info, double2 *moments, int *invalid,
    int frames, int window, int begin, int end, bool automatic,
    int first, int count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  if (info[p] != 0) { atomicExch(invalid, 1); return; }
  double total = 0.0;
  for (int c = 0; c < frames; ++c) {
    const float value = eigenvalues[p * frames + c];
    if (!isfinite(value)) { atomicExch(invalid, 1); return; }
    total += fmax(0.0, static_cast<double>(value));
  }
  const double threshold = total / frames;
  const int center_index = (window * window) / 2;
  double scores[32];
  for (int c = 0; c < frames; ++c) {
    const bool is_static = automatic
        ? (total > 0.0 && fmax(0.0, static_cast<double>(eigenvalues[p * frames + c])) >= threshold)
        : (c < begin || c >= end);
    double score = 0.0;
    if (is_static)
      for (int s = 0; s < frames; ++s) {
        const double centered = patches[(static_cast<size_t>(p) * frames + s) * window * window + center_index];
        score += centered * vectors[(static_cast<size_t>(p) * frames + c) * frames + s];
      }
    scores[c] = score;
  }
  double sum = 0.0, squared = 0.0;
  for (int t = 0; t < frames; ++t) {
    double value = means[p * frames + t];
    for (int c = 0; c < frames; ++c)
      value += scores[c] * vectors[(static_cast<size_t>(p) * frames + c) * frames + t];
    sum += value;
    squared += value * value;
  }
  moments[first + p] = make_double2(sum, squared);
}

__global__ void subtract_maps(float *output, const float *static_map, int pixels) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < pixels) output[p] -= static_map[p];
}

__global__ void retained_fraction(const float *eigenvalues, const int *info, float *output,
                                   int frames, int begin, int end, bool automatic,
                                   int first, int count,
                                   int *invalid) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  if (info[p] != 0) {
    atomicExch(invalid, 1);
    output[first + p] = 0.0f;
    return;
  }
  double total = 0.0, retained = 0.0;
  for (int i = 0; i < frames; ++i) {
    const float raw = eigenvalues[p * frames + i];
    if (!isfinite(raw)) { atomicExch(invalid, 1); break; }
    const double value = fmax(0.0, static_cast<double>(raw));
    total += value;
  }
  const double threshold = total / frames;
  for (int i = 0; i < frames; ++i) {
    const double value = fmax(0.0, static_cast<double>(eigenvalues[p * frames + i]));
    if (automatic ? value < threshold : (i >= begin && i < end)) retained += value;
  }
  output[first + p] = total > 0.0 ? static_cast<float>(retained / total) : 0.0f;
}

class LocalPca final : public holoflow::core::ISyncTask {
public:
  LocalPca(LocalPcaSettings settings, const holoflow::core::TDesc &desc, cudaStream_t stream)
      : settings_(settings), desc_(desc), stream_(stream),
        frames_(static_cast<int>(desc.shape[0])), height_(static_cast<int>(desc.shape[1])),
        width_(static_cast<int>(desc.shape[2])),
        out_width_((width_ - 1) / settings.stride + 1),
        out_height_((height_ - 1) / settings.stride + 1),
        gram_(curaii::make_unique_device_ptr<float>(kBatch * frames_ * frames_)),
        patches_(curaii::make_unique_device_ptr<float>(
            static_cast<size_t>(kBatch) * frames_ * settings.window_size * settings.window_size)),
        eigenvalues_(curaii::make_unique_device_ptr<float>(kBatch * frames_)),
        info_(curaii::make_unique_device_ptr<int>(kBatch)),
        invalid_(curaii::make_unique_device_ptr<int>(1)) {
    if (settings_.map != LocalPcaMap::Energy) {
      means_ = curaii::make_unique_device_ptr<float>(kBatch * frames_);
      original_moments_ = curaii::make_unique_device_ptr<double2>(width_ * height_);
      static_moments_ = curaii::make_unique_device_ptr<double2>(width_ * height_);
      spatial_scratch_ = curaii::make_unique_device_ptr<double2>(width_ * height_);
      if (settings_.map == LocalPcaMap::Dynamic)
        static_map_ = curaii::make_unique_device_ptr<float>(width_ * height_);
    }
    CUcontext context = nullptr;
    const CUresult result = stream != nullptr
        ? cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context)
        : cuCtxGetCurrent(&context);
    if (result != CUDA_SUCCESS || context == nullptr)
      throw std::runtime_error("LocalPca requires an active CUDA context");
    context_ = context;
    CUBLAS_CHECK(cublasSetStream(blas_.get(), stream_));
    CUBLAS_CHECK(cublasSetMathMode(blas_.get(), CUBLAS_PEDANTIC_MATH));
    CUSOLVER_CHECK(cusolverDnSetStream(solver_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&jacobi_));
    CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(jacobi_, 1));
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(
        solver_.get(), settings_.map == LocalPcaMap::Energy ? CUSOLVER_EIG_MODE_NOVECTOR
                                                           : CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        frames_, gram_.get(), frames_, eigenvalues_.get(), &work_size, jacobi_, kBatch));
    work_ = curaii::make_unique_device_ptr<float>(work_size);
    work_size_ = work_size;
  }

  ~LocalPca() override { if (jacobi_) cusolverDnDestroySyevjInfo(jacobi_); }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto started = std::chrono::steady_clock::now();
    const int patches = out_width_ * out_height_;
    logger()->info("[LocalPca] processing {} patches ({} frames, {}x{} kernel, stride {}, selection {})",
                   patches, frames_, settings_.window_size, settings_.window_size,
                   settings_.stride, settings_.automatic ? "automatic" : "manual");
    // GPU work can continue after all batches have been enqueued. Keep reporting while the
    // execution thread waits for the stream, without synchronizing every batch.
    std::jthread progress([this, started](std::stop_token stop) {
      std::mutex mutex;
      std::condition_variable_any wake;
      std::unique_lock lock(mutex);
      while (!stop.stop_requested()) {
        wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
        if (!stop.stop_requested()) {
          const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - started).count();
          logger()->info("[LocalPca] still processing ({} s elapsed)", elapsed);
        }
      }
    });
    const auto *input = ctx.inputs[0].data();
    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
    if (settings_.map != LocalPcaMap::Energy) {
      if (desc_.dtype == holoflow::core::DType::U8)
        original_moments_kernel<<<(width_ * height_ + 255) / 256, 256, 0, stream_>>>(
            reinterpret_cast<const uint8_t *>(input), original_moments_.get(), width_ * height_, frames_);
      else if (desc_.dtype == holoflow::core::DType::U16)
        original_moments_kernel<<<(width_ * height_ + 255) / 256, 256, 0, stream_>>>(
            reinterpret_cast<const uint16_t *>(input), original_moments_.get(), width_ * height_, frames_);
      else
        original_moments_kernel<<<(width_ * height_ + 255) / 256, 256, 0, stream_>>>(
            reinterpret_cast<const float *>(input), original_moments_.get(), width_ * height_, frames_);
      CUDA_CHECK(cudaGetLastError());
      if (settings_.map == LocalPcaMap::Original) {
        enqueue_spatial_contrast_from_moments(original_moments_.get(), spatial_scratch_.get(),
            output, width_, height_, frames_, settings_.window_size, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        progress.request_stop();
        logger()->info("[LocalPca] completed original LSCI map");
        return holoflow::core::OpResult::Ok;
      }
    }
    CUDA_CHECK(cudaMemsetAsync(invalid_.get(), 0, sizeof(int), stream_));
    for (int first = 0; first < patches; first += kBatch) {
      if (ctx.cancelled && ctx.cancelled->load(std::memory_order_acquire)) {
        // The task owns the batch workspace, so finish already submitted work before returning.
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        progress.request_stop();
        logger()->info("[LocalPca] cancelled after {} of {} patches were submitted",
                       first, patches);
        return holoflow::core::OpResult::Cancelled;
      }
      // Bound queued GPU work to one batch so shutdown does not wait for the whole image.
      if (first > 0)
        CUDA_CHECK(cudaStreamSynchronize(stream_));
      const int count = (std::min)(kBatch, patches - first);
      if (desc_.dtype == holoflow::core::DType::U8)
        enqueue(reinterpret_cast<const uint8_t *>(input), first, count);
      else if (desc_.dtype == holoflow::core::DType::U16)
        enqueue(reinterpret_cast<const uint16_t *>(input), first, count);
      else
        enqueue(reinterpret_cast<const float *>(input), first, count);
      CUSOLVER_CHECK(cusolverDnSsyevjBatched(
          solver_.get(), settings_.map == LocalPcaMap::Energy ? CUSOLVER_EIG_MODE_NOVECTOR
                                                             : CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER,
          frames_, gram_.get(), frames_, eigenvalues_.get(), work_.get(),
          work_size_, info_.get(), jacobi_, count));
      if (settings_.map == LocalPcaMap::Energy) {
        retained_fraction<<<(count + 127) / 128, 128, 0, stream_>>>(
            eigenvalues_.get(), info_.get(), output, frames_, settings_.begin,
            settings_.end, settings_.automatic, first, count, invalid_.get());
      } else {
        reconstruct(first, count);
      }
      CUDA_CHECK(cudaGetLastError());
    }
    int invalid = 0;
    CUDA_CHECK(cudaMemcpyAsync(&invalid, invalid_.get(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    progress.request_stop();
    if (invalid) throw std::runtime_error("LocalPca eigendecomposition failed");
    if (settings_.map != LocalPcaMap::Energy) {
      if (settings_.map == LocalPcaMap::Dynamic)
        enqueue_spatial_contrast_from_moments(original_moments_.get(), spatial_scratch_.get(),
            output, width_, height_, frames_, settings_.window_size, stream_);
      float *static_output = settings_.map == LocalPcaMap::Static ? output : static_map_.get();
      enqueue_spatial_contrast_from_moments(static_moments_.get(), spatial_scratch_.get(),
          static_output, width_, height_, frames_, settings_.window_size, stream_);
      if (settings_.map == LocalPcaMap::Dynamic)
        subtract_maps<<<(width_ * height_ + 255) / 256, 256, 0, stream_>>>(
            output, static_map_.get(), width_ * height_);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    logger()->info("[LocalPca] completed {} patches in {} ms", patches, elapsed);
    return holoflow::core::OpResult::Ok;
  }

  bool can_reuse(const holoflow::core::TDesc &desc, const LocalPcaSettings &settings,
                 cudaStream_t stream) const {
    CUcontext context = nullptr;
    const CUresult result = stream != nullptr
        ? cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context)
        : cuCtxGetCurrent(&context);
    return result == CUDA_SUCCESS && context == context_ && desc.shape == desc_.shape &&
           desc.dtype == desc_.dtype && desc.mem_loc == desc_.mem_loc &&
           desc.strides == desc_.strides && desc.offset == desc_.offset &&
           settings.stride == settings_.stride &&
           settings.window_size == settings_.window_size &&
           settings.map == settings_.map;
  }

  void reconfigure(LocalPcaSettings settings, cudaStream_t stream) {
    settings_ = settings;
    stream_ = stream;
    CUBLAS_CHECK(cublasSetStream(blas_.get(), stream_));
    CUSOLVER_CHECK(cusolverDnSetStream(solver_.get(), stream_));
  }

private:
  template <typename T> void enqueue(const T *input, int first, int count) {
    gather_centered_patches<<<count, 32, 0, stream_>>>(
        input, patches_.get(), means_.get(), width_, height_, out_width_, settings_.stride,
        settings_.window_size, frames_, first, count);
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f, beta = 0.0f;
    const int pixels = settings_.window_size * settings_.window_size;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N, frames_, frames_, pixels,
        &alpha, patches_.get(), pixels, static_cast<long long>(frames_) * pixels,
        patches_.get(), pixels, static_cast<long long>(frames_) * pixels,
        &beta, gram_.get(), frames_, static_cast<long long>(frames_) * frames_, count));
  }

  void reconstruct(int first, int count) {
    reconstructed_moments_kernel<<<(count + 127) / 128, 128, 0, stream_>>>(
        patches_.get(), means_.get(), gram_.get(), eigenvalues_.get(),
        info_.get(), static_moments_.get(), invalid_.get(),
        frames_, settings_.window_size, settings_.begin, settings_.end,
        settings_.automatic, first, count);
  }

  LocalPcaSettings settings_;
  holoflow::core::TDesc desc_;
  cudaStream_t stream_;
  CUcontext context_ = nullptr;
  int frames_, height_, width_, out_width_, out_height_;
  curaii::CublasHandle blas_;
  curaii::CusolverDnHandle solver_;
  syevjInfo_t jacobi_ = nullptr;
  curaii::unique_device_ptr<float> gram_, patches_, eigenvalues_, work_;
  curaii::unique_device_ptr<float> means_, static_map_;
  curaii::unique_device_ptr<double2> original_moments_, static_moments_, spatial_scratch_;
  curaii::unique_device_ptr<int> info_, invalid_;
  int work_size_ = 0;
};

} // namespace

holoflow::core::InferResult LocalPcaFactory::infer(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &jsettings) const {
  const auto s = jsettings.get<LocalPcaSettings>();
  if (inputs.size() != 1) throw std::invalid_argument("LocalPca expects one input");
  const auto &input = inputs.front();
  if (input.rank() != 3 || input.shape[0] < 2 || input.shape[0] > 32 ||
      input.shape[1] == 0 || input.shape[2] == 0 ||
      input.shape[1] > static_cast<size_t>((std::numeric_limits<int>::max)()) / input.shape[2] ||
      input.mem_loc != holoflow::core::MemLoc::Device ||
      (input.dtype != holoflow::core::DType::U8 &&
       input.dtype != holoflow::core::DType::U16 &&
       input.dtype != holoflow::core::DType::F32) ||
      input.strides != holoflow::core::TDesc(input.shape, input.dtype, input.mem_loc).strides)
    throw std::invalid_argument("LocalPca requires contiguous device [2..32,H,W] U8/U16/F32 input");
  if (s.window_size < 3 || s.window_size > 15 || s.window_size % 2 == 0 ||
      s.stride != 1 ||
      (!s.automatic && (s.begin < 0 || s.begin >= s.end ||
                        s.end > static_cast<int>(input.shape[0]))))
    throw std::invalid_argument("LocalPca requires odd 3..15 window, valid stride and component range");
  return {.input_descs = {input},
          .output_descs = {holoflow::core::TDesc(
              {1, (input.shape[1] - 1) / s.stride + 1,
               (input.shape[2] - 1) / s.stride + 1},
              holoflow::core::DType::F32, input.mem_loc)},
          .in_place = {}, .owned_inputs = {false}, .owned_outputs = {false},
          .kind = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask> LocalPcaFactory::create(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<LocalPca>(settings.get<LocalPcaSettings>(), inputs.front(), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask> LocalPcaFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask> old_task,
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  ctx.invalidate_execution();
  auto *task = dynamic_cast<LocalPca *>(old_task.get());
  const auto config = settings.get<LocalPcaSettings>();
  if (task && task->can_reuse(inputs.front(), config, ctx.stream)) {
    task->reconfigure(config, ctx.stream);
    return old_task;
  }
  old_task.reset();
  return create(inputs, settings, ctx);
}

} // namespace holotask::syncs
