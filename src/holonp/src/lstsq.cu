// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/lstsq.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"

namespace holonp {

void to_json(nlohmann::json &j, const LstsqSettings &s) { j = {{"rcond", s.rcond}}; }
void from_json(const nlohmann::json &j, LstsqSettings &s) {
  if (j.is_null()) {
    s = {};
    return;
  }
  s.rcond = j.value("rcond", 1e-6f);
}

namespace {

inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("Lstsq: " + message);
}

struct Layout {
  int  rows;
  int  cols;
  int  rhs;
  bool vector_rhs;
};

__global__ void lstsq_transpose_kernel(const float *__restrict__ input, float *__restrict__ output,
                                       int rows, int cols) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= rows * cols)
    return;

  const int row            = index / cols;
  const int col            = index % cols;
  output[row + col * rows] = input[index];
}

__global__ void lstsq_solution_kernel(const float *__restrict__ u, const float *__restrict__ s,
                                      const float *__restrict__ vt, const float *__restrict__ b,
                                      float *__restrict__ x, std::uint16_t *__restrict__ rank,
                                      int rows, int cols, int rhs, int vector_rhs, float rcond,
                                      const int *__restrict__ info) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int count = cols * rhs;
  if (index >= count)
    return;
  if (*info != 0) {
    x[index] = __int_as_float(0x7fc00000);
    if (index == 0)
      *rank = 0;
    return;
  }

  const int   parameter = index / rhs;
  const int   col       = index % rhs;
  const float threshold = rcond * s[0];
  float       result    = 0.0f;
  for (int k = 0; k < cols; ++k) {
    if (s[k] <= threshold)
      continue;
    float projection = 0.0f;
    for (int q = 0; q < rows; ++q)
      projection += u[q + k * rows] * b[vector_rhs ? q : q * rhs + col];
    result += vt[k + parameter * cols] * projection / s[k];
  }
  x[index] = result;

  if (index == 0) {
    std::uint16_t computed_rank = 0;
    for (int k = 0; k < cols; ++k)
      if (s[k] > threshold)
        ++computed_rank;
    *rank = computed_rank;
  }
}

__global__ void lstsq_residual_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                      const float *__restrict__ x, float *__restrict__ residuals,
                                      int rows, int cols, int rhs, int vector_rhs,
                                      const std::uint16_t *__restrict__ rank,
                                      const int *__restrict__ info) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= rhs)
    return;
  if (*info != 0 || *rank < cols) {
    residuals[col] = __int_as_float(0x7fc00000);
    return;
  }

  float residual = 0.0f;
  for (int q = 0; q < rows; ++q) {
    float estimate = 0.0f;
    for (int p = 0; p < cols; ++p)
      estimate += a[q * cols + p] * x[p * rhs + col];
    const float target = b[vector_rhs ? q : q * rhs + col];
    const float error  = target - estimate;
    residual += error * error;
  }
  residuals[col] = residual;
}

class Lstsq : public holoflow::core::ISyncTask {
public:
  Lstsq(LstsqSettings settings, holoflow::core::TDesc a_desc, holoflow::core::TDesc b_desc,
        Layout layout, cudaStream_t stream)
      : settings_(std::move(settings)), a_desc_(std::move(a_desc)), b_desc_(std::move(b_desc)),
        layout_(layout), stream_(stream) {
    CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    CUSOLVER_CHECK(
        cusolverDnSgesvd_bufferSize(handle_, layout_.rows, layout_.cols, &workspace_elements_));
    workspace_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(workspace_elements_));
    info_      = curaii::make_unique_device_ptr<int>(1);
    a_copy_ =
        curaii::make_unique_device_ptr<float>(static_cast<size_t>(layout_.rows) * layout_.cols);
    singular_values_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(layout_.cols));
    ut_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(layout_.rows) * layout_.cols);
    vt_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(layout_.cols) * layout_.cols);
  }

  ~Lstsq() override {
    if (handle_ != nullptr)
      CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    constexpr int block      = 256;
    const int     a_elements = layout_.rows * layout_.cols;
    const int     a_grid     = (a_elements + block - 1) / block;
    lstsq_transpose_kernel<<<a_grid, block, 0, stream_>>>(
        reinterpret_cast<const float *>(ctx.inputs[0].data()), a_copy_.get(), layout_.rows,
        layout_.cols);
    CUDA_CHECK(cudaGetLastError());
    CUSOLVER_CHECK(cusolverDnSgesvd(handle_, 'S', 'S', layout_.rows, layout_.cols, a_copy_.get(),
                                    layout_.rows, singular_values_.get(), ut_.get(), layout_.rows,
                                    vt_.get(), layout_.cols, workspace_.get(), workspace_elements_,
                                    nullptr, info_.get()));
    CUDA_CHECK(cudaMemcpyAsync(ctx.outputs[3].data(), singular_values_.get(),
                               static_cast<size_t>(layout_.cols) * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));

    const int x_elements = layout_.cols * layout_.rhs;
    const int grid       = (x_elements + block - 1) / block;
    lstsq_solution_kernel<<<grid, block, 0, stream_>>>(
        ut_.get(), singular_values_.get(), vt_.get(),
        reinterpret_cast<const float *>(ctx.inputs[1].data()),
        reinterpret_cast<float *>(ctx.outputs[0].data()),
        reinterpret_cast<std::uint16_t *>(ctx.outputs[2].data()), layout_.rows, layout_.cols,
        layout_.rhs, layout_.vector_rhs, settings_.rcond, info_.get());
    CUDA_CHECK(cudaGetLastError());

    if (layout_.rows > layout_.cols) {
      const int residual_grid = (layout_.rhs + block - 1) / block;
      lstsq_residual_kernel<<<residual_grid, block, 0, stream_>>>(
          reinterpret_cast<const float *>(ctx.inputs[0].data()),
          reinterpret_cast<const float *>(ctx.inputs[1].data()),
          reinterpret_cast<const float *>(ctx.outputs[0].data()),
          reinterpret_cast<float *>(ctx.outputs[1].data()), layout_.rows, layout_.cols, layout_.rhs,
          layout_.vector_rhs, reinterpret_cast<const std::uint16_t *>(ctx.outputs[2].data()),
          info_.get());
      CUDA_CHECK(cudaGetLastError());
    }
    return holoflow::core::OpResult::Ok;
  }

  const LstsqSettings         &settings() const { return settings_; }
  const holoflow::core::TDesc &a_desc() const { return a_desc_; }
  const holoflow::core::TDesc &b_desc() const { return b_desc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  LstsqSettings                    settings_;
  holoflow::core::TDesc            a_desc_;
  holoflow::core::TDesc            b_desc_;
  Layout                           layout_;
  cudaStream_t                     stream_;
  cusolverDnHandle_t               handle_ = nullptr;
  int                              workspace_elements_;
  curaii::unique_device_ptr<float> workspace_;
  curaii::unique_device_ptr<int>   info_;
  curaii::unique_device_ptr<float> a_copy_;
  curaii::unique_device_ptr<float> singular_values_;
  curaii::unique_device_ptr<float> ut_;
  curaii::unique_device_ptr<float> vt_;
};

Layout infer_layout(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings) {
  check(input_descs.size() == 2, "expected exactly two inputs");
  const auto &a        = input_descs[0];
  const auto &b        = input_descs[1];
  const auto  settings = jsettings.get<LstsqSettings>();
  check(a.mem_loc == holoflow::core::MemLoc::Device && b.mem_loc == holoflow::core::MemLoc::Device,
        "only Device tensors are supported");
  check(a.dtype == holoflow::core::DType::F32 && b.dtype == holoflow::core::DType::F32,
        "only F32 inputs are currently supported");
  check(a.shape.size() == 2, "A must be 2-D");
  check(b.shape.size() == 1 || b.shape.size() == 2, "B must be 1-D or 2-D");
  check(utils::is_c_contiguous(a) && utils::is_c_contiguous(b), "inputs must be C-contiguous");
  check(a.shape[0] > 0 && a.shape[1] > 0 && b.shape[0] == a.shape[0],
        "incompatible matrix dimensions");
  check(a.shape[0] >= a.shape[1], "underdetermined systems are not yet supported");
  check(std::isfinite(settings.rcond) && settings.rcond >= 0.0f,
        "rcond must be finite and nonnegative");
  const int rhs = b.shape.size() == 1 ? 1 : static_cast<int>(b.shape[1]);
  check(rhs > 0, "B must have at least one right-hand side");
  check(a.shape[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            a.shape[1] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            rhs <= std::numeric_limits<int>::max(),
        "matrix dimensions exceed cuSOLVER limits");
  return {.rows       = static_cast<int>(a.shape[0]),
          .cols       = static_cast<int>(a.shape[1]),
          .rhs        = rhs,
          .vector_rhs = b.shape.size() == 1};
}

} // namespace

holoflow::core::InferResult LstsqFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  const auto  layout  = infer_layout(input_descs, jsettings);
  const auto &b       = input_descs[1];
  const auto  x_shape = b.shape.size() == 1 ? std::vector<size_t>{static_cast<size_t>(layout.cols)}
                                            : std::vector<size_t>{static_cast<size_t>(layout.cols),
                                                                  static_cast<size_t>(layout.rhs)};
  const auto  rank_desc = holonp::utils::make_contiguous_desc({}, holoflow::core::DType::U16,
                                                              holoflow::core::MemLoc::Device);
  return {.input_descs   = {input_descs[0], input_descs[1]},
          .output_descs  = {holoflow::core::TDesc(x_shape, holoflow::core::DType::F32,
                                                  holoflow::core::MemLoc::Device),
                            holoflow::core::TDesc(
                                {layout.rows > layout.cols ? static_cast<size_t>(layout.rhs) : 0},
                                holoflow::core::DType::F32, holoflow::core::MemLoc::Device),
                            rank_desc,
                            holoflow::core::TDesc({static_cast<size_t>(layout.cols)},
                                                  holoflow::core::DType::F32,
                                                  holoflow::core::MemLoc::Device)},
          .in_place      = {},
          .owned_inputs  = {false, false},
          .owned_outputs = {false, false, false, false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
LstsqFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto layout = infer_layout(input_descs, jsettings);
  return std::make_unique<Lstsq>(jsettings.get<LstsqSettings>(), input_descs[0], input_descs[1],
                                 layout, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
LstsqFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                     std::span<const holoflow::core::TDesc>     input_descs,
                     const nlohmann::json                      &jsettings,
                     const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old      = dynamic_cast<Lstsq *>(old_task.get());
  const auto settings = jsettings.get<LstsqSettings>();
  if (old != nullptr && old->settings() == settings && input_descs.size() == 2 &&
      utils::same_desc(input_descs[0], old->a_desc()) &&
      utils::same_desc(input_descs[1], old->b_desc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
