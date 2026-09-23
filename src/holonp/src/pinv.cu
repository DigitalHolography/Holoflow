// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/pinv.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"

namespace holonp {

void to_json(nlohmann::json &j, const PinvSettings &s) { j = {{"rcond", s.rcond}}; }
void from_json(const nlohmann::json &j, PinvSettings &s) {
  if (j.is_null()) {
    s = {};
    return;
  }
  s.rcond = j.value("rcond", 1e-6f);
}

namespace {

inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("Pinv: " + message);
}

__global__ void pinv_kernel(const float *__restrict__ ut, const float *__restrict__ s,
                            const float *__restrict__ vt, float *__restrict__ output, int rows,
                            int cols, int rank, float rcond) {
  const int output_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int output_size  = cols * rows;
  if (output_index >= output_size)
    return;

  const int   row       = output_index / rows;
  const int   col       = output_index % rows;
  const float threshold = rcond * s[0];
  float       result    = 0.0f;
  for (int k = 0; k < rank; ++k) {
    const float singular = s[k];
    if (singular > threshold)
      result += ut[row + k * cols] * (1.0f / singular) * vt[k + col * rank];
  }
  output[output_index] = result;
}

class Pinv : public holoflow::core::ISyncTask {
public:
  Pinv(PinvSettings settings, holoflow::core::TDesc idesc, int rows, int cols, int rank,
       cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), rows_(rows), cols_(cols),
        rank_(rank), stream_(stream) {
    CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle_, cols_, rows_, &workspace_elements_));
    workspace_  = curaii::make_unique_device_ptr<float>(static_cast<size_t>(workspace_elements_));
    info_       = curaii::make_unique_device_ptr<int>(1);
    input_copy_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(rows_) * cols_);
    singular_values_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(rank_));
    ut_              = curaii::make_unique_device_ptr<float>(static_cast<size_t>(cols_) * rank_);
    vt_              = curaii::make_unique_device_ptr<float>(static_cast<size_t>(rank_) * rows_);
  }

  ~Pinv() override {
    if (handle_ != nullptr)
      CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    const auto bytes = static_cast<size_t>(rows_) * cols_ * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(input_copy_.get(), ctx.inputs[0].data(), bytes,
                               cudaMemcpyDeviceToDevice, stream_));
    CUSOLVER_CHECK(cusolverDnSgesvd(handle_, 'S', 'S', cols_, rows_, input_copy_.get(), cols_,
                                    singular_values_.get(), ut_.get(), cols_, vt_.get(), rank_,
                                    workspace_.get(), workspace_elements_, nullptr, info_.get()));

    const int output_elements = cols_ * rows_;
    const int block           = 256;
    const int grid            = (output_elements + block - 1) / block;
    pinv_kernel<<<grid, block, 0, stream_>>>(ut_.get(), singular_values_.get(), vt_.get(),
                                             reinterpret_cast<float *>(ctx.outputs[0].data()),
                                             rows_, cols_, rank_, settings_.rcond);
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const PinvSettings          &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  PinvSettings                     settings_;
  holoflow::core::TDesc            idesc_;
  int                              rows_;
  int                              cols_;
  int                              rank_;
  int                              workspace_elements_;
  cudaStream_t                     stream_;
  cusolverDnHandle_t               handle_ = nullptr;
  curaii::unique_device_ptr<float> workspace_;
  curaii::unique_device_ptr<int>   info_;
  curaii::unique_device_ptr<float> input_copy_;
  curaii::unique_device_ptr<float> singular_values_;
  curaii::unique_device_ptr<float> ut_;
  curaii::unique_device_ptr<float> vt_;
};

} // namespace

holoflow::core::InferResult PinvFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc    = input_descs[0];
  const auto  settings = jsettings.get<PinvSettings>();
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32, "only F32 input is currently supported");
  check(idesc.shape.size() == 2, "only 2-D matrices are currently supported");
  check(utils::is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.shape[0] > 0 && idesc.shape[1] > 0, "matrix dimensions must be nonzero");
  check(std::isfinite(settings.rcond) && settings.rcond >= 0.0f,
        "rcond must be finite and nonnegative");
  check(idesc.shape[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            idesc.shape[1] <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "matrix dimensions exceed cuSOLVER limits");

  return {.input_descs   = {idesc},
          .output_descs  = {holoflow::core::TDesc({idesc.shape[1], idesc.shape[0]}, idesc.dtype,
                                                  holoflow::core::MemLoc::Device)},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
PinvFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto settings = jsettings.get<PinvSettings>();
  (void)infer(input_descs, jsettings);
  const int rows = static_cast<int>(input_descs[0].shape[0]);
  const int cols = static_cast<int>(input_descs[0].shape[1]);
  return std::make_unique<Pinv>(settings, input_descs[0], rows, cols, std::min(rows, cols),
                                ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
PinvFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old      = dynamic_cast<Pinv *>(old_task.get());
  const auto settings = jsettings.get<PinvSettings>();
  if (old != nullptr && old->settings() == settings && input_descs.size() == 1 &&
      utils::same_desc(input_descs[0], old->idesc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
