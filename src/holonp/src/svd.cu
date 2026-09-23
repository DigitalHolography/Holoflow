// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

#include "holonp/svd.hh"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "curaii/cuda.hh"
#include "curaii/cusolver.hh"
#include "utils/tensor_common.hh"

namespace holonp {

void to_json(nlohmann::json &j, const SVDSettings &s) { j = {{"full_matrices", s.full_matrices}}; }
void from_json(const nlohmann::json &j, SVDSettings &s) {
  if (j.is_null()) {
    s = {};
    return;
  }
  s.full_matrices = j.value("full_matrices", false);
}

namespace {

inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::invalid_argument("SVD: " + message);
}

class SVD : public holoflow::core::ISyncTask {
public:
  SVD(SVDSettings settings, holoflow::core::TDesc idesc, int rows, int cols, int rank,
      cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), rows_(rows), cols_(cols),
        rank_(rank), stream_(stream) {
    CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle_, cols_, rows_, &workspace_elements_));
    workspace_  = curaii::make_unique_device_ptr<float>(static_cast<size_t>(workspace_elements_));
    info_       = curaii::make_unique_device_ptr<int>(1);
    input_copy_ = curaii::make_unique_device_ptr<float>(static_cast<size_t>(rows_) * cols_);
  }

  ~SVD() override {
    if (handle_ != nullptr)
      CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream_));
    const auto bytes = static_cast<size_t>(rows_) * cols_ * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(input_copy_.get(), ctx.inputs[0].data(), bytes,
                               cudaMemcpyDeviceToDevice, stream_));
    auto *s  = reinterpret_cast<float *>(ctx.outputs[1].data());
    auto *u  = reinterpret_cast<float *>(ctx.outputs[0].data());
    auto *vh = reinterpret_cast<float *>(ctx.outputs[2].data());
    CUSOLVER_CHECK(cusolverDnSgesvd(handle_, 'S', 'S', cols_, rows_, input_copy_.get(), cols_, s,
                                    vh, cols_, u, rank_, workspace_.get(), workspace_elements_,
                                    nullptr, info_.get()));
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const SVDSettings           &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  SVDSettings                      settings_;
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
};

} // namespace

holoflow::core::InferResult SVDFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                              const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly one input");
  const auto &idesc    = input_descs[0];
  const auto  settings = jsettings.get<SVDSettings>();
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32, "only F32 input is currently supported");
  check(idesc.shape.size() == 2, "only 2-D matrices are currently supported");
  check(utils::is_c_contiguous(idesc), "input must be C-contiguous");
  check(idesc.shape[0] > 0 && idesc.shape[1] > 0, "matrix dimensions must be nonzero");
  check(!settings.full_matrices, "full_matrices=true is not yet supported");
  check(idesc.shape[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            idesc.shape[1] <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "matrix dimensions exceed cuSOLVER limits");

  const auto rows = static_cast<int>(idesc.shape[0]);
  const auto cols = static_cast<int>(idesc.shape[1]);
  const auto rank = std::min(rows, cols);
  return {.input_descs = {idesc},
          .output_descs =
              {holoflow::core::TDesc({static_cast<size_t>(rows), static_cast<size_t>(rank)},
                                     holoflow::core::DType::F32, holoflow::core::MemLoc::Device),
               holoflow::core::TDesc({static_cast<size_t>(rank)}, holoflow::core::DType::F32,
                                     holoflow::core::MemLoc::Device),
               holoflow::core::TDesc({static_cast<size_t>(rank), static_cast<size_t>(cols)},
                                     holoflow::core::DType::F32, holoflow::core::MemLoc::Device)},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false, false, false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
SVDFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                   const nlohmann::json                  &jsettings,
                   const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto inferred = infer(input_descs, jsettings);
  return std::make_unique<SVD>(jsettings.get<SVDSettings>(), input_descs[0],
                               static_cast<int>(input_descs[0].shape[0]),
                               static_cast<int>(input_descs[0].shape[1]),
                               static_cast<int>(inferred.output_descs[1].shape[0]), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
SVDFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                   std::span<const holoflow::core::TDesc>     input_descs,
                   const nlohmann::json                      &jsettings,
                   const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old      = dynamic_cast<SVD *>(old_task.get());
  const auto settings = jsettings.get<SVDSettings>();
  if (old != nullptr && old->settings() == settings && input_descs.size() == 1 &&
      utils::same_desc(input_descs[0], old->idesc())) {
    old->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
