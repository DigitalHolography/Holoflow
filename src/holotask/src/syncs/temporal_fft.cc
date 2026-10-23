// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/temporal_fft.hh"
#include "curaii/cufft.hh"
#include <limits>
#include <stdexcept>

namespace holotask::syncs {
using namespace holoflow::core;
namespace {
class TemporalFFT : public ISyncTask {
public:
  TemporalFFT(const TDesc &desc, cudaStream_t stream) : desc_(desc) {
    const int pixels    = static_cast<int>(desc.shape[1] * desc.shape[2]);
    int       size[]    = {static_cast<int>(desc.shape[0])};
    size_t    workspace = 0;
    CUFFT_CHECK(cufftMakePlanMany(plan_.get(), 1, size, size, pixels, 1, size, pixels, 1, CUFFT_C2C,
                                  pixels, &workspace));
    CUFFT_CHECK(cufftSetStream(plan_.get(), stream));
  }
  bool matches(const TDesc &desc) const {
    return desc.shape == desc_.shape && desc.strides == desc_.strides &&
           desc.dtype == desc_.dtype && desc.mem_loc == desc_.mem_loc;
  }
  void     set_stream(cudaStream_t stream) { CUFFT_CHECK(cufftSetStream(plan_.get(), stream)); }
  OpResult execute(SyncCtx &ctx) override {
    if (ctx.cancelled && ctx.cancelled->load())
      return OpResult::Cancelled;
    CUFFT_CHECK(cufftExecC2C(plan_.get(), reinterpret_cast<cufftComplex *>(ctx.inputs[0].data()),
                             reinterpret_cast<cufftComplex *>(ctx.outputs[0].data()),
                             CUFFT_FORWARD));
    return OpResult::Ok;
  }

private:
  TDesc               desc_;
  curaii::CufftHandle plan_;
};
} // namespace
InferResult TemporalFFTFactory::infer(std::span<const TDesc> inputs, const nlohmann::json &) const {
  if (inputs.size() != 1 || inputs[0].rank() != 3 || inputs[0].dtype != DType::CF32 ||
      inputs[0].mem_loc != MemLoc::Device)
    throw std::invalid_argument("TemporalFFT requires contiguous device CF32 [T,Y,X]");
  const auto &in     = inputs[0];
  size_t      stride = sizeof(cufftComplex);
  for (size_t axis = 3; axis-- > 0;) {
    if (in.shape[axis] == 0 || in.strides[axis] != stride ||
        in.shape[axis] >
            static_cast<size_t>(std::numeric_limits<int>::max()) / (stride / sizeof(cufftComplex)))
      throw std::invalid_argument("TemporalFFT input layout exceeds supported cuFFT dimensions");
    stride *= in.shape[axis];
  }
  return {.input_descs   = {in},
          .output_descs  = {TDesc(in.shape, in.dtype, in.mem_loc)},
          .in_place      = {{0, 0}},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> TemporalFFTFactory::create(std::span<const TDesc> inputs,
                                                      const nlohmann::json  &settings,
                                                      const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<TemporalFFT>(inputs[0], ctx.stream);
}
std::unique_ptr<ISyncTask> TemporalFFTFactory::update(std::unique_ptr<ISyncTask> old,
                                                      std::span<const TDesc>     inputs,
                                                      const nlohmann::json      &settings,
                                                      const SyncCreateCtx       &ctx) const {
  (void)infer(inputs, settings);
  if (auto *task = dynamic_cast<TemporalFFT *>(old.get()); task && task->matches(inputs[0])) {
    task->set_stream(ctx.stream);
    return old;
  }
  return create(inputs, settings, ctx);
}
} // namespace holotask::syncs
